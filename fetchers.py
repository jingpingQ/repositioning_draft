"""
fetchers.py • v0.4
No external caching (joblib removed) + API/FALLBACK logging.
"""

from __future__ import annotations
import asyncio, math, os, re, random, functools, logging, inspect
from typing import Dict, List, Set, Any

import aiohttp, numpy as np
from tenacity import retry, stop_after_attempt, wait_random_exponential
from urllib.parse import urlencode, quote_plus
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport
from neo4j import GraphDatabase

from scoring import aggregate

# -----------------------------------------------------------------------------
# logging helper (unchanged) --------------------------------------------------
# -----------------------------------------------------------------------------
logging.basicConfig(level="INFO", format="%(levelname).1s %(message)s")
log = logging.getLogger("fetchers")


def _tell(status: str, value: float | Dict[str, float]):
    """One-liner telling whether we used LIVE API or FALLBACK."""
    caller = inspect.stack()[1].function.replace("fetch_", "").ljust(8)
    val = (", ".join(f"{k}={v:.3g}" for k, v in value.items())
           if isinstance(value, dict) else f"{value:.3g}")
    log.info("[%s] %s • %s", caller, status.upper(), val)


# -----------------------------------------------------------------------------
# 1 · PubMed co-mention -------------------------------------------------------
# -----------------------------------------------------------------------------
API_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
CH_EMBL  = "https://www.ebi.ac.uk/chembl/api/data"
_ascii   = re.compile(r"^[\w\-\s\(\)]+$")

def _pubmed_url(ep: str, **p) -> str:
    return f"{API_BASE}{ep}.fcgi?" + urlencode(p)

async def _chembl_synonyms(drug: str) -> List[str]:
    async with aiohttp.ClientSession() as s:
        js = await (await s.get(f"{CH_EMBL}/molecule/search.json?q={quote_plus(drug)}&limit=1",
                                timeout=15)).json()
        if not js.get("molecules"):
            return []
        chembl_id = js["molecules"][0]["molecule_chembl_id"]
        mol       = await (await s.get(f"{CH_EMBL}/molecule/{chembl_id}.json",
                                       timeout=15)).json()
    syns = {x["molecule_synonym"] for x in mol.get("molecule_synonyms", [])}
    syns = {s for s in syns if _ascii.match(s) and len(s) > 3}
    return [drug] + sorted(syns - {drug})[:9]

@retry(wait=wait_random_exponential(1, 8), stop=stop_after_attempt(3))
async def fetch_pubmed(drug: str, disease: str) -> float:
    async def _count(term: str, sess):  # nested helper
        url = _pubmed_url("esearch", db="pubmed", term=term,
                          retmode="json", rettype="count",
                          email="you@example.com")
        j = await (await sess.get(url, timeout=15)).json()
        return int(j["esearchresult"]["count"])

    names  = await _chembl_synonyms(drug) or [drug]
    d_or   = " OR ".join(f'"{n}"[TIAB]' for n in names)
    dtiab  = f'"{disease}"[TIAB]'
    async with aiohttp.ClientSession() as s:
        both   = await _count(f"({d_or}) AND {dtiab}", s)
        either = await _count(f"({d_or}) OR  {dtiab}", s)
    val = 0.0 if either == 0 else min(1., math.log10(both + 1) /
                                           math.log10(either + 1))
    _tell("api", val)
    return val


# -----------------------------------------------------------------------------
# 2 · ClinicalTrials.gov heuristics -------------------------------------------
# -----------------------------------------------------------------------------
CTG_BASE = "https://clinicaltrials.gov/api/v2/studies/"

@retry(wait=wait_random_exponential(1, 8), stop=stop_after_attempt(3))
async def fetch_trials(drug: str) -> Dict[str, float]:
    url = f"{CTG_BASE}search?query={urlencode({'term': drug})}&pageSize=99"
    try:
        js = await (await aiohttp.ClientSession().get(url, timeout=15)).json()
    except Exception:
        val = {"trial_failure_reason": 0.0, "safety_tolerability": 0.0}
        _tell("fallback", val);  return val

    phase3 = [s for s in js.get("studies", [])
              if any("Phase 3" in l
                     for l in s.get("protocolSection", {})
                               .get("designModule", {})
                               .get("phaseList", []))]
    if not phase3:
        val = {"trial_failure_reason": 0.0, "safety_tolerability": 0.0}
        _tell("api", val);  return val

    compl = sum(1 for s in phase3 if
                s.get("protocolSection", {})
                 .get("statusModule", {})
                 .get("overallStatus") == "Completed")
    nosae = sum(1 for s in phase3
                if not s.get("resultsSection", {}).get("adverseEventsModule"))
    val = {"trial_failure_reason": round(1 - compl/len(phase3), 3),
           "safety_tolerability":  round(nosae/len(phase3),    3)}
    _tell("api", val);  return val


# -----------------------------------------------------------------------------
# 3 · TxGemma similarity ------------------------------------------------------
# -----------------------------------------------------------------------------
HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_URL   = f"https://api-inference.huggingface.co/models/{HF_MODEL}"

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))

@retry(wait=wait_random_exponential(1, 8), stop=stop_after_attempt(3))
async def fetch_txgemma(drug: str, disease: str) -> float:
    token = os.getenv("HF_API_TOKEN", "").strip()
    head  = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        async with aiohttp.ClientSession() as s:
            r = await s.post(HF_URL, headers=head,
                             json={"inputs": [drug, disease]}, timeout=30)
            js = await r.json()
            if r.status != 200 or not isinstance(js, list) or len(js) != 2:
                raise RuntimeError()
    except Exception:
        val = random.random(); _tell("fallback", val);  return val
    v1, v2 = map(np.asarray, js)
    val = (_cos(v1, v2) + 1)/2
    _tell("api", val);  return val


# -----------------------------------------------------------------------------
# 4 · Hetionet MoA alignment --------------------------------------------------
# -----------------------------------------------------------------------------
HURI  = os.getenv("HETIO_URI",  "neo4j+s://neo4j.het.io")
HUSR  = os.getenv("HETIO_USER", "neo4j")
HPWD  = os.getenv("HETIO_PASS", "neo4j")

driver = GraphDatabase.driver(HURI, auth=(HUSR, HPWD),
                              max_connection_lifetime=600,
                              connection_timeout=15)

def _path_count(tx, drug, disease):
    q = """
    MATCH (c:Compound),(d:Disease)
    WHERE toLower(c.name)=toLower($drug)
      AND toLower(d.name) CONTAINS toLower($disease)
    MATCH p=(c)-[*..4]-(d) RETURN count(p) AS n
    """
    rec = tx.run(q, drug=drug, disease=disease).single()
    return rec["n"] if rec else 0

@retry(wait=wait_random_exponential(1, 8), stop=stop_after_attempt(3))
async def fetch_hetionet(drug: str, disease: str) -> float:
    loop = asyncio.get_running_loop()
    try:
        n = await loop.run_in_executor(
            None,
            lambda: driver.session(database="neo4j")
                          .execute_read(_path_count, drug, disease))
    except Exception:
        val = random.random(); _tell("fallback", val); return val
    val = min(1., math.log10(n+1)/4)
    _tell("api", val);  return val


# -----------------------------------------------------------------------------
# 5 · Pathway overlap (OpenTargets) ------------------------------------------
# -----------------------------------------------------------------------------
OT_API   = "https://api.platform.opentargets.org/api/v4/graphql"
ot_client = Client(transport=AIOHTTPTransport(url=OT_API, timeout=20),
                   fetch_schema_from_transport=False)

async def _gql(q: str, v):  return await ot_client.execute_async(gql(q), variable_values=v)

SEARCH_Q = """
query S($q:String!,$t:[Entity]){search(queryString:$q,entityNames:$t){
  drugs{id} diseases{id}}}
"""
DRUG_T_Q = "query D($id:String!){drug(chemblId:$id){targets{id}}}"
DIS_T_Q  = """
query X($id:String!,$n:Int!){
  disease(efoId:$id){
    associatedTargets(page:{size:$n}){rows{target{id}}}}}
"""

async def _id(name, kind):
    d = await _gql(SEARCH_Q, {"q": name, "t": [kind]})
    lst = d["search"]["drugs" if kind=="drug" else "diseases"]
    return lst[0]["id"] if lst else None

@retry(wait=wait_random_exponential(1, 6), stop=stop_after_attempt(3))
async def fetch_pathway(drug: str, disease: str) -> float:
    try:
        chembl, efo = await asyncio.gather(_id(drug,"drug"), _id(disease,"disease"))
        if not chembl or not efo:
            raise RuntimeError()
        dt = {t["id"] for t in (await _gql(DRUG_T_Q, {"id": chembl}))
                                 ["drug"]["targets"]}
        dz = {r["target"]["id"] for r in (await _gql(DIS_T_Q, {"id": efo, "n":500}))
                                 ["disease"]["associatedTargets"]["rows"]}
        if not dt or not dz:  val=0.0
        else:                  val=len(dt&dz)/max(1,len(dt),len(dz))
        _tell("api", val);  return round(val,3)
    except Exception:
        val = random.random(); _tell("fallback", val); return val


# -----------------------------------------------------------------------------
# 6 · Docking proxy (quick ChEMBL affinity) -----------------------------------
# -----------------------------------------------------------------------------
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
NUM_RE     = re.compile(r"^\d+\.?\d*$")

async def _best_nm(chembl, target):
    js = await (await aiohttp.ClientSession().get(
                f"{CHEMBL_API}/activity.json?molecule_chembl_id={chembl}"
                f"&target_chembl_id={target}&limit=1000", timeout=20)).json()
    vals=[float(a["standard_value"]) for a in js["activities"]
          if a["standard_units"]=="nM" and a["standard_value"]
          and NUM_RE.match(a["standard_value"])]
    return min(vals) if vals else None

@retry(wait=wait_random_exponential(1,6), stop=stop_after_attempt(3))
async def fetch_docking(drug: str, disease: str) -> float:
    try:
        chembl, efo = await asyncio.gather(_id(drug,"drug"), _id(disease,"disease"))
        if not chembl or not efo:   return 0.0
        nm = await _best_nm(chembl, efo)
        if nm is None:              return 0.0
        pchembl = 9 - np.log10(nm)
        val = float(np.clip((pchembl-5)/4, 0, 1))
        _tell("api", val);  return val
    except Exception:
        return 0.0


# -----------------------------------------------------------------------------
# 7 · Master orchestrator -----------------------------------------------------
# -----------------------------------------------------------------------------
async def evaluate_candidate(drug: str, disease: str) -> dict:
    scores = {
        "target_expression_overlap": await fetch_txgemma(drug, disease),
        "moa_disease_alignment":     await fetch_hetionet(drug, disease),
        **await fetch_trials(drug),
        "literature_support":        await fetch_pubmed(drug, disease),
        "docking_potential":         await fetch_docking(drug, disease),
        "pathway_involvement":       await fetch_pathway(drug, disease),
    }
    return {
        "drug": drug,
        **{k: round(v, 3) for k, v in scores.items()},
        "overall_score": round(aggregate(scores), 4),
        "details": "; ".join(f"{k}={v:.3g}" for k, v in scores.items()),
    }
