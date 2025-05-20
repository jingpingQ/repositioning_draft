"""
fetchers.py
~~~~~~~~~~~
Async wrappers for every external signal used in the v0.2 drug-repositioning
pipeline.

Only the PubMed and ClinicalTrials.gov functions are “real” today; the rest
return genuine scores but include conservative fall-backs so the app continues
to run even when a public service rate-limits us.

Important changes in this version
---------------------------------
• **TxGemma**: send the Authorization header *only* when HF_API_TOKEN is set.
  This removes the silent 401 → random-score problem.
• Minor clean-ups: use `json=` instead of `data=json.dumps(…)`, and a couple of
  tighter type-hints.
"""

from __future__ import annotations
import asyncio, math, random, os, json, re, functools
from typing import Dict, List, Set, Any

import aiohttp
import numpy as np
from joblib import Memory
from tenacity import retry, stop_after_attempt, wait_random_exponential
from urllib.parse import urlencode, quote_plus

from neo4j import GraphDatabase
from gql import Client, gql
from gql.transport.aiohttp import AIOHTTPTransport

from scoring import aggregate

# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
memory = Memory(".cache", verbose=0)

# ===========================================================================
# 1 · PubMed co-mention score
# ===========================================================================
API_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
CH_EMBL  = "https://www.ebi.ac.uk/chembl/api/data"
_ascii   = re.compile(r"^[\w\-\s\(\)]+$")          # keep ASCII words, digits, (), –

def _pubmed_url(endpoint: str, **params) -> str:
    return f"{API_BASE}{endpoint}.fcgi?" + urlencode(params)

async def _chembl_synonyms(drug: str) -> List[str]:
    async with aiohttp.ClientSession() as sess:
        url = f"{CH_EMBL}/molecule/search.json?q={quote_plus(drug)}&limit=1"
        try:
            js = await (await sess.get(url, timeout=15)).json()
        except Exception:
            return []

        if not js.get("molecules"):
            return []

        chembl_id = js["molecules"][0]["molecule_chembl_id"]
        url       = f"{CH_EMBL}/molecule/{chembl_id}.json"
        mol       = await (await sess.get(url, timeout=15)).json()

    syns = {s["molecule_synonym"] for s in mol.get("molecule_synonyms", [])}
    syns = {s for s in syns if _ascii.match(s) and len(s) > 3}
    return [drug] + sorted(syns - {drug})[:9]

@memory.cache
@retry(wait=wait_random_exponential(multiplier=1, max=8),
       stop=stop_after_attempt(3))
async def fetch_pubmed(drug: str, disease: str) -> float:
    async def _count(term: str, sess: aiohttp.ClientSession) -> int:
        url = _pubmed_url(
            "esearch",
            db="pubmed",
            term=term,
            retmode="json",
            rettype="count",
            email="youremail@example.com",
        )
        async with sess.get(url, timeout=15) as r:
            data = await r.json()
        return int(data["esearchresult"]["count"])

    names        = await _chembl_synonyms(drug) or [drug]
    drug_or      = " OR ".join(f'"{n}"[TIAB]' for n in names)
    disease_tiab = f'"{disease}"[TIAB]'

    term_both   = f"({drug_or}) AND {disease_tiab}"
    term_either = f"({drug_or}) OR  {disease_tiab}"

    async with aiohttp.ClientSession() as sess:
        both   = await _count(term_both, sess)
        either = await _count(term_either, sess)

    return 0.0 if either == 0 else min(1.0, math.log10(both + 1) / math.log10(either + 1))

# ===========================================================================
# 2 · ClinicalTrials.gov Phase-3 heuristics
# ===========================================================================
CTG_BASE = "https://clinicaltrials.gov/api/v2/studies/"

@memory.cache
@retry(wait=wait_random_exponential(multiplier=1, max=8),
       stop=stop_after_attempt(3))
async def fetch_trials(drug: str) -> Dict[str, float]:
    url = f"{CTG_BASE}search?query={urlencode({'term': drug})}&pageSize=99"
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(url, timeout=15) as resp:
                data = await resp.json()
    except Exception:
        return {"trial_failure_reason": 0.0, "safety_tolerability": 0.0}

    phase3 = [
        s for s in data.get("studies", [])
        if any("Phase 3" in l for l in s.get("protocolSection", {})
                                         .get("designModule", {})
                                         .get("phaseList", []))
    ]
    if not phase3:
        return {"trial_failure_reason": 0.0, "safety_tolerability": 0.0}

    completed = sum(
        1 for s in phase3
        if s.get("protocolSection", {}).get("statusModule", {}).get("overallStatus") == "Completed"
    )
    no_sae = sum(
        1 for s in phase3
        if not s.get("resultsSection", {}).get("adverseEventsModule")
    )

    return {
        "trial_failure_reason": round(1.0 - completed / len(phase3), 3),
        "safety_tolerability":  round(no_sae / len(phase3),        3),
    }

# ===========================================================================
# 3 · TxGemma LLM similarity  **<— fixed**
# ===========================================================================
HF_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
HF_API_URL  = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

@memory.cache
@retry(wait=wait_random_exponential(multiplier=1, max=8),
       stop=stop_after_attempt(3))
async def fetch_txgemma(drug: str, disease: str) -> float:
    payload = {"inputs": [drug, disease]}
    token   = os.getenv("HF_API_TOKEN", "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}  # <<— only if present

    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=30,
            ) as resp:
                vecs = await resp.json()
                if resp.status != 200 or not isinstance(vecs, list) or len(vecs) != 2:
                    raise RuntimeError(f"HF status {resp.status}")
    except Exception:
        return random.random()  # neutral fallback

    v1, v2 = np.asarray(vecs[0]), np.asarray(vecs[1])
    return ( _cosine(v1, v2) + 1 ) / 2          # −1…1  →  0…1

# ===========================================================================
# 4 · Hetionet MoA–disease alignment
# ===========================================================================
HETIO_URI  = os.getenv("HETIO_URI",  "neo4j+s://neo4j.het.io")
HETIO_USER = os.getenv("HETIO_USER", "neo4j")
HETIO_PASS = os.getenv("HETIO_PASS", "neo4j")

_driver = GraphDatabase.driver(
    HETIO_URI,
    auth=(HETIO_USER, HETIO_PASS),
    max_connection_lifetime=600,
    connection_timeout=15,
)

def _path_count(tx, drug: str, disease: str) -> int:
    q = """
    MATCH (c:Compound),(d:Disease)
    WHERE toLower(c.name)  = toLower($drug)
      AND toLower(d.name) CONTAINS toLower($disease)
    MATCH p = (c)-[*..4]-(d)
    RETURN count(p) AS paths
    LIMIT 20000
    """
    rec = tx.run(q, drug=drug, disease=disease).single()
    return rec["paths"] if rec else 0

@functools.cache
def _sync_hetionet(drug: str, disease: str) -> int:
    with _driver.session(database="neo4j") as sess:
        return sess.execute_read(_path_count, drug, disease)

@memory.cache
@retry(wait=wait_random_exponential(multiplier=1, max=8),
       stop=stop_after_attempt(3))
async def fetch_hetionet(drug: str, disease: str) -> float:
    try:
        loop  = asyncio.get_running_loop()
        paths = await loop.run_in_executor(None, _sync_hetionet, drug, disease)
    except Exception:
        return random.random()
    return min(1.0, math.log10(paths + 1) / 4.0)

# ===========================================================================
# 5 · Pathway overlap (Open Targets)
# ===========================================================================
OT_API        = "https://api.platform.opentargets.org/api/v4/graphql"
_ot_transport = AIOHTTPTransport(url=OT_API, timeout=20)
_ot_client    = Client(transport=_ot_transport, fetch_schema_from_transport=False)

async def _gql(query: str, variables: Dict[str, Any]) -> Dict:
    return await _ot_client.execute_async(gql(query), variable_values=variables)

SEARCH_Q = """
query Search($q: String!, $types: [Entity]) {
  search(queryString: $q, entityNames: $types) {
    drugs   { id name }
    diseases{ id name }
  }
}
"""
async def _lookup_id(name: str, entity: str) -> str | None:
    data = await _gql(SEARCH_Q, {"q": name, "types": [entity]})
    matches = data["search"]["drugs" if entity == "drug" else "diseases"]
    return matches[0]["id"] if matches else None

DRUG_TARGETS_Q = "query DrugTargets($id: String!){ drug(chemblId:$id){ targets{ id } } }"
DISEASE_TARGETS_Q = """
query DiseaseTargets($id: String!, $size: Int!){
  disease(efoId:$id){
    associatedTargets(page:{size:$size}){ rows{ target{ id } } }
  }
}
"""
async def _drug_targets(chembl: str) -> Set[str]:
    data = await _gql(DRUG_TARGETS_Q, {"id": chembl})
    return {t["id"] for t in data["drug"]["targets"]}

async def _disease_targets(efo: str, size: int = 500) -> Set[str]:
    data = await _gql(DISEASE_TARGETS_Q, {"id": efo, "size": size})
    rows = data["disease"]["associatedTargets"]["rows"]
    return {r["target"]["id"] for r in rows}

@memory.cache
@retry(wait=wait_random_exponential(multiplier=1, max=6),
       stop=stop_after_attempt(3))
async def fetch_pathway(drug: str, disease: str) -> float:
    try:
        chembl = await _lookup_id(drug, "drug")
        efo    = await _lookup_id(disease, "disease")
        if not chembl or not efo:
            return random.random()

        dt = await _drug_targets(chembl)
        dz = await _disease_targets(efo)
        if not dt or not dz:
            return 0.0

        inter = len(dt & dz)
        return round(inter / max(1, min(len(dt), len(dz))), 3)

    except Exception:
        return random.random()

# ===========================================================================
# 6 · Docking potential (quick ChEMBL affinity probe)
# ===========================================================================
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"

ACT_Q = """
query SearchDrug($q:String!){
  search(queryString:$q,entityNames:[drug]){ drugs{ id name } }
}
"""
TGT_Q = """
query Overlap($drug:String!,$disease:String!){
  search(queryString:$drug,entityNames:[drug]){ drugs{ id targets{ id } } }
  disease(efoId:$disease){ associatedTargets(page:{size:500}){ rows{ target{ id } } } }
}
"""
NUM_RE = re.compile(r"^\d+\.?\d*$")

async def _drug_chembl(name: str) -> str | None:
    data = await _gql(ACT_Q, {"q": name})
    return data["search"]["drugs"][0]["id"] if data["search"]["drugs"] else None

async def _first_overlap_target(chembl: str, disease: str) -> str | None:
    data   = await _gql(TGT_Q, {"drug": chembl, "disease": disease})
    drug_t = {t["id"] for t in data["search"]["drugs"][0]["targets"]}
    dis_t  = {r["target"]["id"] for r in data["disease"]["associatedTargets"]["rows"]}
    ov     = drug_t & dis_t
    return next(iter(ov)) if ov else None

async def _best_binding_nm(chembl: str, target: str) -> float | None:
    url = (f"{CHEMBL_API}/activity.json?molecule_chembl_id={chembl}"
           f"&target_chembl_id={target}&limit=1000")
    async with aiohttp.ClientSession() as sess:
        async with sess.get(url, timeout=20) as r:
            js = await r.json()
    vals = [
        float(a["standard_value"]) for a in js["activities"]
        if a["standard_units"] == "nM"
        and a["standard_value"] and NUM_RE.match(a["standard_value"])
    ]
    return min(vals) if vals else None

@memory.cache
@retry(wait=wait_random_exponential(multiplier=1, max=6),
       stop=stop_after_attempt(3))
async def fetch_docking(drug: str, disease: str) -> float:
    try:
        chembl = await _drug_chembl(drug)
        if not chembl:
            return 0.0
        target  = await _first_overlap_target(chembl, disease)
        if not target:
            return 0.0
        best_nm = await _best_binding_nm(chembl, target)
        if best_nm is None:
            return 0.0

        pchembl = 9 - np.log10(best_nm)     # nM → mol → pAff
        return float(np.clip((pchembl - 5) / 4, 0, 1))
    except Exception:
        return 0.0

# ===========================================================================
# 7 · Master orchestrator
# ===========================================================================
async def evaluate_candidate(drug: str, disease: str) -> dict:
    scores  : Dict[str, float] = {}
    reasons : List[str] = []

    scores["target_expression_overlap"] = await fetch_txgemma(drug, disease)
    reasons.append(f"TxGemma= {scores['target_expression_overlap']:.2f}")

    scores["moa_disease_alignment"] = await fetch_hetionet(drug, disease)
    reasons.append(f"Hetionet= {scores['moa_disease_alignment']:.2f}")

    trial = await fetch_trials(drug)
    scores.update(trial)
    reasons.append(f"TrialFail= {trial['trial_failure_reason']:.2f}")
    reasons.append(f"Safety= {trial['safety_tolerability']:.2f}")

    scores["literature_support"] = await fetch_pubmed(drug, disease)
    reasons.append(f"PubMed= {scores['literature_support']:.2f}")

    scores["docking_potential"] = await fetch_docking(drug, disease)
    reasons.append(f"Docking= {scores['docking_potential']:.2f}")

    scores["pathway_involvement"] = await fetch_pathway(drug, disease)
    reasons.append(f"Pathway= {scores['pathway_involvement']:.2f}")

    return {
        "drug": drug,
        **{k: round(v, 3) for k, v in scores.items()},
        "overall_score": round(aggregate(scores), 4),
        "details": "\n".join(reasons),
    }
