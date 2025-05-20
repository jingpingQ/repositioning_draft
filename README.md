 Drug-Repositioning Space 

A single‑page Gradio web‑app lets you paste one or more drugs that stalled in Phase 3 and a disease you care about. The backend queries a handful of open biomedical APIs, scores each drug on several orthogonal signals of repurposing potential, then returns a ranked table.

## Quick start

```bash
# 1 — clone / download the repo
pip install -r requirements.txt          # Python ≥3.10 recommended

# 2 — (optional) set creds for private APIs
export HF_API_TOKEN="<your‑HF‑token>"   # speeds up embeddings
export HETIO_URI=… HETIO_USER=… HETIO_PASS=…   # if you host your own Hetionet

# 3 — run the demo
python app.py                            # opens http://127.0.0.1:7860

```



## Structure

```
app.py            # Gradio UI and orchestration
fetchers.py       # **all external API calls + fallbacks**
scoring.py        # component → weight map and aggregator
requirements.txt  # Python deps (≈ 250 MB once transformers are cached)
```

fetchers.py is the work‑horse; every fetch_*() function returns either a float in  or a small dict of floats.  All heavy network work is async; each call retries with exponential back‑off and logs whether it hit the live API or a fallback.



##Calculation

```
target_expression_overlap : TxGemma surrogate
moa_disease_alignment : Hetionet v1.0 public Neo4j
trial_failure_reason : ClinicalTrials.gov v2
safety_tolerability : ClinicalTrials.gov
literature_support : PubMed E‑utilities + ChEMBL synonyms
docking_potential : ChEMBL bioactivity table
pathway_involvement : OpenTargets v4 GraphQL + REST fallback

Weighted sum of the above
`
```
