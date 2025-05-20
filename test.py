python - <<'PY'
import requests, urllib.parse, json, math
drug="teplizumab"; disease="type 1 diabetes mellitus"

def chembl_syns(d):
    js=requests.get(f"https://www.ebi.ac.uk/chembl/api/data/molecule/search.json?q={urllib.parse.quote(d)}&limit=1").json()
    if not js["molecules"]: return []
    mol=requests.get(f"https://www.ebi.ac.uk/chembl/api/data/molecule/{js['molecules'][0]['molecule_chembl_id']}.json").json()
    return [d]+[s["molecule_synonym"] for s in mol["molecule_synonyms"]][:10]

def esearch(term):
    u="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    q={"db":"pubmed","term":term,"retmode":"json","rettype":"count"}
    return int(requests.get(u,params=q).json()["esearchresult"]["count"])

names = chembl_syns(drug)
drug_or=" OR ".join(f'"{n}"[TIAB]' for n in names)
both=esearch(f"({drug_or}) AND \"{disease}\"[TIAB]")
either=esearch(f"({drug_or}) OR  \"{disease}\"[TIAB]")
print("Synonyms:", names)
print("Both:", both, "Either:", either)
print("Score:", 0 if either==0 else round(math.log10(both+1)/math.log10(either+1),3))
PY

