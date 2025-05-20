"""
Lightweight wrappers for whatever transformer checkpoints you load.
Keeps model code out of fetchers.py so startup is fast.
"""

from functools import cache
from transformers import AutoModel, AutoTokenizer

@cache
def load_biobert(model_name: str = "dmis-lab/biobert-v1.1") -> tuple:
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name)
    mdl.eval()
    return tok, mdl

