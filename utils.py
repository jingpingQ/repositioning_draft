import hashlib, json, os, time, pathlib

def slugify(text: str) -> str:
    """Deterministic short hash, handy for cache keys / filenames."""
    return hashlib.md5(text.encode()).hexdigest()[:10]

def ensure_dir(path: str | pathlib.Path):
    os.makedirs(path, exist_ok=True)

