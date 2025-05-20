WEIGHTS = {
    "target_expression_overlap": 0.20,
    "moa_disease_alignment":     0.10,   
    "literature_support":        0.30,   
    "docking_potential":         0.10,
    "pathway_involvement":       0.05,
}

def _normalize(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))

def aggregate(component_scores: dict[str, float]) -> float:
    """Return weighted 0-1 repositioning score."""
    return sum(WEIGHTS[k] * _normalize(component_scores.get(k, 0.0))
               for k in WEIGHTS)
