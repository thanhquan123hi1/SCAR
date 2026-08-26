"""Rule-based probability decoding for One-vs-Rest medical segmentation."""

from __future__ import annotations

from typing import Any

try:
    import torch
except ImportError:
    torch = None


# Default challenge postprocessing rules per view (Standard unblended channels)
DEFAULT_POSTPROCESS_RULES: dict[str, list[dict[str, Any]]] = {
    "2ch": [
        {"class_id": 1, "threshold": 0.50, "terms": [{"channel": 0, "weight": 1.0}], "priority": 1},
        {"class_id": 2, "threshold": 0.50, "terms": [{"channel": 1, "weight": 1.0}], "priority": 2},
        {"class_id": 3, "threshold": 0.50, "terms": [{"channel": 2, "weight": 1.0}], "priority": 3},
    ],
    "4ch": [
        {"class_id": 1, "threshold": 0.50, "terms": [{"channel": 0, "weight": 1.0}], "priority": 1},
        {"class_id": 4, "threshold": 0.50, "terms": [{"channel": 3, "weight": 1.0}], "priority": 2},
        {"class_id": 2, "threshold": 0.50, "terms": [{"channel": 1, "weight": 1.0}], "priority": 3},
        {"class_id": 3, "threshold": 0.50, "terms": [{"channel": 2, "weight": 1.0}], "priority": 4},
    ],
    "sa": [
        {"class_id": 1, "threshold": 0.50, "terms": [{"channel": 0, "weight": 1.0}], "priority": 1},
        {"class_id": 4, "threshold": 0.50, "terms": [{"channel": 3, "weight": 1.0}], "priority": 2},
        {"class_id": 2, "threshold": 0.50, "terms": [{"channel": 1, "weight": 1.0}], "priority": 3},
        {"class_id": 3, "threshold": 0.50, "terms": [{"channel": 2, "weight": 1.0}], "priority": 4},
    ],
    "ras": [
        {"class_id": 1, "threshold": 0.50, "terms": [{"channel": 0, "weight": 1.0}], "priority": 1},
    ],
}


def _score_from_terms(prob_maps: torch.Tensor, terms: list[dict[str, Any]]) -> torch.Tensor:
    dims = (prob_maps.shape[0],) + tuple(prob_maps.shape[2:])
    score = torch.zeros(dims, dtype=prob_maps.dtype, device=prob_maps.device)
    for term in terms:
        channel = int(term["channel"])
        if channel < prob_maps.shape[1]:
            weight = float(term.get("weight", 1.0))
            score = score + weight * prob_maps[:, channel]
    return score


def decode_with_rules(
    fg_logits: torch.Tensor,
    rules: list[dict[str, Any]] | None = None,
    view: str | None = None,
    activation: str = "sigmoid",
) -> torch.Tensor:
    """Convert independent foreground logits into multi-class integer labels via priority rules.
    
    Args:
        fg_logits: Tensor of shape (B, K, H, W) or (B, K, D, H, W), where K = num_classes - 1.
        rules: List of rule dicts with keys: class_id, threshold, terms, priority.
        view: View identifier ('2ch', '4ch', 'sa', 'ras') to fetch default rules if rules is None.
        activation: 'sigmoid' for One-vs-Rest heads (default), 'softmax' for multi-class CE heads.
        
    Returns:
        pred: Tensor of shape (B, H, W) or (B, D, H, W) with int64 class labels.
    """
    if rules is None and view is not None:
        view_key = view.lower().replace("sax", "sa").replace("lge_", "")
        rules = DEFAULT_POSTPROCESS_RULES.get(view_key, None)

    if activation == "softmax":
        prob_maps = torch.softmax(fg_logits, dim=1)
    else:
        prob_maps = torch.sigmoid(fg_logits)
    spatial_shape = (prob_maps.shape[0],) + tuple(prob_maps.shape[2:])
    pred = torch.zeros(spatial_shape, dtype=torch.long, device=prob_maps.device)

    if not rules:
        # Fallback to simple thresholding per channel
        for k in range(prob_maps.shape[1]):
            class_id = k + 1
            pred = torch.where(prob_maps[:, k] > 0.5, torch.full_like(pred, class_id), pred)
        return pred

    ordered = sorted(rules, key=lambda x: int(x.get("priority", 999)))
    for rule in ordered:
        class_id = int(rule["class_id"])
        threshold = float(rule["threshold"])
        terms = rule.get("terms", [])
        if not terms:
            continue
        score = _score_from_terms(prob_maps, terms)
        pred = torch.where(score > threshold, torch.full_like(pred, class_id), pred)

    return pred
