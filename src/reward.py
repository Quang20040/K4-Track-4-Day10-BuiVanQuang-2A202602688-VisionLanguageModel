"""Verifiable IoU-based Reward Function for Grounding Reinforcement Learning (GRPO - E5).
Implements deterministic, verifiable bipartite greedy matching matching the rubric and guide.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from scripts.score import iou_xyxy, parse_output


def entry_to_gt_boxes(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract ground truth boxes on 0-1000 scale from a dataset entry.

    Args:
        entry: Dataset JSONL row with 'objects' list.

    Returns:
        List of dicts: [{"label": str, "bbox": (x1, y1, x2, y2)}]
    """
    out = []
    for o in entry.get("objects", []):
        xc, yc, w, h = o["yolo"]
        out.append({
            "label": o["label"],
            "bbox": (
                (xc - w / 2.0) * 1000.0,
                (yc - h / 2.0) * 1000.0,
                (xc + w / 2.0) * 1000.0,
                (yc + h / 2.0) * 1000.0,
            ),
        })
    return out


def compute_grounding_iou_reward(
    completion: Optional[str],
    gt_boxes: List[Dict[str, Any]],
    iou_thr: float = 0.5,
) -> float:
    """Compute verifiable reward in [0.0, 1.0] for a single model completion against ground truth boxes.

    Reward formulation:
      - Invalid JSON syntax: 0.0
      - Both GT and prediction empty: 1.0
      - Otherwise: 0.2 (valid JSON bonus) + 0.8 * (total_matched_iou) / max(n_gt, n_pred, 1)
        where matching is greedy on descending IoU for the same class with IoU >= iou_thr.

    Penalizes:
      - Hallucinated extra boxes (via max(n_gt, n_pred) denominator)
      - Missing boxes (via max(n_gt, n_pred) denominator)
      - Class mismatch (requires same label)
      - Inaccurate localization (weight scaled by actual IoU)
      - Format violations (immediate 0.0)

    Args:
        completion: Raw generated text from model.
        gt_boxes: List of dicts [{"label": label, "bbox": (x1, y1, x2, y2)}].
        iou_thr: Minimum IoU threshold to consider a match valid (default: 0.5).

    Returns:
        float reward in [0.0, 1.0].
    """
    pred = parse_output(completion)
    if pred is None:
        return 0.0  # Malformed JSON or schema violation

    n_gt = len(gt_boxes)
    n_pred = len(pred)

    if n_gt == 0 and n_pred == 0:
        return 1.0

    # Perform greedy matching per class
    classes = {g["label"] for g in gt_boxes} | {p["label"] for p in pred}
    total_matched_iou = 0.0

    for lab in classes:
        g_boxes = [g["bbox"] for g in gt_boxes if g["label"] == lab]
        p_boxes = [p["bbox"] for p in pred if p["label"] == lab]

        if not g_boxes or not p_boxes:
            continue

        # All pairs sorted descending by IoU
        pairs = sorted(
            (
                (iou_xyxy(gb, pb), gi, pi)
                for gi, gb in enumerate(g_boxes)
                for pi, pb in enumerate(p_boxes)
            ),
            key=lambda x: x[0],
            reverse=True,
        )

        used_g = set()
        used_p = set()

        for iou, gi, pi in pairs:
            if iou < iou_thr:
                break
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)
            total_matched_iou += iou

    # Reward scale: 0.2 for valid JSON structure + 0.8 * precision/recall scaled IoU
    denom = max(n_gt, n_pred, 1)
    reward = 0.2 + 0.8 * (total_matched_iou / denom)
    return max(0.0, min(1.0, float(reward)))
