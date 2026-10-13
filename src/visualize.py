"""Visualization and error analysis utilities for Ground Truth vs Predictions.
Generates side-by-side or annotated comparison figures showing:
  - True Positives (high IoU match)
  - False Positives (hallucinated box)
  - False Negatives (missed road damage)
  - Multi-box scenarios
  - Subtle/difficult crack detection
  - Class confusion (e.g., pothole misclassified as manhole)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from scripts.score import iou_xyxy, parse_output
from src.grounding import box_to_pixel

# Distinct palettes for Ground Truth vs Model Prediction
GT_COLOR = "#00E676"     # Bright Green for GT
PRED_COLOR = "#FF1744"   # Bright Red/Crimson for Prediction
MATCH_COLOR = "#00E5FF"  # Cyan for matched


def draw_side_by_side_comparison(
    image: Image.Image,
    gt_boxes: List[Dict[str, Any]],
    pred_boxes: List[Dict[str, Any]],
    title: str = "",
    subtitle: str = "",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Render side-by-side comparison figure: Left is Ground Truth, Right is Model Prediction."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    w, h = image.size

    # Left: Ground Truth
    axes[0].imshow(image)
    axes[0].set_title(f"Ground Truth ({len(gt_boxes)} boxes)", fontsize=13, fontweight="bold", color="#1B5E20")
    axes[0].axis("off")

    for obj in gt_boxes:
        label = obj.get("label", "")
        bb = obj.get("bbox_2d") or obj.get("bbox")
        if bb is None:
            continue
        x1, y1, x2, y2 = box_to_pixel(bb, width=w, height=h)
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=GT_COLOR, facecolor="none", linestyle="-"
        )
        axes[0].add_patch(rect)
        axes[0].text(
            x1, max(12, y1 - 4), f"GT: {label}",
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(facecolor=GT_COLOR, edgecolor="none", pad=2, alpha=0.9)
        )

    # Right: Prediction
    axes[1].imshow(image)
    axes[1].set_title(f"Prediction ({len(pred_boxes)} boxes)", fontsize=13, fontweight="bold", color="#B71C1C")
    axes[1].axis("off")

    for obj in pred_boxes:
        label = obj.get("label", "")
        bb = obj.get("bbox_2d") or obj.get("bbox")
        if bb is None:
            continue
        x1, y1, x2, y2 = box_to_pixel(bb, width=w, height=h)
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor=PRED_COLOR, facecolor="none", linestyle="--"
        )
        axes[1].add_patch(rect)
        axes[1].text(
            x1, max(12, y1 - 4), f"Pred: {label}",
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(facecolor=PRED_COLOR, edgecolor="none", pad=2, alpha=0.9)
        )

    if title:
        fig.suptitle(f"{title}\n{subtitle}", fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, bbox_inches="tight")
        print(f"Saved comparison figure to {p}")

    return fig
