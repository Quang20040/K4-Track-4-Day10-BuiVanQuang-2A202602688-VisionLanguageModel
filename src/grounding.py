"""Grounding transformations, format conversions, and box drawing utilities.
Handles conversions between YOLO format (normalized [0,1], center x, center y, w, h)
and Qwen3-VL Grounding JSON format (0-1000 scale [x1, y1, x2, y2]).
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont

CLASSES = ("pothole", "crack", "manhole")

CLASS_COLORS = {
    "pothole": "#FF3838",  # Red
    "crack": "#FF9D2E",    # Orange
    "manhole": "#2ECC71",  # Green
    "other": "#9B59B6",    # Purple
}

DEFAULT_PROMPT = (
    "Locate road damage in this image: pothole, crack, and manhole. "
    "Provide coordinates on a 0-1000 scale formatted as a JSON list of objects "
    "with 'label' and 'bbox_2d': [x1, y1, x2, y2]. If no road damage is present, output []."
)


def yolo_to_box(label: str, yolo: Union[List[float], Tuple[float, ...]]) -> Dict[str, Any]:
    """Convert YOLO format [xc, yc, w, h] (normalized [0, 1]) to grounding box [x1, y1, x2, y2] (0-1000 scale).

    Args:
        label: Class name ('pothole', 'crack', 'manhole').
        yolo: 4-tuple or list [center_x, center_y, width, height] in [0, 1].

    Returns:
        Dict: {"label": label, "bbox_2d": [x1, y1, x2, y2]}
    """
    xc, yc, w, h = yolo
    x1 = (xc - w / 2.0) * 1000.0
    y1 = (yc - h / 2.0) * 1000.0
    x2 = (xc + w / 2.0) * 1000.0
    y2 = (yc + h / 2.0) * 1000.0

    # Round to integer and clamp to [0, 1000]
    rx1 = int(round(max(0.0, min(1000.0, x1))))
    ry1 = int(round(max(0.0, min(1000.0, y1))))
    rx2 = int(round(max(0.0, min(1000.0, x2))))
    ry2 = int(round(max(0.0, min(1000.0, y2))))

    # Ensure strictly x1 < x2 and y1 < y2 to prevent degenerate boxes
    if rx2 <= rx1:
        if rx2 < 1000:
            rx2 = rx1 + 1
        else:
            rx1 = max(0, rx2 - 1)

    if ry2 <= ry1:
        if ry2 < 1000:
            ry2 = ry1 + 1
        else:
            ry1 = max(0, ry2 - 1)

    return {"label": label, "bbox_2d": [rx1, ry1, rx2, ry2]}


def objects_to_answer(objects: List[Dict[str, Any]]) -> str:
    """Convert list of YOLO object dicts into standard canonical grounding JSON string.
    Boxes are sorted deterministically top-to-bottom, left-to-right (y1, x1) to stabilize sequence generation.

    Args:
        objects: List of dicts, each with keys 'label' and 'yolo'.

    Returns:
        JSON string representation: e.g. '[{"label":"crack","bbox_2d":[100,200,300,400]}]'
    """
    boxes = [yolo_to_box(obj["label"], obj["yolo"]) for obj in objects]
    # Stable canonical sorting by (y1, x1)
    boxes = sorted(boxes, key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"))


def box_to_pixel(
    bbox_2d: Union[List[Union[int, float]], Tuple[Union[int, float], ...]],
    width: int = 640,
    height: int = 360,
) -> Tuple[float, float, float, float]:
    """Convert grounding box [x1, y1, x2, y2] (0-1000 scale) to image pixel coordinates.

    Args:
        bbox_2d: 4 numbers in [0, 1000].
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        (x1_px, y1_px, x2_px, y2_px)
    """
    x1 = bbox_2d[0] * width / 1000.0
    y1 = bbox_2d[1] * height / 1000.0
    x2 = bbox_2d[2] * width / 1000.0
    y2 = bbox_2d[3] * height / 1000.0
    return (x1, y1, x2, y2)


def pixel_to_box(
    bbox_px: Union[List[float], Tuple[float, float, float, float]],
    width: int = 640,
    height: int = 360,
    label: str = "",
) -> Dict[str, Any]:
    """Convert pixel coordinates to grounding box (0-1000 scale).

    Args:
        bbox_px: (x1, y1, x2, y2) in pixels.
        width: Image width in pixels.
        height: Image height in pixels.
        label: Optional label string.

    Returns:
        Dict: {"label": label, "bbox_2d": [x1, y1, x2, y2]}
    """
    x1 = int(round(max(0.0, min(1000.0, bbox_px[0] / width * 1000.0))))
    y1 = int(round(max(0.0, min(1000.0, bbox_px[1] / height * 1000.0))))
    x2 = int(round(max(0.0, min(1000.0, bbox_px[2] / width * 1000.0))))
    y2 = int(round(max(0.0, min(1000.0, bbox_px[3] / height * 1000.0))))
    if x2 <= x1:
        x2 = min(1000, x1 + 1)
    if y2 <= y1:
        y2 = min(1000, y1 + 1)
    res: Dict[str, Any] = {"bbox_2d": [x1, y1, x2, y2]}
    if label:
        res["label"] = label
    return res


def draw_boxes_on_image(
    image: Image.Image,
    boxes: List[Dict[str, Any]],
    coord: str = "norm1000",
    color_map: Optional[Dict[str, str]] = None,
    line_width: int = 3,
    show_labels: bool = True,
) -> Image.Image:
    """Draw bounding boxes and class labels on a PIL Image.

    Args:
        image: PIL Image object.
        boxes: List of box dicts. Can be format {"label": ..., "bbox_2d": [x1,y1,x2,y2]}
               or {"label": ..., "bbox": (x1,y1,x2,y2)}.
        coord: Coordinate system: 'norm1000' (0-1000) or 'pixel'.
        color_map: Mapping from class label to hex/rgb color string.
        line_width: Outline thickness.
        show_labels: Whether to render label text.

    Returns:
        A new PIL Image with drawn boxes.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    colors = color_map or CLASS_COLORS

    for item in boxes:
        label = item.get("label", "unknown")
        raw_box = item.get("bbox_2d") if "bbox_2d" in item else item.get("bbox")
        if raw_box is None:
            continue

        if coord == "norm1000":
            x1, y1, x2, y2 = box_to_pixel(raw_box, width=w, height=h)
        else:
            x1, y1, x2, y2 = raw_box

        color = colors.get(label, colors.get("other", "#FFFFFF"))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        if show_labels:
            text = f"{label}"
            # Draw small background tag for label text
            text_bbox = draw.textbbox((x1, max(0.0, y1 - 16)), text)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1 + 2, max(0.0, y1 - 16)), text, fill="#FFFFFF")

    return img
