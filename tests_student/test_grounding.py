"""Unit tests for src/grounding.py.
Verifies YOLO to Grounding Box conversion, rounding, coordinate clamping,
sorting stability, pixel conversions, and edge cases.
"""

import json
import pytest
from src.grounding import (
    CLASSES,
    box_to_pixel,
    objects_to_answer,
    pixel_to_box,
    yolo_to_box,
)


def test_yolo_to_box_standard():
    # Center (0.5, 0.5), width 0.2, height 0.4
    # x1 = (0.5 - 0.1) * 1000 = 400
    # y1 = (0.5 - 0.2) * 1000 = 300
    # x2 = (0.5 + 0.1) * 1000 = 600
    # y2 = (0.5 + 0.2) * 1000 = 700
    box = yolo_to_box("pothole", [0.5, 0.5, 0.2, 0.4])
    assert box == {"label": "pothole", "bbox_2d": [400, 300, 600, 700]}


def test_yolo_to_box_boundary_clamping():
    # Box overflowing on borders
    box = yolo_to_box("crack", [0.0, 1.0, 0.2, 0.2])
    # x1 = -100 -> clamped to 0
    # y1 = 900
    # x2 = 100
    # y2 = 1100 -> clamped to 1000
    assert box == {"label": "crack", "bbox_2d": [0, 900, 100, 1000]}


def test_yolo_to_box_degenerate_prevented():
    # Tiny box where x1 == x2 after rounding
    box = yolo_to_box("manhole", [0.5, 0.5, 0.0001, 0.0001])
    b = box["bbox_2d"]
    assert b[0] < b[2], "x1 must be strictly less than x2"
    assert b[1] < b[3], "y1 must be strictly less than y2"


def test_objects_to_answer_empty():
    ans = objects_to_answer([])
    assert ans == "[]"


def test_objects_to_answer_sorting():
    # Two objects: obj1 lower down, obj2 higher up
    objs = [
        {"label": "crack", "yolo": [0.5, 0.8, 0.1, 0.1]},     # y ~ 800
        {"label": "pothole", "yolo": [0.5, 0.2, 0.1, 0.1]},   # y ~ 200
    ]
    ans = objects_to_answer(objs)
    parsed = json.loads(ans)
    assert len(parsed) == 2
    assert parsed[0]["label"] == "pothole"
    assert parsed[1]["label"] == "crack"
    assert parsed[0]["bbox_2d"][1] < parsed[1]["bbox_2d"][1]


def test_pixel_and_box_roundtrip():
    w, h = 640, 360
    orig_box = [150, 200, 450, 600]
    px = box_to_pixel(orig_box, width=w, height=h)
    recovered = pixel_to_box(px, width=w, height=h)
    assert recovered["bbox_2d"] == orig_box
