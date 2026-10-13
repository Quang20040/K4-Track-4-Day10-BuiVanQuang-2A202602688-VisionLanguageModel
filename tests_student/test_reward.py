"""Unit tests for src/reward.py (GRPO IoU Verifiable Reward).
Tests all critical reward scenarios:
1. Perfect prediction (Reward == 1.0)
2. Invalid JSON (Reward == 0.0)
3. Both empty (Reward == 1.0)
4. Wrong class (Reward == 0.2)
5. Extra box hallucination (Penalized)
6. Giant box with IoU < 0.5 (Reward == 0.2)
7. All-empty exploitation attempt (Reward == 0.2)
"""

import json
import pytest
from src.reward import compute_grounding_iou_reward


def test_reward_perfect_single_box():
    gt = [{"label": "pothole", "bbox": (100, 200, 300, 400)}]
    pred_str = json.dumps([{"label": "pothole", "bbox_2d": [100, 200, 300, 400]}])
    reward = compute_grounding_iou_reward(pred_str, gt)
    assert reward == pytest.approx(1.0)


def test_reward_both_empty():
    gt = []
    pred_str = "[]"
    reward = compute_grounding_iou_reward(pred_str, gt)
    assert reward == pytest.approx(1.0)


def test_reward_invalid_json():
    gt = [{"label": "pothole", "bbox": (100, 200, 300, 400)}]
    for malformed in ["not json", "{'label': 'pothole'}", "", None, "[1, 2]"]:
        reward = compute_grounding_iou_reward(malformed, gt)
        assert reward == 0.0, f"Expected 0.0 for {malformed}"


def test_reward_wrong_class():
    gt = [{"label": "pothole", "bbox": (100, 200, 300, 400)}]
    # Same coords but wrong label
    pred_str = json.dumps([{"label": "crack", "bbox_2d": [100, 200, 300, 400]}])
    reward = compute_grounding_iou_reward(pred_str, gt)
    # Valid JSON gives 0.2, matched IoU is 0 -> 0.2
    assert reward == pytest.approx(0.2)


def test_reward_extra_box_penalized():
    gt = [{"label": "pothole", "bbox": (100, 200, 300, 400)}]
    # One perfect box + one hallucinated extra box
    pred_str = json.dumps([
        {"label": "pothole", "bbox_2d": [100, 200, 300, 400]},
        {"label": "crack", "bbox_2d": [500, 500, 600, 600]},
    ])
    reward = compute_grounding_iou_reward(pred_str, gt)
    # 0.2 + 0.8 * (1.0 / max(1, 2, 1)) = 0.2 + 0.8 * 0.5 = 0.6
    assert reward == pytest.approx(0.6)
    assert reward < 1.0


def test_reward_giant_box_iou_below_threshold():
    # Box covering whole screen
    gt = [{"label": "pothole", "bbox": (100, 100, 200, 200)}]  # Area 10000
    pred_str = json.dumps([{"label": "pothole", "bbox_2d": [0, 0, 1000, 1000]}])  # Area 1000000 -> IoU = 0.01 < 0.5
    reward = compute_grounding_iou_reward(pred_str, gt, iou_thr=0.5)
    assert reward == pytest.approx(0.2)


def test_reward_all_empty_exploitation():
    gt = [{"label": "pothole", "bbox": (100, 200, 300, 400)}]
    pred_str = "[]"
    reward = compute_grounding_iou_reward(pred_str, gt)
    assert reward == pytest.approx(0.2)
