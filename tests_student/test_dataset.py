"""Unit tests for src/dataset.py.
Tests Dataset subset sampling and MultimodalDataCollator tensor alignment.
"""

import json
import pytest
import torch
from src.dataset import MultimodalDataCollator, RoadDamageDataset


def test_collator_padding_and_concatenation():
    collator = MultimodalDataCollator(pad_token_id=0, ignore_index=-100)

    # Item 1: sequence length 5
    item1 = {
        "input_ids": torch.tensor([10, 20, 30, 40, 50]),
        "attention_mask": torch.tensor([1, 1, 1, 1, 1]),
        "labels": torch.tensor([-100, -100, 30, 40, 50]),
        "pixel_values": torch.randn(2, 3, 14, 14),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }

    # Item 2: sequence length 3
    item2 = {
        "input_ids": torch.tensor([10, 25, 35]),
        "attention_mask": torch.tensor([1, 1, 1]),
        "labels": torch.tensor([-100, 25, 35]),
        "pixel_values": torch.randn(3, 3, 14, 14),
        "image_grid_thw": torch.tensor([[1, 3, 3]]),
    }

    batch = collator([item1, item2])

    assert batch["input_ids"].shape == (2, 5)
    assert batch["attention_mask"].shape == (2, 5)
    assert batch["labels"].shape == (2, 5)

    # Item 2 was padded by 2 tokens
    assert torch.equal(batch["input_ids"][1, 3:], torch.tensor([0, 0]))
    assert torch.equal(batch["attention_mask"][1, 3:], torch.tensor([0, 0]))
    assert torch.equal(batch["labels"][1, 3:], torch.tensor([-100, -100]))

    # Vision tensors concatenated along dim 0
    assert batch["pixel_values"].shape == (5, 3, 14, 14)
    assert batch["image_grid_thw"].shape == (2, 3)


def test_dataset_fraction_sampling(tmp_path):
    # Create fake JSONL with 20 rows
    rows = [{"image_id": f"img_{i}", "image_path": f"images/{i}.jpg", "objects": []}
            for i in range(20)]
    jsonl_file = tmp_path / "test.jsonl"
    jsonl_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    ds_full = RoadDamageDataset(jsonl_file, fraction=1.0)
    assert len(ds_full) == 20

    ds_10pct = RoadDamageDataset(jsonl_file, fraction=0.1, seed=42)
    assert len(ds_10pct) == 2
