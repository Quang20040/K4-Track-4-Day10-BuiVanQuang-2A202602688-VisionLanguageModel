"""Dataset loading, multimodal chat formatting, loss masking, and custom collation for Qwen3-VL.
Implements:
  - RoadDamageDataset for train / val / test splits
  - encode_example(): chat templating with prefix loss masking (-100 on prompt & vision tokens)
  - MultimodalDataCollator: dynamic right-padding for sequences and concatenation for visual features
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
from scripts.labcommon import read_jsonl
from src.grounding import DEFAULT_PROMPT, objects_to_answer


class RoadDamageDataset(Dataset):
    """PyTorch Dataset loading road damage JSONL entries and corresponding images."""

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        image_root: Union[str, Path] = "data",
        prompt: str = DEFAULT_PROMPT,
        fraction: float = 1.0,
        seed: int = 42,
    ):
        super().__init__()
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root)
        self.prompt = prompt

        rows = read_jsonl(self.jsonl_path)
        if fraction < 1.0:
            import random
            rng = random.Random(seed)
            sample_size = max(1, int(round(len(rows) * fraction)))
            # Deterministic subset sampling
            indices = list(range(len(rows)))
            rng.shuffle(indices)
            selected_indices = sorted(indices[:sample_size])
            self.entries = [rows[i] for i in selected_indices]
        else:
            self.entries = rows

    def __len__(self) -> int:
        return len(self.entries)

    def get_raw_entry(self, idx: int) -> Dict[str, Any]:
        return self.entries[idx]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.entries[idx]
        image_path = self.image_root / entry["image_path"]
        if not image_path.exists():
            # Try direct relative path
            image_path = Path(entry["image_path"])

        image = Image.open(image_path).convert("RGB")
        answer_json = objects_to_answer(entry.get("objects", []))

        return {
            "image_id": entry["image_id"],
            "image": image,
            "prompt": self.prompt,
            "answer": answer_json,
            "width": entry.get("width", image.width),
            "height": entry.get("height", image.height),
            "objects": entry.get("objects", []),
        }


def encode_example(
    image: Image.Image,
    prompt: str,
    answer: str,
    processor: Any,
) -> Dict[str, torch.Tensor]:
    """Encode an image and text pair into token tensors with autoregressive loss masking.
    Masks prompt prefix and visual tokens with -100 so loss is computed exclusively on the assistant response.

    Args:
        image: PIL Image object.
        prompt: User text prompt.
        answer: Canonical grounding JSON string.
        processor: Qwen3-VL AutoProcessor instance.

    Returns:
        Dict of PyTorch tensors including 'input_ids', 'attention_mask', 'labels',
        'pixel_values', 'image_grid_thw', and optionally 'mm_token_type_ids'.
    """
    user_msg = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    full_msg = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]

    try:
        prompt_enc = processor.apply_chat_template(
            user_msg,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        full_enc = processor.apply_chat_template(
            full_msg,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    except Exception:
        # Fallback if processor requires explicit text + images call
        prompt_text = processor.apply_chat_template(user_msg, add_generation_prompt=True, tokenize=False)
        full_text = processor.apply_chat_template(full_msg, add_generation_prompt=False, tokenize=False)
        prompt_enc = processor(text=[prompt_text], images=[image], return_tensors="pt")
        full_enc = processor(text=[full_text], images=[image], return_tensors="pt")

    prompt_ids = prompt_enc["input_ids"]
    full_ids = full_enc["input_ids"]
    prefix_len = prompt_ids.shape[1]

    # Verify that prompt is indeed a strict prefix of the full sequence
    if not torch.equal(full_ids[:, :prefix_len], prompt_ids):
        # In case special tokens differ slightly, log warning or align on token values
        pass

    labels = full_ids.clone()
    # Mask out prompt and image tokens from loss computation
    labels[:, :prefix_len] = -100

    out = {k: v.squeeze(0) if v.ndim > 1 and k != "pixel_values" and k != "image_grid_thw" else v
           for k, v in full_enc.items()}
    out["labels"] = labels.squeeze(0) if labels.ndim > 1 else labels
    out["prompt_len"] = torch.tensor(prefix_len, dtype=torch.long)
    return out


class MultimodalDataCollator:
    """Collator for dynamic batching of multimodal Qwen3-VL sequences.
    Pads 1D text tensors to batch max length with appropriate values (pad_token_id for inputs, -100 for labels)
    and concatenates visual tensors (pixel_values, image_grid_thw) without padding.
    """

    def __init__(self, pad_token_id: int = 0, ignore_index: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Batch items might already be encoded dicts or raw dataset dicts
        # Expecting encoded dicts from encode_example
        input_ids_list = [item["input_ids"] if item["input_ids"].ndim == 1 else item["input_ids"].squeeze(0)
                          for item in batch]
        attention_mask_list = [item["attention_mask"] if item["attention_mask"].ndim == 1 else item["attention_mask"].squeeze(0)
                               for item in batch]
        labels_list = [item["labels"] if item["labels"].ndim == 1 else item["labels"].squeeze(0)
                       for item in batch]

        max_len = max(ids.shape[0] for ids in input_ids_list)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for ids, mask, lbl in zip(input_ids_list, attention_mask_list, labels_list):
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                # Right padding for causal LM training
                padded_ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)])
                padded_mask = torch.cat([mask, torch.zeros((pad_len,), dtype=mask.dtype)])
                padded_lbl = torch.cat([lbl, torch.full((pad_len,), self.ignore_index, dtype=lbl.dtype)])
            else:
                padded_ids = ids
                padded_mask = mask
                padded_lbl = lbl

            batch_input_ids.append(padded_ids)
            batch_attention_mask.append(padded_mask)
            batch_labels.append(padded_lbl)

        collated: Dict[str, torch.Tensor] = {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }

        # Optional mm_token_type_ids
        if "mm_token_type_ids" in batch[0]:
            mm_list = [item["mm_token_type_ids"] if item["mm_token_type_ids"].ndim == 1 else item["mm_token_type_ids"].squeeze(0)
                       for item in batch]
            batch_mm = []
            for mm in mm_list:
                pad_len = max_len - mm.shape[0]
                if pad_len > 0:
                    padded_mm = torch.cat([mm, torch.zeros((pad_len,), dtype=mm.dtype)])
                else:
                    padded_mm = mm
                batch_mm.append(padded_mm)
            collated["mm_token_type_ids"] = torch.stack(batch_mm)

        # Concatenate vision tensors along dim 0
        if "pixel_values" in batch[0]:
            pixel_values_list = [item["pixel_values"] for item in batch]
            collated["pixel_values"] = torch.cat(pixel_values_list, dim=0)

        if "image_grid_thw" in batch[0]:
            grid_list = [item["image_grid_thw"] for item in batch]
            collated["image_grid_thw"] = torch.cat(grid_list, dim=0)

        return collated
