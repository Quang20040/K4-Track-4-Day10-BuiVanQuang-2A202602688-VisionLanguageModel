"""Inference engine for Qwen3-VL-4B on Road Damage Test / Val splits.
Features:
  - Deterministic greedy decoding (do_sample=False, max_new_tokens=256)
  - Left-padding for batched vision-language generation
  - Resume capability: skips already-inferred image_ids to prevent wasted GPU compute
  - Writes standard work/*.jsonl format expected by scripts/score.py
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import torch
from scripts.labcommon import read_jsonl, write_jsonl
from src.grounding import DEFAULT_PROMPT


def run_inference_on_split(
    model: Any,
    processor: Any,
    split_jsonl: Union[str, Path] = "data/splits/test.jsonl",
    image_root: Union[str, Path] = "data",
    prompt: str = DEFAULT_PROMPT,
    output_file: Union[str, Path] = "work/e0.jsonl",
    batch_size: int = 4,
    max_new_tokens: int = 256,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> List[Dict[str, str]]:
    """Execute batch inference on all images in split_jsonl and write output to output_file."""
    split_path = Path(split_jsonl)
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    entries = read_jsonl(split_path)
    total_images = len(entries)
    print(f"\n[INFERENCE] Running inference on {total_images} images from {split_path} -> {out_path}")

    # Set left padding for generation
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"

    # Check already completed predictions for resuming
    completed_preds: Dict[str, str] = {}
    if out_path.exists():
        for r in read_jsonl(out_path):
            if "image_id" in r and "raw" in r:
                completed_preds[r["image_id"]] = r["raw"]
        print(f"Found {len(completed_preds)} existing predictions in {out_path}. Resuming...")

    model.eval()

    # Filter remaining entries
    remaining_entries = [e for e in entries if e["image_id"] not in completed_preds]
    print(f"Remaining images to process: {len(remaining_entries)}")

    # Process in batches
    for i in range(0, len(remaining_entries), batch_size):
        chunk = remaining_entries[i:i + batch_size]
        batch_messages = []
        batch_ids = []

        for e in chunk:
            img_path = Path(image_root) / e["image_path"]
            if not img_path.exists():
                img_path = Path(e["image_path"])

            img = Image.open(img_path).convert("RGB")
            msg = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            batch_messages.append(msg)
            batch_ids.append(e["image_id"])

        try:
            inputs = processor.apply_chat_template(
                batch_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
        except Exception:
            # Fallback per-item batching if processor does not accept list of message lists
            texts = [
                processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                for m in batch_messages
            ]
            images = [m[0]["content"][0]["image"] for m in batch_messages]
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)

        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") else 0,
            )

        # Slice off prompt tokens to keep only generated response
        generated_tokens = outputs[:, prompt_len:]
        decoded_responses = processor.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )

        # Append immediately to out_path
        with open(out_path, "a", encoding="utf-8") as f:
            for img_id, raw_resp in zip(batch_ids, decoded_responses):
                completed_preds[img_id] = raw_resp
                f.write(json.dumps({"image_id": img_id, "raw": raw_resp}, ensure_ascii=False) + "\n")

        done = len(completed_preds)
        if done % 20 == 0 or done == total_images:
            print(f"Progress: {done}/{total_images} ({(done / total_images) * 100:.1f}%)")

    # Final verification: ensure every image_id has exactly one prediction in proper order
    final_rows = []
    missing = 0
    for e in entries:
        img_id = e["image_id"]
        if img_id in completed_preds:
            final_rows.append({"image_id": img_id, "raw": completed_preds[img_id]})
        else:
            missing += 1
            final_rows.append({"image_id": img_id, "raw": "[]"})

    write_jsonl(out_path, final_rows)
    print(f"[INFERENCE COMPLETE] Successfully written {len(final_rows)} predictions to {out_path} (Missing: {missing})\n")
    return final_rows
