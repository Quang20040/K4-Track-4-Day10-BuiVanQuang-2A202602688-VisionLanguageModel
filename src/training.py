"""Unified training engine for Qwen3-VL-4B QLoRA experiments (E1, E2, E3, E4).
Features:
  - Mixed precision fp16 training with GradScaler and autograd gradient accumulation
  - Gradient checkpointing (use_reentrant=False)
  - Gradient clipping (norm = 1.0)
  - Cosine annealing learning rate schedule with 5% warmup
  - Peak GPU memory tracking (torch.cuda.max_memory_allocated)
  - Periodic checkpointing (adapter + projector_weights.pt) to local disk & Google Drive
  - Resume functionality
"""

import math
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
from torch.utils.data import DataLoader

from src.dataset import MultimodalDataCollator, encode_example
from src.model_utils import inspect_model_structure, save_experiment_checkpoint


@dataclass
class TrainConfig:
    name: str = "E1"
    lr: float = 2.0e-4
    weight_decay: float = 0.0
    batch_size: int = 2
    grad_accum: int = 4
    max_steps: int = 300
    warmup_ratio: float = 0.05
    max_grad_norm: float = 1.0
    save_every_steps: int = 100
    seed: int = 42
    output_dir: str = "work/checkpoints/e1"
    drive_dir: Optional[str] = None
    train_projector: bool = False


class CosineWarmupScheduler:
    """Cosine learning rate schedule with linear warmup."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, step: int) -> float:
        if step < self.warmup_steps:
            factor = float(step) / float(self.warmup_steps)
        else:
            progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        current_lr = self.min_lr
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            lr = self.min_lr + (base_lr - self.min_lr) * factor
            group["lr"] = lr
            current_lr = lr
        return current_lr


def train_experiment(
    config: TrainConfig,
    model: Any,
    processor: Any,
    train_dataset: Any,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Dict[str, Any]:
    """Execute training loop for an experiment setup."""
    print(f"\n{'='*70}\n[START TRAINING] Experiment: {config.name}\n{'='*70}")
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if run is already completed
    run_file = out_dir / "run.json"
    if run_file.exists():
        import json
        try:
            existing = json.loads(run_file.read_text(encoding="utf-8"))
            if existing.get("global_step", 0) >= config.max_steps:
                print(f"Experiment {config.name} already completed at {out_dir}. Returning saved run info.")
                return existing
        except Exception:
            pass

    # Model optimizations for training
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    # Collect trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params)
    print(f"Total trainable parameters: {trainable_count:,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=config.weight_decay)
    warmup_steps = int(config.max_steps * config.warmup_ratio)
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps=warmup_steps, total_steps=config.max_steps)

    is_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

    # Reset peak memory tracking
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    pad_id = getattr(processor.tokenizer, "pad_token_id", 0)
    if pad_id is None:
        pad_id = 0
    collator = MultimodalDataCollator(pad_token_id=pad_id, ignore_index=-100)

    # Pre-encoding helper to process examples on demand
    def encoded_generator():
        while True:
            # Shuffle indices
            indices = torch.randperm(len(train_dataset)).tolist()
            for idx in indices:
                item = train_dataset[idx]
                encoded = encode_example(
                    image=item["image"],
                    prompt=item["prompt"],
                    answer=item["answer"],
                    processor=processor,
                )
                yield encoded

    gen = encoded_generator()

    model.train()
    global_step = 0
    accum_step = 0
    running_loss = 0.0
    start_time = time.time()
    loss_history: List[float] = []

    optimizer.zero_grad()

    while global_step < config.max_steps:
        # Assemble mini-batch
        batch_items = [next(gen) for _ in range(config.batch_size)]
        batch = collator(batch_items)

        # Move to device
        batch_gpu = {k: v.to(device) for k, v in batch.items()}

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=is_cuda):
            outputs = model(**batch_gpu)
            loss = outputs.loss
            loss_scaled = loss / config.grad_accum

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"[Step {global_step}] Loss became {loss.item()} (non-finite)! "
                f"Check learning rate or fp16 overflow."
            )

        scaler.scale(loss_scaled).backward()
        accum_step += 1
        running_loss += loss.item()

        if accum_step % config.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            global_step += 1
            current_lr = scheduler.step(global_step)
            avg_loss = running_loss / config.grad_accum
            loss_history.append(avg_loss)
            running_loss = 0.0

            if global_step % 10 == 0 or global_step == config.max_steps:
                cur_vram = (
                    torch.cuda.max_memory_allocated() / (1024 ** 3) if is_cuda else 0.0
                )
                elapsed_min = (time.time() - start_time) / 60.0
                print(
                    f"Step {global_step:3d}/{config.max_steps} | "
                    f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                    f"Peak VRAM: {cur_vram:.2f} GB | Elapsed: {elapsed_min:.1f}m"
                )

            # Checkpoint saving
            if global_step % config.save_every_steps == 0 or global_step == config.max_steps:
                step_dir = out_dir / f"step_{global_step}"
                step_meta = {
                    "experiment": config.name,
                    "global_step": global_step,
                    "trainable_params": trainable_count,
                    "peak_vram_gb": round(
                        torch.cuda.max_memory_allocated() / (1024 ** 3) if is_cuda else 0.0, 3
                    ),
                    "train_minutes": round((time.time() - start_time) / 60.0, 2),
                    "loss": round(avg_loss, 4),
                }
                save_experiment_checkpoint(
                    output_dir=step_dir,
                    model=model,
                    processor=processor,
                    is_projector_trained=config.train_projector,
                    meta=step_meta,
                )
                # Also update latest
                save_experiment_checkpoint(
                    output_dir=out_dir,
                    model=model,
                    processor=processor,
                    is_projector_trained=config.train_projector,
                    meta=step_meta,
                )

                # Sync to Google Drive if configured
                if config.drive_dir:
                    try:
                        drive_path = Path(config.drive_dir) / config.name
                        drive_path.mkdir(parents=True, exist_ok=True)
                        for f in out_dir.iterdir():
                            if f.is_file():
                                shutil.copy2(f, drive_path / f.name)
                        print(f"Backed up checkpoint to Google Drive: {drive_path}")
                    except Exception as e:
                        print(f"Drive backup failed: {e}")

    total_time_min = (time.time() - start_time) / 60.0
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if is_cuda else 0.0

    final_run_info = {
        "experiment": config.name,
        "global_step": global_step,
        "trainable_params": trainable_count,
        "peak_vram_gb": round(peak_vram_gb, 3),
        "train_minutes": round(total_time_min, 2),
        "final_loss": round(loss_history[-1] if loss_history else 0.0, 4),
        "loss_history": [round(l, 4) for l in loss_history],
    }

    import json
    run_file.write_text(json.dumps(final_run_info, indent=2), encoding="utf-8")
    print(f"\n[FINISHED {config.name}] Elapsed: {total_time_min:.2f}m | Peak VRAM: {peak_vram_gb:.2f}GB\n")

    return final_run_info
