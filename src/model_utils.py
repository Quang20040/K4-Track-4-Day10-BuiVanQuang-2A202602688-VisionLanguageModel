"""Model loading, QLoRA setup, parameter inspection, and projector weight management for Qwen3-VL-4B.
Implements:
  - 4-bit NF4 double quantization with ViT and lm_head skipped
  - PEFT LoRA target module configuration for E1, E2, E3
  - Float32 unfreezing for visual.merger and visual.deepstack_merger_list in E2/E3
  - Atomic saving and loading of LoRA adapter + separate projector_weights.pt
"""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

try:
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
    )
except ImportError:
    from transformers import AutoProcessor, BitsAndBytesConfig  # type: ignore
    AutoModelForImageTextToText = None  # type: ignore

try:
    from transformers import Qwen3VLForConditionalGeneration  # type: ignore
except ImportError:
    Qwen3VLForConditionalGeneration = None  # type: ignore

from peft import LoraConfig, PeftModel, get_peft_model


def get_model_class():
    """Return the appropriate Model class for Qwen3-VL."""
    if Qwen3VLForConditionalGeneration is not None:
        return Qwen3VLForConditionalGeneration
    if AutoModelForImageTextToText is not None:
        return AutoModelForImageTextToText
    from transformers import AutoModelForVision2Seq
    return AutoModelForVision2Seq


def create_bnb_config(quant_type: str = "nf4") -> BitsAndBytesConfig:
    """Create 4-bit BitsAndBytesConfig keeping ViT and lm_head in fp16.

    As documented in GUIDE Step 3:
    ViT and lm_head are skipped from 4-bit quantization (kept in fp16/fp32)
    so that projector weights (merger & deepstack_merger_list) remain full precision
    and can be fine-tuned without gradient quantization failure.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head"],
    )


def load_model_and_processor(
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct",
    quantization: str = "nf4-4bit",
    device_map: Union[str, Dict[str, int]] = "auto",
    padding_side: str = "right",
) -> Tuple[Any, Any]:
    """Load Qwen3-VL-4B with 4-bit NF4 quantization and its AutoProcessor."""
    quant_config = create_bnb_config() if "4bit" in quantization else None
    model_cls = get_model_class()

    print(f"Loading model '{model_id}' with class {model_cls.__name__}...")
    model = model_cls.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    processor.tokenizer.padding_side = padding_side

    return model, processor


def inspect_model_structure(model: Any) -> Dict[str, Any]:
    """Inspect parameter counts, module types, and precision dtypes across model blocks."""
    total_params = 0
    trainable_params = 0
    dtypes = {}
    modules_by_block = {"visual": 0, "language_model": 0, "lm_head": 0, "other": 0}

    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel
        if param.requires_grad:
            trainable_params += numel
        dt = str(param.dtype)
        dtypes[dt] = dtypes.get(dt, 0) + numel

        if "visual" in name:
            modules_by_block["visual"] += numel
        elif "language_model" in name:
            modules_by_block["language_model"] += numel
        elif "lm_head" in name:
            modules_by_block["lm_head"] += numel
        else:
            modules_by_block["other"] += numel

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percent": 100.0 * trainable_params / total_params if total_params else 0.0,
        "dtypes": dtypes,
        "params_by_block": modules_by_block,
    }


def apply_setup(
    model: Any,
    setup_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    train_projector: Optional[bool] = None,
) -> Any:
    """Configure LoRA adapter and unfreeze projector modules according to experiment setup.

    Setups:
      - E1: LoRA on LLM attention & MLP (36 layers * 7 = 252 modules). ViT and Projector frozen.
      - E2: E1 + full projector in float32 (~109M params: visual.merger and visual.deepstack_merger_list).
      - E3: E2 + LoRA on ViT blocks (24 blocks * 4 = 96 modules). Total LoRA modules = 348.
      - E4: Ablation setup matching E1/E2/E3 with modified rank or data size.
    """
    # 1. Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Required for gradient checkpointing with frozen base / quantized model
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    is_e3 = setup_name.upper() == "E3" or (setup_name.upper().startswith("E4") and "e3" in setup_name.lower())
    should_train_projector = (
        train_projector if train_projector is not None
        else setup_name.upper() in ("E2", "E3")
    )

    # 2. Target module regex patterns
    # LLM targets: 36 layers x 7 = 252 modules
    llm_pattern = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
    # ViT targets: 24 blocks x 4 = 96 modules
    vit_pattern = r".*visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.(linear_fc1|linear_fc2))"

    if is_e3:
        target_regex = f"({llm_pattern}|{vit_pattern})"
    else:
        target_regex = llm_pattern

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_regex,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, peft_config)

    # Count matched LoRA modules
    matched_lora_modules = [
        name for name, mod in peft_model.named_modules()
        if hasattr(mod, "lora_A")
    ]
    print(f"[{setup_name}] Matched LoRA modules count: {len(matched_lora_modules)}")

    # 3. Unfreeze projector if requested (E2, E3)
    projector_param_count = 0
    if should_train_projector:
        for name, mod in peft_model.named_modules():
            if any(k in name for k in ["visual.merger", "visual.deepstack_merger_list"]):
                mod.to(torch.float32)
        for name, param in peft_model.named_parameters():
            if "visual.merger" in name or "visual.deepstack_merger_list" in name:
                param.data = param.data.to(torch.float32)
                param.requires_grad = True
                projector_param_count += param.numel()
        for name, param in peft_model.named_parameters():
            if param.requires_grad and param.dtype != torch.float32:
                param.data = param.data.to(torch.float32)
        print(f"[{setup_name}] Unfroze Projector (float32): {projector_param_count:,} params")

    stats = inspect_model_structure(peft_model)
    print(
        f"[{setup_name}] Total params: {stats['total_params']:,} | "
        f"Trainable params: {stats['trainable_params']:,} "
        f"({stats['trainable_percent']:.4f}%)"
    )

    return peft_model


def save_experiment_checkpoint(
    output_dir: Union[str, Path],
    model: Any,
    processor: Any,
    is_projector_trained: bool = False,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Save LoRA adapter, extra projector weights (if trained), processor, and metadata."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Save PEFT LoRA adapter
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(str(out_path))

    # 2. Save separate projector weights for E2/E3
    if is_projector_trained:
        projector_weights = {
            name: param.detach().cpu()
            for name, param in model.named_parameters()
            if ("visual.merger" in name or "visual.deepstack_merger_list" in name) and param.requires_grad
        }
        if projector_weights:
            proj_file = out_path / "projector_weights.pt"
            torch.save(projector_weights, proj_file)
            print(f"Saved {len(projector_weights)} projector tensors to {proj_file}")

    # 3. Save processor
    if processor is not None and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(out_path))

    # 4. Save metadata
    if meta is not None:
        (out_path / "run.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def load_experiment_checkpoint(
    base_model: Any,
    checkpoint_dir: Union[str, Path],
    is_projector_trained: bool = False,
) -> Any:
    """Load LoRA adapter and restore projector weights onto the base model."""
    ckpt_path = Path(checkpoint_dir)
    print(f"Loading LoRA adapter from {ckpt_path}...")
    model = PeftModel.from_pretrained(base_model, str(ckpt_path))

    proj_file = ckpt_path / "projector_weights.pt"
    if is_projector_trained or proj_file.exists():
        if proj_file.exists():
            print(f"Loading projector weights from {proj_file}...")
            proj_weights = torch.load(proj_file, map_location="cpu")
            state_dict = model.state_dict()
            loaded = 0
            for name, tensor in proj_weights.items():
                if name in state_dict:
                    state_dict[name].copy_(
                        tensor.to(state_dict[name].device, dtype=state_dict[name].dtype)
                    )
                    loaded += 1
            print(f"Restored {loaded} projector tensors.")
        else:
            print(f"WARNING: Projector weights requested but {proj_file} does not exist!")

    model.eval()
    return model
