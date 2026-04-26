"""GPU training config — Unsloth 4-bit LoRA + TRL GRPOTrainer.

Mirrors the seige pattern: load model via Unsloth, apply LoRA,
return (model, tokenizer) for use in GRPOTrainer.

All hyperparameters are overridable via environment variables so
the same script runs on RunPod, Colab, and local GPU without changes.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Optional

import torch


# ── Environment variable helpers ─────────────────────────────────────────────

def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)

def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))

def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))

def _env_bool(key: str, default: bool) -> bool:
    return os.getenv(key, str(int(default))) in ("1", "true", "True", "yes")


# ── Global constants (overridable) ────────────────────────────────────────────

# Agent models (small LLMs that learn to output JSON actions)
AGENT_MODEL_ID   = _env_str("SIEGE_AGENT_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
# Target model (the model agents manipulate — stays frozen throughout training)
TARGET_MODEL_ID  = _env_str("SIEGE_TARGET_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

ENV_URL          = _env_str("SIEGE_ENV_URL", "http://localhost:8000")
WANDB_PROJECT    = _env_str("WANDB_PROJECT", "interp-arena")

MAX_SEQ_LENGTH   = _env_int("SIEGE_AGENT_MAX_SEQ_LENGTH", 4096)
LOAD_IN_4BIT     = _env_bool("SIEGE_LOAD_IN_4BIT", True)
LORA_R           = _env_int("SIEGE_LORA_R", 16)
LORA_ALPHA       = _env_int("SIEGE_LORA_ALPHA", 32)


def lora_target_modules() -> list[str]:
    raw = _env_str("SIEGE_LORA_TARGET_MODULES", "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    # Qwen2.5 and Llama-family modules
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclasses.dataclass
class UnslothConfig:
    """Centralized config for Unsloth + GRPO training."""
    agent_model_id: str       = AGENT_MODEL_ID
    target_model_id: str      = TARGET_MODEL_ID
    env_url: str              = ENV_URL
    wandb_project: str        = WANDB_PROJECT
    max_seq_length: int       = MAX_SEQ_LENGTH
    load_in_4bit: bool        = LOAD_IN_4BIT
    lora_r: int               = LORA_R
    lora_alpha: int           = LORA_ALPHA

    # GRPO
    num_train_epochs: int     = _env_int("SIEGE_GRPO_EPOCHS", 1)
    per_device_batch: int     = _env_int("SIEGE_GRPO_PER_DEVICE_BATCH", 1)
    grad_accum_steps: int     = _env_int("SIEGE_GRPO_GRAD_ACCUM", 8)
    learning_rate: float      = _env_float("SIEGE_GRPO_LR", 5e-6)
    logging_steps: int        = _env_int("SIEGE_GRPO_LOGGING_STEPS", 5)
    num_generations: int      = _env_int("SIEGE_GRPO_NUM_GENERATIONS", 4)
    max_prompt_length: int    = _env_int("SIEGE_GRPO_MAX_PROMPT_LENGTH", 512)
    max_completion_length: int = _env_int("SIEGE_GRPO_MAX_COMPLETION_LENGTH", 128)
    temperature: float        = _env_float("SIEGE_GRPO_TEMPERATURE", 0.8)
    save_steps: int           = _env_int("SIEGE_GRPO_SAVE_STEPS", 50)
    beta: float               = 0.04

    # Generational training
    num_generations_training: int = _env_int("SIEGE_NUM_GENERATIONS", 3)
    steps_per_agent: int          = _env_int("SIEGE_STEPS_PER_AGENT", 300)


def grpo_config(output_dir: str, run_name: str, cfg: Optional[UnslothConfig] = None):
    """Build a GRPOConfig from env/config."""
    from trl import GRPOConfig

    c = cfg or UnslothConfig()
    return GRPOConfig(
        num_train_epochs=c.num_train_epochs,
        per_device_train_batch_size=c.per_device_batch,
        gradient_accumulation_steps=c.grad_accum_steps,
        learning_rate=c.learning_rate,
        logging_steps=c.logging_steps,
        output_dir=output_dir,
        report_to=_env_str("SIEGE_REPORT_TO", "wandb"),
        run_name=run_name,
        num_generations=c.num_generations,
        max_prompt_length=c.max_prompt_length,
        max_completion_length=c.max_completion_length,
        temperature=c.temperature,
        beta=c.beta,
        use_vllm=False,
        reward_weights=None,
        save_steps=c.save_steps,
    )


def _patch_torch() -> None:
    """Compatibility patches for Unsloth on various RunPod/Colab torch builds."""
    if hasattr(torch, "_inductor") and not hasattr(torch._inductor, "config"):
        class _CompatConfig:  # pragma: no cover
            pass
        torch._inductor.config = _CompatConfig()

    # TorchAO experimental dtypes that may be absent
    _fallback = {
        "int1": torch.int8, "int2": torch.int8, "int3": torch.int8,
        "int4": torch.int8, "int5": torch.int8, "int6": torch.int8, "int7": torch.int8,
        "uint1": torch.uint8, "uint2": torch.uint8, "uint3": torch.uint8,
        "uint4": torch.uint8, "uint5": torch.uint8, "uint6": torch.uint8, "uint7": torch.uint8,
    }
    for name, fb in _fallback.items():
        if not hasattr(torch, name):
            setattr(torch, name, fb)


def _unsloth_compile_cache_paths() -> list[Path]:
    """Locations Unsloth uses for TRL/GRPO trainer rewrites (see unsloth_zoo.compiler)."""
    return [
        Path.cwd() / "unsloth_compiled_cache",
        Path("/tmp/unsloth_compiled_cache"),
    ]


def _clear_unsloth_compile_caches() -> list[str]:
    """Remove on-disk Unsloth compile outputs. Returns list of removed paths (as strings)."""
    removed: list[str] = []
    for p in _unsloth_compile_cache_paths():
        if not p.is_dir():
            continue
        try:
            shutil.rmtree(p)
            removed.append(str(p))
        except OSError:
            shutil.rmtree(p, ignore_errors=True)
            if not p.exists():
                removed.append(str(p))
    return removed


def _evict_unsloth_from_sys_modules() -> None:
    """Drop half-imported unsloth* modules so a retry can re-run patches."""
    for k in [m for m in list(sys.modules) if m == "unsloth" or m.startswith("unsloth_") or m.startswith("unsloth.")]:
        del sys.modules[k]
    importlib.invalidate_caches()


def _is_unsloth_grpo_compile_failure(exc: BaseException) -> bool:
    """True when Unsloth’s TRL/GRPO trainer rewrite left bad generated code (often after Py 3.14)."""
    msg = f"{type(exc).__name__}: {exc!s}"
    if isinstance(exc, (SyntaxError, RuntimeError, ImportError)):
        return (
            "UnslothGRPO" in msg
            or "Direct module loading failed" in msg
            or "invalid syntax" in msg.lower()
        )
    return False


def load_agent_model(cfg: Optional[UnslothConfig] = None):
    """Load small LLM agent model with Unsloth 4-bit LoRA.

    Returns (model, tokenizer) ready for GRPOTrainer.
    """
    c = cfg or UnslothConfig()
    os.environ.setdefault("TRANSFORMERS_NO_TORCHAO", "1")
    _patch_torch()

    if sys.version_info >= (3, 14):
        raise RuntimeError(
            "Unsloth+TRL GRPO is not supported on Python 3.14+ yet (Unsloth’s compiled "
            "UnslothGRPOTrainer can raise SyntaxError). Use Python 3.10–3.13, e.g.:\n"
            "  uv python install 3.12\n"
            "  rm -rf .venv && uv venv --python 3.12 && uv sync --extra gpu\n"
            "Add a .python-version file with 3.12 or use: UV_PYTHON=3.12 uv sync --extra gpu\n"
            "If you only changed Python after a failed run, also remove: "
            "unsloth_compiled_cache/ and /tmp/unsloth_compiled_cache"
        ) from None

    _flm: type | None = None
    for attempt in (1, 2):
        try:
            from unsloth import FastLanguageModel as _Flm  # noqa: PLC0415

            _flm = _Flm
            break
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "unsloth is required for GRPO training. Install with:\n"
                "  uv sync --extra gpu\n"
                "or: pip install unsloth"
            ) from e
        except Exception as e:
            if attempt == 1 and _is_unsloth_grpo_compile_failure(e):
                removed = _clear_unsloth_compile_caches()
                _evict_unsloth_from_sys_modules()
                warnings.warn(
                    "Unsloth GRPO trainer compile failed; cleared compile caches and retrying import once. "
                    f"python={sys.version.split()[0]} {sys.executable!r} removed={removed!r}",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            raise
    if _flm is None:  # pragma: no cover
        raise RuntimeError("unsloth import: internal state (all retries failed)")
    FastLanguageModel = _flm

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=c.agent_model_id,
        max_seq_length=c.max_seq_length,
        load_in_4bit=c.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=c.lora_r,
        lora_alpha=c.lora_alpha,
        target_modules=lora_target_modules(),
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer


def load_frozen_opponent(adapter_path: str, cfg: Optional[UnslothConfig] = None):
    """Load a frozen opponent from a saved LoRA adapter (eval mode, no grad)."""
    from peft import PeftModel  # noqa: PLC0415
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

    c = cfg or UnslothConfig()
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    base = AutoModelForCausalLM.from_pretrained(
        c.agent_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=c.load_in_4bit,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer
