"""Hook builders for TransformerLens — Red attack hooks and Blue defense hooks.

TransformerLens hook naming convention:
  blocks.{L}.hook_resid_post          → residual stream after layer L
  blocks.{L}.hook_resid_pre           → residual stream before layer L
  blocks.{L}.attn.hook_attn_scores    → attention logits (pre-softmax)
  blocks.{L}.hook_mlp_out             → MLP output
  hook_logits                         → final logit tensor

All hook functions follow the TransformerLens signature:
    hook_fn(value: Tensor, hook: HookPoint) -> Tensor
"""

from __future__ import annotations

import re
from typing import Callable, Optional

import torch


# ── Type alias ────────────────────────────────────────────────────────────────
HookFn = Callable[[torch.Tensor, object], torch.Tensor]
HookPair = tuple[str, HookFn]   # (hook_name, hook_fn)


# ── Hook name helpers ─────────────────────────────────────────────────────────

def resid_post(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_post"

def resid_pre(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_pre"

def attn_scores(layer: int) -> str:
    return f"blocks.{layer}.attn.hook_attn_scores"

def mlp_out(layer: int) -> str:
    return f"blocks.{layer}.hook_mlp_out"

LOGITS_HOOK = "hook_logits"


# ── Red: attack hooks ─────────────────────────────────────────────────────────

def make_steer_hook(direction: torch.Tensor, strength: float) -> HookFn:
    """Add `strength * direction` to every token's residual stream at this layer."""
    direction = _unit(direction)

    def hook(value: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        # value: (batch, seq, d_model)
        d = direction.to(value.device, value.dtype)
        return value + strength * d

    return hook


def make_amplify_attn_hook(head: int, scale: float) -> HookFn:
    """Scale the attention scores for a specific head (pre-softmax).

    scale > 1 → head attends more sharply
    scale < 1 → head attends more uniformly
    0         → head fully suppressed
    """
    def hook(value: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        # value: (batch, heads, seq_q, seq_k)
        value = value.clone()
        value[:, head, :, :] *= scale
        return value

    return hook


def make_patch_hook(position: int, patch_value: torch.Tensor) -> HookFn:
    """Replace residual stream at *position* with *patch_value*."""
    def hook(value: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        # value: (batch, seq, d_model)
        p = patch_value.to(value.device, value.dtype)
        value = value.clone()
        value[:, position, :] = p
        return value

    return hook


def make_logit_bias_hook(token_ids: list[int], bias: float) -> HookFn:
    """Add *bias* to the final logits for *token_ids* (steers generation)."""
    def hook(value: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        # value: (batch, seq, vocab) or (batch, vocab)
        value = value.clone()
        value[..., token_ids] += bias
        return value

    return hook


# ── Blue: defense hooks ───────────────────────────────────────────────────────

def make_ablate_hook(direction: torch.Tensor) -> HookFn:
    """Project *direction* out of the residual stream (orthogonal ablation)."""
    direction = _unit(direction)

    def hook(value: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        d = direction.to(value.device, value.dtype)
        # v - (v·d) * d   for each token
        # value: (batch, seq, d_model)
        proj = (value @ d).unsqueeze(-1) * d  # (batch, seq, d_model)
        return value - proj

    return hook


def make_suppress_head_hook(head: int) -> HookFn:
    """Zero out a specific attention head's contribution."""
    def hook(value: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        value = value.clone()
        value[:, head, :, :] = 0.0
        return value

    return hook


def make_clamp_hook(clamp_min: float, clamp_max: float) -> HookFn:
    """Clamp residual stream values to [clamp_min, clamp_max]."""
    def hook(value: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        return value.clamp(clamp_min, clamp_max)

    return hook


def make_restore_hook(
    position: int,
    baseline_activation: torch.Tensor,
) -> HookFn:
    """Replace activation at *position* with the clean baseline."""
    def hook(value: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        b = baseline_activation.to(value.device, value.dtype)
        value = value.clone()
        value[:, position, :] = b
        return value

    return hook


def make_logit_filter_hook(token_ids: list[int]) -> HookFn:
    """Set logits for prohibited token ids to -inf."""
    def hook(value: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        value = value.clone()
        value[..., token_ids] = float("-inf")
        return value

    return hook


# ── Detection utility ─────────────────────────────────────────────────────────

def make_detection_hook(
    ref_direction: torch.Tensor,
    threshold: float,
    detected_layers: list[int],
    layer_id: int,
) -> HookFn:
    """Blue monitoring hook: records *layer_id* if steering is detected.

    Detects steering by measuring cosine similarity of the per-token residual
    shift relative to `ref_direction`.  If mean cosine sim > threshold, the
    layer is flagged.
    """
    ref = _unit(ref_direction)

    def hook(value: torch.Tensor, hook) -> torch.Tensor:  # noqa: ANN001
        r = ref.to(value.device, value.dtype)
        # value: (batch, seq, d_model)
        norms = value.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cos_sim = (value / norms) @ r          # (batch, seq)
        mean_cos = cos_sim.abs().mean().item()
        if mean_cos > threshold:
            detected_layers.append(layer_id)
        return value  # no modification — pure observation

    return hook


# ── Internals ─────────────────────────────────────────────────────────────────

def _unit(v: torch.Tensor) -> torch.Tensor:
    norm = v.norm()
    return v / norm if norm > 1e-8 else v
