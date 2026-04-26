"""Action space — mechanistic interventions on model internals.

Both Red and Blue agents operate primarily on the model's *internal*
computation graph (residual stream, attention heads, logits) via
TransformerLens hook points.

Hook name conventions (TransformerLens):
  blocks.{L}.hook_resid_pre       residual at layer L input
  blocks.{L}.hook_resid_post      residual at layer L output  ← primary target
  blocks.{L}.attn.hook_attn_scores  attention logits (pre-softmax)
  blocks.{L}.hook_mlp_out         MLP output at layer L
  hook_logits                     final logit tensor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ── Enumerations ──────────────────────────────────────────────────────────────

class RedActionType(str, Enum):
    """Red agent interventions — all mechanistic except MODIFY_PROMPT."""

    # Core mechanistic (primary)
    STEER_RESIDUAL   = "steer_residual"    # add direction*strength to resid_post@L
    AMPLIFY_ATTN     = "amplify_attn"      # scale attention scores at (L, head)
    PATCH_ACTIVATION = "patch_activation"  # replace activation at (L, pos) with value
    LOGIT_BIAS       = "logit_bias"        # bias final logits toward target tokens

    # Prompt-level (secondary / baseline)
    MODIFY_PROMPT    = "modify_prompt"
    APPEND_SUFFIX    = "append_suffix"
    # Side-channel: run an extra decode on *text*; surfaced on the observation
    QUERY_MODEL      = "query_model"


class BlueActionType(str, Enum):
    """Blue agent interventions — monitor and correct internal states."""

    # Core mechanistic (primary)
    ABLATE_DIRECTION  = "ablate_direction"   # project out direction from resid@L
    CLAMP_ACTIVATION  = "clamp_activation"   # clamp values in resid@L to safe range
    RESTORE_BASELINE  = "restore_baseline"   # replace resid@L,pos with clean baseline
    SUPPRESS_HEAD     = "suppress_head"      # zero out attn head output at L
    LOGIT_FILTER      = "logit_filter"       # suppress prohibited token ids in logits

    # Prompt-level (secondary)
    SANITIZE_PROMPT   = "sanitize_prompt"
    BLOCK_OUTPUT      = "block_output"       # nuclear: replace output with safe message
    NOOP              = "noop"               # observe only


# ── Action dataclasses ────────────────────────────────────────────────────────

@dataclass
class RedAction:
    """An action issued by the Red (attack) agent.

    Fields used depend on ``type`` — unused fields are None.

    steer_residual
        layer, direction_id (key in DirectionRegistry), strength
    amplify_attn
        layer, head, scale  (scale < 1 suppresses, > 1 amplifies)
    patch_activation
        layer, position, patch_value  (1-D tensor, hidden_size)
    logit_bias
        target_token_ids, bias_strength
    modify_prompt / append_suffix
        text
    """

    type: RedActionType

    # mechanistic fields
    layer: Optional[int] = None
    direction_id: Optional[str] = None   # key in DirectionRegistry
    strength: Optional[float] = None

    head: Optional[int] = None
    scale: Optional[float] = None

    position: Optional[int] = None
    patch_value: Optional[object] = None  # torch.Tensor at runtime

    target_token_ids: Optional[list[int]] = None
    bias_strength: Optional[float] = None

    # prompt-level fields
    text: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()
                if v is not None and k != "patch_value"}


@dataclass
class BlueAction:
    """An action issued by the Blue (defense) agent.

    ablate_direction
        layer, direction_id  — projects out the direction from resid_post
    clamp_activation
        layer, clamp_min, clamp_max
    restore_baseline
        layer, position       — resets that position's activation to the
                                clean (unsteered) baseline captured at reset()
    suppress_head
        layer, head
    logit_filter
        prohibited_token_ids  — sets those logit positions to -inf
    sanitize_prompt / block_output / noop
        no extra fields
    """

    type: BlueActionType

    # mechanistic fields
    layer: Optional[int] = None
    direction_id: Optional[str] = None
    position: Optional[int] = None

    head: Optional[int] = None

    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None

    prohibited_token_ids: Optional[list[int]] = None

    # prompt-level
    text: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ── Convenience constructors ──────────────────────────────────────────────────

def red_steer(layer: int, direction_id: str, strength: float) -> RedAction:
    return RedAction(type=RedActionType.STEER_RESIDUAL,
                     layer=layer, direction_id=direction_id, strength=strength)

def red_amplify(layer: int, head: int, scale: float) -> RedAction:
    return RedAction(type=RedActionType.AMPLIFY_ATTN,
                     layer=layer, head=head, scale=scale)

def red_logit_bias(token_ids: list[int], strength: float) -> RedAction:
    return RedAction(type=RedActionType.LOGIT_BIAS,
                     target_token_ids=token_ids, bias_strength=strength)

def red_modify(text: str) -> RedAction:
    return RedAction(type=RedActionType.MODIFY_PROMPT, text=text)

def red_suffix(text: str) -> RedAction:
    return RedAction(type=RedActionType.APPEND_SUFFIX, text=text)


def red_query(text: str) -> RedAction:
    """Run *text* as a standalone prompt on the target LM (see arena step)."""
    return RedAction(type=RedActionType.QUERY_MODEL, text=text)

def blue_ablate(layer: int, direction_id: str) -> BlueAction:
    return BlueAction(type=BlueActionType.ABLATE_DIRECTION,
                      layer=layer, direction_id=direction_id)

def blue_suppress_head(layer: int, head: int) -> BlueAction:
    return BlueAction(type=BlueActionType.SUPPRESS_HEAD,
                      layer=layer, head=head)

def blue_logit_filter(token_ids: list[int]) -> BlueAction:
    return BlueAction(type=BlueActionType.LOGIT_FILTER,
                      prohibited_token_ids=token_ids)

def blue_restore(layer: int, position: int) -> BlueAction:
    return BlueAction(type=BlueActionType.RESTORE_BASELINE,
                      layer=layer, position=position)

def blue_sanitize() -> BlueAction:
    return BlueAction(type=BlueActionType.SANITIZE_PROMPT)

def blue_block() -> BlueAction:
    return BlueAction(type=BlueActionType.BLOCK_OUTPUT)

def blue_noop() -> BlueAction:
    return BlueAction(type=BlueActionType.NOOP)
