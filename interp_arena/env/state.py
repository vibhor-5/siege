"""State dataclasses — mechanistic-interpretability-native."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import torch


@dataclass
class ArenaState:
    """Full observable state after each step.

    The game is defined by:
      - Red's goal  : make model_output match  target_output
      - Blue's goal : ensure model_output never contains prohibited_patterns

    Both agents observe the model's *internal* states (residual stream,
    attention maps, logits) and use them to plan mechanistic interventions.
    """

    # ── Episode spec ──────────────────────────────────────────────────────────
    prompt: str                      # Original user prompt (read-only during ep.)
    target_output: str               # What Red is trying to elicit
    prohibited_patterns: list[str]   # What Blue is trying to prevent

    # ── Current prompt (after interventions) ──────────────────────────────────
    modified_prompt: str

    # ── Model output for this step ────────────────────────────────────────────
    model_output: str

    # ── Scalar scores ─────────────────────────────────────────────────────────
    target_similarity: float   # [0,1]  how close output is to Red's target
    safety_score: float        # [0,1]  1=safe, 0=prohibited content present

    step: int

    # ── Mechanistic internals (always populated after load) ───────────────────
    # Shape: list[Tensor(batch, seq, hidden)]  — one entry per layer
    residual_stream: Optional[list[torch.Tensor]] = None

    # Shape: list[Tensor(batch, heads, seq, seq)]  — one entry per layer
    attention_maps: Optional[list[torch.Tensor]] = None

    # Shape: Tensor(batch, vocab)  — final logits before sampling
    logits: Optional[torch.Tensor] = None

    # ── Audit trail ───────────────────────────────────────────────────────────
    applied_red_actions: list[dict] = field(default_factory=list)
    applied_blue_actions: list[dict] = field(default_factory=list)

    # Blue detection info (did Blue detect Red's hook at each layer?)
    blue_detections: list[int] = field(default_factory=list)   # layer ids detected

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (safe for JSON / wandb logging)."""
        return {
            "prompt": self.prompt,
            "target_output": self.target_output,
            "prohibited_patterns": self.prohibited_patterns,
            "modified_prompt": self.modified_prompt,
            "model_output": self.model_output,
            "target_similarity": self.target_similarity,
            "safety_score": self.safety_score,
            "step": self.step,
            "applied_red_actions": self.applied_red_actions,
            "applied_blue_actions": self.applied_blue_actions,
            "blue_detections": self.blue_detections,
        }
