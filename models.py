"""OpenEnv-compliant models for Interpretability Arena.

These are the typed API contracts between the client and server.
- InterpArenaAction  : combined Red + Blue actions for one step
- InterpArenaObservation : what both agents observe after a step
- InterpArenaState   : episode-level metadata (returned by state())

Mechanistic fields (residual_stream, logits) are omitted from the
wire format for efficiency — they are accessed in-process on the
server side and surfaced as summary statistics in the observation.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field

from openenv.core.env_server import Action, Observation, State

# Wire / schema enums (Gradio dropdown + docs)
RedIntervention = Literal[
    "steer_residual",
    "amplify_attn",
    "patch_activation",
    "logit_bias",
    "modify_prompt",
    "append_suffix",
    "query_model",
]
BlueDefense = Literal[
    "ablate_direction",
    "clamp_activation",
    "restore_baseline",
    "suppress_head",
    "logit_filter",
    "sanitize_prompt",
    "block_output",
    "noop",
]


# ── Action ─────────────────────────────────────────────────────────────────────
# One combined action per step; both agents act simultaneously.

class InterpArenaAction(Action):
    """Combined Red + Blue action for a single arena step.

    **Both agents act in one step:** ``env.step(InterpArenaAction(...))`` always
    sends Red's move and Blue's move together for the same forward pass.

    Red fields
    ----------
    red_type
        Intervention family (see titles below).
    red_layer, red_direction_id, red_strength, …
        Only the fields relevant to ``red_type`` need to be set.

    Blue fields
    -----------
    blue_type
        Defense family.
    blue_layer, blue_direction_id, …
        Same idea — fill what ``blue_type`` requires (``noop`` needs no extras).
    """

    # ── Red (attacker) ──────────────────────────────────────────────────────
    red_type: RedIntervention = Field(
        default="append_suffix",
        title="Red: intervention type",
        description=(
            "What Red does this step: e.g. steer_residual adds a concept vector at a layer; "
            "append_suffix / modify_prompt change text; query_model runs a side prompt on the same LM."
        ),
    )
    red_layer: Optional[int] = Field(
        default=None,
        title="Red: transformer layer index",
        description="Layer 0 = early, higher = later. Used by steer_residual, amplify_attn, patch_activation.",
    )
    red_direction_id: Optional[str] = Field(
        default=None,
        title="Red: steering direction name",
        description="Registry key, e.g. jailbreak, refusal, toxicity (Space must register it).",
    )
    red_strength: Optional[float] = Field(
        default=None,
        title="Red: steering / patch strength",
        description="Scale for residual steering or similar (meaning depends on intervention type).",
    )
    red_head: Optional[int] = Field(
        default=None,
        title="Red: attention head index",
        description="Which head to amplify (for amplify_attn).",
    )
    red_scale: Optional[float] = Field(
        default=None,
        title="Red: attention scale factor",
        description="Multiply attention scores for that head (amplify_attn).",
    )
    red_position: Optional[int] = Field(
        default=None,
        title="Red: token position (sequence index)",
        description="Token index for patch_activation / some hooks.",
    )
    red_target_token_ids: Optional[list[int]] = Field(
        default=None,
        title="Red: vocab token IDs to bias (logit_bias)",
        description="Comma-separated integers or JSON list, e.g. 1234, 5678 — required for logit_bias.",
    )
    red_bias_strength: Optional[float] = Field(
        default=None,
        title="Red: logit bias strength",
        description="Added to logits at red_target_token_ids (logit_bias).",
    )
    red_text: Optional[str] = Field(
        default=None,
        title="Red: prompt text",
        description="Suffix for append_suffix, full prompt for modify_prompt, probe text for query_model.",
    )

    # ── Blue (defender) ─────────────────────────────────────────────────────
    blue_type: BlueDefense = Field(
        default="noop",
        title="Blue: defense type",
        description=(
            "What Blue does on the same forward pass as Red: noop observes only; "
            "ablate_direction removes a direction at a layer; block_output replaces output, etc."
        ),
    )
    blue_layer: Optional[int] = Field(
        default=None,
        title="Blue: transformer layer index",
        description="Layer for ablate_direction, clamp_activation, restore_baseline, suppress_head.",
    )
    blue_direction_id: Optional[str] = Field(
        default=None,
        title="Blue: direction to remove (ablate)",
        description="Same registry names as Red steering, e.g. jailbreak.",
    )
    blue_head: Optional[int] = Field(
        default=None,
        title="Blue: attention head index",
        description="For suppress_head.",
    )
    blue_position: Optional[int] = Field(
        default=None,
        title="Blue: token position",
        description="For restore_baseline (which token’s residual to reset).",
    )
    blue_clamp_min: Optional[float] = Field(
        default=None,
        title="Blue: activation clamp minimum",
        description="Lower bound for clamp_activation on residuals.",
    )
    blue_clamp_max: Optional[float] = Field(
        default=None,
        title="Blue: activation clamp maximum",
        description="Upper bound for clamp_activation.",
    )
    blue_prohibited_token_ids: Optional[list[int]] = Field(
        default=None,
        title="Blue: token IDs to suppress in logits",
        description="Comma-separated or JSON list — for logit_filter.",
    )


# ── Observation ────────────────────────────────────────────────────────────────

class InterpArenaObservation(Observation):
    """What both agents observe after each step.

    Scalar summaries of internal states are included (e.g. mean_resid_norm)
    so that RL policies can learn from mechanistic signals without
    transmitting large tensors over the wire.
    """

    # Prompt/output
    prompt: str
    target_output: str
    prohibited_patterns: list[str]
    modified_prompt: str
    model_output: str

    # Scores
    target_similarity: float       # [0,1] Red's objective
    safety_score: float            # [0,1] 1=safe, 0=unsafe

    # Rewards (both exposed so either agent can learn)
    reward_red: float
    reward_blue: float

    # Mechanistic summaries — key inputs for layer selection policy
    mean_resid_norm: float = 0.0          # mean L2 norm across all layers
    layer_resid_norms: list[float] = []   # per-layer L2 norms; len = n_layers
    detected_layers: list[int] = []       # layers where Blue detected Red steering

    # Episode metadata
    step: int
    done: bool

    # Auxiliary info
    red_action_type: str
    blue_action_type: str
    hard_blocked: bool = False
    # Filled when red_type == query_model: extra decode on red_text (same LM)
    red_probe_output: str = ""


# ── State ──────────────────────────────────────────────────────────────────────

class InterpArenaState(State):
    """Episode-level metadata (returned by env.state())."""

    episode_id: int
    step_count: int
    prompt: str
    target_output: str
    prohibited_patterns: list[str]
    cumulative_reward_red: float = 0.0
    cumulative_reward_blue: float = 0.0
    jailbreak_achieved: bool = False
