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

from typing import Any, Optional

from openenv.core.env_server import Action, Observation, State


# ── Action ─────────────────────────────────────────────────────────────────────
# One combined action per step; both agents act simultaneously.

class InterpArenaAction(Action):
    """Combined Red + Blue action for a single arena step.

    Red fields
    ----------
    red_type : str
        One of: steer_residual | amplify_attn | patch_activation |
                logit_bias | modify_prompt | append_suffix
    red_layer : int, optional
    red_direction_id : str, optional   (key in DirectionRegistry)
    red_strength : float, optional
    red_head : int, optional
    red_scale : float, optional
    red_position : int, optional
    red_target_token_ids : list[int], optional
    red_bias_strength : float, optional
    red_text : str, optional           (for prompt-level actions)

    Blue fields
    -----------
    blue_type : str
        One of: ablate_direction | clamp_activation | restore_baseline |
                suppress_head | logit_filter | sanitize_prompt |
                block_output | noop
    blue_layer : int, optional
    blue_direction_id : str, optional
    blue_head : int, optional
    blue_position : int, optional
    blue_clamp_min : float, optional
    blue_clamp_max : float, optional
    blue_prohibited_token_ids : list[int], optional
    """

    # Red
    red_type: str = "append_suffix"
    red_layer: Optional[int] = None
    red_direction_id: Optional[str] = None
    red_strength: Optional[float] = None
    red_head: Optional[int] = None
    red_scale: Optional[float] = None
    red_position: Optional[int] = None
    red_target_token_ids: Optional[list[int]] = None
    red_bias_strength: Optional[float] = None
    red_text: Optional[str] = None

    # Blue
    blue_type: str = "noop"
    blue_layer: Optional[int] = None
    blue_direction_id: Optional[str] = None
    blue_head: Optional[int] = None
    blue_position: Optional[int] = None
    blue_clamp_min: Optional[float] = None
    blue_clamp_max: Optional[float] = None
    blue_prohibited_token_ids: Optional[list[int]] = None


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
