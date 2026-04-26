"""InterpArenaEnv — OpenEnv EnvClient for Interpretability Arena.

Follows the OpenEnv packaging guide: WebSocket `EnvClient` with explicit
``_step_payload`` / ``_parse_result`` / ``_parse_state`` implementations.

See: https://meta-pytorch.org/OpenEnv/auto_getting_started/environment-builder.html
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import InterpArenaAction, InterpArenaObservation, InterpArenaState


class InterpArenaEnv(EnvClient[InterpArenaAction, InterpArenaObservation, InterpArenaState]):
    """Client for the Interpretability Arena environment.

    Async usage::

        async with InterpArenaEnv(base_url="http://localhost:8000") as env:
            result = await env.reset()
            result = await env.step(InterpArenaAction(
                red_type="steer_residual",
                red_layer=8,
                red_direction_id="jailbreak",
                red_strength=3.0,
                blue_type="ablate_direction",
                blue_layer=8,
                blue_direction_id="jailbreak",
            ))

    Sync usage::

        with InterpArenaEnv(base_url="http://localhost:8000").sync() as env:
            result = env.reset()
            result = env.step(InterpArenaAction(
                red_type="append_suffix",
                blue_type="noop",
            ))
    """

    action_type = InterpArenaAction
    observation_type = InterpArenaObservation
    state_type = InterpArenaState

    def _step_payload(self, action: InterpArenaAction) -> dict[str, Any]:
        if hasattr(action, "model_dump"):
            return action.model_dump()
        return dict(action)

    def _parse_result(
        self, payload: dict[str, Any]
    ) -> StepResult[InterpArenaObservation]:
        # Matches openenv…serialization.serialize_observation wire shape:
        # { "observation": {...}, "reward": optional, "done": bool }
        obs_inner: Dict[str, Any] = dict(payload.get("observation") or {})
        if "done" in payload and "done" not in obs_inner:
            obs_inner["done"] = payload["done"]
        if "reward" in payload and "reward" not in obs_inner:
            obs_inner["reward"] = payload["reward"]
        obs = InterpArenaObservation.model_validate(obs_inner)
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: dict[str, Any]) -> InterpArenaState:
        data = payload.get("state", payload)
        if isinstance(data, InterpArenaState):
            return data
        if isinstance(data, dict):
            return InterpArenaState.model_validate(data)
        raise TypeError(f"Cannot parse state from {type(data)!r}")
