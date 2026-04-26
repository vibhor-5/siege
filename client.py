"""InterpArenaEnv — OpenEnv EnvClient for Interpretability Arena."""

from __future__ import annotations

from openenv.core.env_client import EnvClient

from models import InterpArenaAction, InterpArenaObservation, InterpArenaState


class InterpArenaEnv(EnvClient):
    """Client for the Interpretability Arena environment.

    Async usage::

        async with InterpArenaEnv(base_url="http://localhost:8000") as env:
            obs = await env.reset()
            result = await env.step(InterpArenaAction(
                red_type="steer_residual",
                red_layer=8,
                red_direction_id="jailbreak",
                red_strength=3.0,
                blue_type="ablate_direction",
                blue_layer=8,
                blue_direction_id="jailbreak",
            ))
            print(result.observation.model_output)
            print("Red reward:", result.reward)

    Sync usage::

        with InterpArenaEnv(base_url="http://localhost:8000").sync() as env:
            obs = env.reset()
            result = env.step(InterpArenaAction(red_type="noop", blue_type="noop"))
    """

    action_type = InterpArenaAction
    observation_type = InterpArenaObservation
    state_type = InterpArenaState
