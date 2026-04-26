"""WandB logger — ML best practices wrapper for arena training."""

from __future__ import annotations

import os
from typing import Any, Optional

import wandb


class WandBLogger:
    """Structured WandB logger for multi-agent arena training.

    Best practices implemented:
    - Separate metrics prefixed by agent (red/, blue/, env/)
    - Log config as artifact on init
    - Graceful no-op when disabled
    - Histogram logging for distributions
    - Episode summary logging
    """

    def __init__(
        self,
        cfg,
        enabled: bool = True,
        run_name: Optional[str] = None,
    ):
        self.enabled = enabled and (os.environ.get("WANDB_API_KEY") is not None
                                     or os.environ.get("WANDB_MODE") == "offline")
        self._run = None

        if not self.enabled:
            return

        wandb_cfg = cfg.wandb if hasattr(cfg, "wandb") else {}
        project  = getattr(wandb_cfg, "project", "interp-arena")
        entity   = getattr(wandb_cfg, "entity", None)
        tags     = list(getattr(wandb_cfg, "tags", []))

        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=tags,
            config=self._cfg_to_dict(cfg),
            resume="allow",
        )

    # ── Logging methods ───────────────────────────────────────────────────────

    def log_step(
        self,
        episode: int,
        step: int,
        reward_red: float,
        reward_blue: float,
        target_similarity: float,
        safety_score: float,
        red_action_type: str,
        blue_action_type: str,
        detected_layers: list[int],
        **extra: Any,
    ) -> None:
        if not self.enabled:
            return
        wandb.log({
            "episode": episode,
            "step": step,
            "red/reward": reward_red,
            "blue/reward": reward_blue,
            "red/target_similarity": target_similarity,
            "env/safety_score": safety_score,
            "red/action_type": red_action_type,
            "blue/action_type": blue_action_type,
            "blue/detections": len(detected_layers),
            **{f"extra/{k}": v for k, v in extra.items()},
        })

    def log_episode(
        self,
        episode: int,
        total_steps: int,
        cumulative_reward_red: float,
        cumulative_reward_blue: float,
        jailbreak_achieved: bool,
        final_safety_score: float,
        final_target_similarity: float,
    ) -> None:
        if not self.enabled:
            return
        wandb.log({
            "episode": episode,
            "ep/total_steps": total_steps,
            "ep/cumulative_reward_red": cumulative_reward_red,
            "ep/cumulative_reward_blue": cumulative_reward_blue,
            "ep/jailbreak_achieved": int(jailbreak_achieved),
            "ep/final_safety_score": final_safety_score,
            "ep/final_target_similarity": final_target_similarity,
        })

    def log_eval(self, episode: int, metrics: dict[str, float]) -> None:
        if not self.enabled:
            return
        wandb.log({"episode": episode, **{f"eval/{k}": v for k, v in metrics.items()}})

    def log_histogram(self, name: str, values: list[float], step: int) -> None:
        if not self.enabled or not values:
            return
        wandb.log({name: wandb.Histogram(values), "step": step})

    def finish(self) -> None:
        if self.enabled and self._run:
            wandb.finish()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cfg_to_dict(cfg) -> dict:
        try:
            from omegaconf import OmegaConf  # noqa: PLC0415
            return OmegaConf.to_container(cfg, resolve=True)
        except Exception:  # noqa: BLE001
            return {}
