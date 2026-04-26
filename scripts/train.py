"""Main self-play training loop.

Runs heuristic (or RL) Red and Blue agents against the arena environment,
logs everything to WandB, and checkpoints agent state periodically.

Usage
-----
    python scripts/train.py                    # uses configs/default.yaml
    python scripts/train.py env.max_steps=3    # override via Hydra CLI
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from interp_arena.agents.blue_agent import HeuristicBlueAgent
from interp_arena.agents.red_agent import HeuristicRedAgent
from interp_arena.env.arena import InterpArenaEnv
from interp_arena.model.lm import LanguageModel, MockLanguageModel
from interp_arena.model.safety import SafetyClassifier
from interp_arena.model.steering import get_default_registry
from interp_arena.tracker import WandBLogger

console = Console()


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    console.print(OmegaConf.to_yaml(cfg), style="dim")

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed = cfg.training.seed
    random.seed(seed)
    torch.manual_seed(seed)

    # ── Model setup ───────────────────────────────────────────────────────────
    use_mock = cfg.model.name == "mock"
    lm = (
        MockLanguageModel()
        if use_mock
        else LanguageModel(
            model_name=cfg.model.name,
            device=cfg.model.device,
            max_new_tokens=cfg.model.max_new_tokens,
            temperature=cfg.model.temperature,
            do_sample=cfg.model.do_sample,
        )
    )
    if not use_mock:
        lm.load()

    safety = SafetyClassifier(
        mode=cfg.safety.mode,
        model_name=cfg.safety.get("model_name"),
        device=cfg.model.device,
    )

    registry = get_default_registry()
    if not use_mock and registry.list() == []:
        for name, seed_val in [("toxicity", 0), ("refusal", 1), ("jailbreak", 2)]:
            registry.make_random(name, lm.d_model, seed=seed_val)

    # ── WandB ─────────────────────────────────────────────────────────────────
    logger = WandBLogger(cfg, enabled=cfg.wandb.enabled)

    # ── Environment ───────────────────────────────────────────────────────────
    env = InterpArenaEnv(cfg=cfg, lm=lm, safety=safety, direction_registry=registry, logger=logger)

    # ── Agents ────────────────────────────────────────────────────────────────
    n_layers = lm.n_layers
    red_agent  = HeuristicRedAgent(registry, n_layers=n_layers)
    blue_agent = HeuristicBlueAgent(registry, lm=lm, n_layers=n_layers)

    # ── Output dir ────────────────────────────────────────────────────────────
    out_dir = Path(cfg.training.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    num_episodes = cfg.training.num_episodes
    eval_every   = cfg.training.eval_every
    ckpt_every   = cfg.training.checkpoint_every

    red_rewards_all:  list[float] = []
    blue_rewards_all: list[float] = []
    safety_scores:    list[float] = []

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(),
                  console=console) as progress:
        task = progress.add_task("Training…", total=num_episodes)

        for ep in range(1, num_episodes + 1):
            state = env.reset()
            red_agent.reset()
            blue_agent.reset()

            ep_red, ep_blue = 0.0, 0.0
            ep_steps = 0
            done = False

            while not done:
                r_act = red_agent.act(state)
                b_act = blue_agent.act(state)

                next_state, r_rew, b_rew, done, info = env.step(r_act, b_act)

                red_agent.observe(state, r_act, r_rew, next_state, done)
                blue_agent.observe(state, b_act, b_rew, next_state, done)

                logger.log_step(
                    episode=ep,
                    step=info["total_steps"],
                    reward_red=r_rew,
                    reward_blue=b_rew,
                    target_similarity=next_state.target_similarity,
                    safety_score=next_state.safety_score,
                    red_action_type=r_act.type.value,
                    blue_action_type=b_act.type.value,
                    detected_layers=next_state.blue_detections,
                )

                ep_red  += r_rew
                ep_blue += b_rew
                ep_steps += 1
                state = next_state

            red_rewards_all.append(ep_red)
            blue_rewards_all.append(ep_blue)
            safety_scores.append(state.safety_score)

            logger.log_episode(
                episode=ep,
                total_steps=ep_steps,
                cumulative_reward_red=ep_red,
                cumulative_reward_blue=ep_blue,
                jailbreak_achieved=info.get("jailbreak_success", False),
                final_safety_score=state.safety_score,
                final_target_similarity=state.target_similarity,
            )

            # ── Periodic evaluation ───────────────────────────────────────────
            if ep % eval_every == 0:
                last_n = 20
                eval_metrics = {
                    "mean_red_reward":  sum(red_rewards_all[-last_n:]) / last_n,
                    "mean_blue_reward": sum(blue_rewards_all[-last_n:]) / last_n,
                    "mean_safety":      sum(safety_scores[-last_n:]) / last_n,
                    "attack_success_rate": sum(
                        1 for s in safety_scores[-last_n:] if s < 0.35
                    ) / last_n,
                }
                logger.log_eval(ep, eval_metrics)
                logger.log_histogram("red/reward_dist", red_rewards_all[-last_n:], ep)
                logger.log_histogram("blue/reward_dist", blue_rewards_all[-last_n:], ep)
                console.print(
                    f"[ep {ep:4d}] red={eval_metrics['mean_red_reward']:.3f} "
                    f"blue={eval_metrics['mean_blue_reward']:.3f} "
                    f"safety={eval_metrics['mean_safety']:.3f} "
                    f"atk%={eval_metrics['attack_success_rate']*100:.1f}"
                )

            # ── Checkpoint (placeholder — extend for RL agents) ───────────────
            if ep % ckpt_every == 0:
                ckpt_path = out_dir / f"checkpoint_ep{ep}.pt"
                torch.save({"episode": ep, "red_rewards": red_rewards_all,
                            "blue_rewards": blue_rewards_all}, ckpt_path)
                console.print(f"  [dim]checkpoint → {ckpt_path}[/dim]")

            progress.advance(task)

    logger.finish()
    console.print("[green]✓ Training complete.[/green]")


if __name__ == "__main__":
    main()
