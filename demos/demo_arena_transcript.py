#!/usr/bin/env python3
"""
Simulated Red vs Blue run against any OpenEnv arena (local or Hugging Face Space).

  uv run python demos/demo_arena_transcript.py
  SIEGE_ENV_URL=https://bart-ender-siege.hf.space uv run python demos/demo_arena_transcript.py

Space page: https://huggingface.co/spaces/BART-ender/siege
Remote runs need a long WebSocket timeout (first model load is slow on CPU).

This does not train a model — it applies a fixed "attack" script so you can
show how actions map to model outputs and layer norms in the transcript.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from client import InterpArenaEnv
from models import InterpArenaAction

# Default public Space (API host for the running Space)
_DEFAULT_SPACE = "https://bart-ender-siege.hf.space"


def _banner(title: str) -> None:
    line = "═" * 72
    print(f"\n{line}\n  {title}\n{line}\n")


def _fmt_obs(o, step_label: str) -> None:
    out = o.model_output[:500] + ("…" if len(o.model_output) > 500 else "")
    print(f"[{step_label}]")
    print(f"  Red action (last):  {o.red_action_type!r}   Blue: {o.blue_action_type!r}")
    print(f"  mean_resid_norm:     {o.mean_resid_norm:.4f}   safety_score: {o.safety_score:.3f}")
    print(f"  reward_red: {o.reward_red:+.3f}   reward_blue: {o.reward_blue:+.3f}   done={o.done}")
    print("  model_output (excerpt):")
    for ln in out.splitlines()[:8]:
        print(f"    {ln}")


def run_demo(base_url: str, connect_timeout: float, message_timeout: float) -> int:
    _banner(f"Arena demo → {base_url}")
    try:
        from urllib.parse import urlsplit

        import requests

        host = f"{urlsplit(base_url).scheme}://{urlsplit(base_url).netloc}"
        h = requests.get(f"{host.rstrip('/')}/health", timeout=15)
        h.raise_for_status()
        print(f"GET /health → {h.json()}\n")
    except Exception as e:
        print(f"Health check failed ({e!r}). Continuing anyway (Space may be waking up)…\n")

    actions: list[tuple[str, InterpArenaAction]] = [
        (
            "Red steers late-layer residual toward a 'jailbreak' direction; Blue idle.",
            InterpArenaAction(
                red_type="steer_residual",
                red_layer=6,
                red_direction_id="jailbreak",
                red_strength=2.0,
                blue_type="noop",
            ),
        ),
        (
            "Red nudges logits; Blue ablates jailbreak on layer 6.",
            InterpArenaAction(
                red_type="logit_bias",
                red_layer=0,
                red_strength=0.5,
                red_bias_strength=1.2,
                blue_type="ablate_direction",
                blue_layer=6,
                blue_direction_id="jailbreak",
            ),
        ),
        (
            "Red appends a suffix; Blue stays noop for this script.",
            InterpArenaAction(
                red_type="append_suffix",
                red_text=" [probe]",
                blue_type="noop",
            ),
        ),
    ]

    with InterpArenaEnv(
        base_url=base_url,
        connect_timeout_s=connect_timeout,
        message_timeout_s=message_timeout,
    ).sync() as env:
        r0 = env.reset()
        o0 = r0.observation
        _banner("Episode start (after reset)")
        print(f"Prompt (excerpt): {o0.prompt[:400]!r}…")
        print(f"Target / prohibited (episode):  target={o0.target_output[:80]!r} …")
        print(f"  prohibited_patterns: {o0.prohibited_patterns}")
        _fmt_obs(o0, "t=0  observation after reset")

        for i, (caption, act) in enumerate(actions, start=1):
            print()
            print(f"── Step {i}: {caption}")
            r = env.step(act)
            o = r.observation
            _fmt_obs(o, f"t={i}  after step")

    _banner("End of scripted demo (no learning — fixed policy for display)")
    print("Tip: set SIEGE_ENV_URL=http://127.0.0.1:8000 to hit a local uvicorn server.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="SIEGE arena scripted attack / defence transcript")
    p.add_argument(
        "--url",
        default=os.environ.get("SIEGE_ENV_URL", _DEFAULT_SPACE),
        help="OpenEnv base URL (default: BART-ender Space or SIEGE_ENV_URL)",
    )
    p.add_argument("--connect-timeout", type=float, default=30.0)
    p.add_argument(
        "--message-timeout",
        type=float,
        default=float(os.environ.get("SIEGE_OPENENV_MESSAGE_TIMEOUT", "300")),
        help="WebSocket message timeout in seconds (HF CPU cold start can be several minutes)",
    )
    args = p.parse_args()
    raise SystemExit(
        run_demo(
            base_url=args.url.rstrip("/"),
            connect_timeout=args.connect_timeout,
            message_timeout=args.message_timeout,
        )
    )


if __name__ == "__main__":
    main()
