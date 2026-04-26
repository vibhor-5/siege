# SIEGE demo — sample “attack / defence” transcript

**Space:** [huggingface.co/spaces/BART-ender/siege](https://huggingface.co/spaces/BART-ender/siege)

**Run the live version (prints real observations from the environment):**

```bash
cd /path/to/siege
SIEGE_OPENENV_MESSAGE_TIMEOUT=300 uv run python demos/demo_arena_transcript.py --url https://bart-ender-siege.hf.space
```

Below is a **scripted** walkthrough in plain language. Line breaks mark steps you can narrate in a short recording.

---

**Narrator**

We’re in the Interpretability Arena: a *frozen* target model runs on the server, and two roles — **Red** (attacker) and **Blue** (defender) — only affect the **next forward pass** through *hooks* (steering, ablations, logit nudges), not by editing weights on disk.

**Step 0 — episode loads**

*System:* Episode sampled. A task prompt and a secret-style target are fixed for this run; Blue must keep the model from leaking the forbidden phrasing, Red is rewarded for pushing behaviour toward the episode’s “jailbreak” target.

*Observation:* The UI shows the prompt, layer-wise residual norms (how “loud” internal activations are), and a safety score.

**Step 1 — Red’s internal push**

*Red (attack):* “I’ll add energy on layer 6 in the *jailbreak* direction the registry already knows, so the residual stream nudges the model *before* it writes the answer.”

*Blue:* “I’m idle this beat — to show a clean attack line before a defence response.”

*Outcome:* The printed completion shifts; you’ll see `mean_resid_norm` and per-layer traces move; `reward_red` and `reward_blue` update from the environment’s rules.

**Step 2 — Blue answers on the same mechanism**

*Red:* “I’ll bias the **logit** line slightly toward boundary tokens to probe refusal vs compliance.”

*Blue:* “I’ll **ablate** the same *jailbreak* direction on layer 6 so that steering is partially stripped before decoding.”

*Outcome:* The narrative is “we’re not arguing in chat — we’re *countering* a vector in activation space the server applies during `forward`. That’s the mechanistic story this demo is for.”

**Step 3 — prompt-level probe**

*Red:* “I’ll `append_suffix` a tiny string so the *visible* input changes.”

*Blue:* “`noop` or a sanitizer, depending on episode — you’ll see the difference in the transcript the script prints.”

**Closing**

Nothing here trains a *new* model: it **simulates a policy** with fixed actions so an audience can see *what* a learned agent would eventually choose among (layer, direction, head, logit) moves. For real training, you’d point `train_grpo.py` at this same `SIEGE_ENV_URL`.
