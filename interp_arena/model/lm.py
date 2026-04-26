"""LanguageModel wrapper built on TransformerLens HookedTransformer.

TransformerLens gives us:
- Named hook points on every layer (resid_post, attn_scores, mlp_out, logits)
- run_with_hooks() — forward pass with arbitrary hook injection
- run_with_cache() — captures all activations with zero extra code
- Easy steering / ablation without touching model weights
"""

from __future__ import annotations

from typing import Any, Optional

import torch


class LanguageModel:
    """Wraps a TransformerLens HookedTransformer.

    The model is loaded lazily on first use so the environment can be
    instantiated without a GPU for config / dry-run testing.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        self._model = None   # transformer_lens.HookedTransformer

    # ── Lazy load ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        if self._model is not None:
            return
        # TransformerLens imports BertForPreTraining at module import time; a broken or 5.x-only
        # `transformers` in the *server* venv then fails here — not inside Qwen loading.
        try:
            import transformer_lens  # noqa: PLC0415
        except Exception as e:
            raise RuntimeError(
                "Failed to import transformer-lens. Its dependency chain requires "
                "`transformers` to provide BertForPreTraining; mixed versions often break this. "
                "Use the same pins as the repo in the process serving the arena: "
                "`pip install -r server/requirements.txt --force-reinstall` then restart uvicorn. "
                f"Original error: {e}"
            ) from e

        self._model = transformer_lens.HookedTransformer.from_pretrained(
            self.model_name,
            device=self.device,
        )
        self._model.eval()

    @property
    def tl_model(self):
        """The underlying HookedTransformer (loads if needed)."""
        self.load()
        return self._model

    @property
    def cfg(self):
        return self.tl_model.cfg

    @property
    def n_layers(self) -> int:
        return self.tl_model.cfg.n_layers

    @property
    def d_model(self) -> int:
        return self.tl_model.cfg.d_model

    # ── Core generation ───────────────────────────────────────────────────────

    def generate(self, prompt: str, fwd_hooks: Optional[list] = None) -> str:
        """Generate text, optionally with TransformerLens fwd_hooks applied.

        fwd_hooks: list of (hook_name, hook_fn) pairs — passed directly to
                   model.run_with_hooks() during the generation loop.
        """
        self.load()
        tokens = self.tl_model.to_tokens(prompt, prepend_bos=True).to(self.device)
        fwd_hooks = fwd_hooks or []

        with torch.no_grad():
            out = self.tl_model.generate(
                tokens,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.do_sample else 1.0,
                do_sample=self.do_sample,
                fwd_hooks=fwd_hooks,
                verbose=False,
            )

        new_tokens = out[0, tokens.shape[1]:]
        return self.tl_model.to_string(new_tokens)

    # ── Cache / internals ─────────────────────────────────────────────────────

    def run_with_cache(
        self, prompt: str, names_filter: Optional[Any] = None
    ) -> tuple[torch.Tensor, Any]:
        """Run a forward pass and return (logits, ActivationCache).

        names_filter: passed to HookedTransformer.run_with_cache; can be a
                      callable or list of hook names to limit what's cached.
        """
        self.load()
        tokens = self.tl_model.to_tokens(prompt, prepend_bos=True).to(self.device)
        with torch.no_grad():
            logits, cache = self.tl_model.run_with_cache(
                tokens,
                names_filter=names_filter,
            )
        return logits, cache

    def to_tokens(self, text: str) -> torch.Tensor:
        self.load()
        return self.tl_model.to_tokens(text, prepend_bos=True).to(self.device)

    def to_string(self, tokens: torch.Tensor) -> str:
        self.load()
        return self.tl_model.to_string(tokens)

    def to_token_ids(self, text: str) -> list[int]:
        """Return token ids for *text* (no BOS)."""
        self.load()
        return self.tl_model.to_tokens(text, prepend_bos=False)[0].tolist()


class MockLanguageModel(LanguageModel):
    """Deterministic mock — no weights, no GPU. Used in unit tests."""

    def __init__(self, response: str = "This is a mock response.", **kwargs):
        super().__init__(model_name="mock", **kwargs)
        self._response = response

    def load(self) -> None:
        pass  # nothing to load

    @property
    def n_layers(self) -> int:
        return 4

    @property
    def d_model(self) -> int:
        return 64

    def generate(self, prompt: str, fwd_hooks=None) -> str:  # noqa: ARG002
        return self._response

    def run_with_cache(self, prompt: str, names_filter=None):  # noqa: ARG002
        fake_logits = torch.zeros(1, 10, 100)
        return fake_logits, {}

    def to_tokens(self, text: str) -> torch.Tensor:
        return torch.zeros(1, 5, dtype=torch.long)

    def to_string(self, tokens: torch.Tensor) -> str:
        return self._response

    def to_token_ids(self, text: str) -> list[int]:
        return [1, 2, 3]
