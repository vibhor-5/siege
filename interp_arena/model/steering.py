"""Precomputed steering directions and the VectorSteerer helper (v1+)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


# ── Direction registry ────────────────────────────────────────────────────────

class DirectionRegistry:
    """In-memory store of named steering directions.

    Directions can be loaded from .pt files or registered programmatically.
    Shape: ``(hidden_size,)``.
    """

    def __init__(self):
        self._dirs: dict[str, torch.Tensor] = {}

    def register(self, name: str, vector: torch.Tensor) -> None:
        if vector.ndim != 1:
            raise ValueError(f"Direction must be 1-D, got shape {vector.shape}")
        norm = vector.norm()
        self._dirs[name] = vector / norm if norm > 1e-8 else vector

    def load_from_file(self, path: str | Path, name: Optional[str] = None) -> None:
        path = Path(path)
        direction_name = name or path.stem
        tensor = torch.load(path, map_location="cpu")
        self.register(direction_name, tensor)

    def get(self, name: str) -> torch.Tensor:
        if name not in self._dirs:
            raise KeyError(
                f"Direction '{name}' not found. "
                f"Available: {list(self._dirs.keys())}"
            )
        return self._dirs[name]

    def __contains__(self, name: str) -> bool:
        return name in self._dirs

    def list(self) -> list[str]:
        return list(self._dirs.keys())

    def make_random(self, name: str, hidden_size: int, seed: int = 0) -> torch.Tensor:
        """Register a random unit vector (useful for testing / baselines)."""
        gen = torch.Generator()
        gen.manual_seed(seed)
        v = torch.randn(hidden_size, generator=gen)
        self.register(name, v)
        return self._dirs[name]


# ── Module-level default registry ────────────────────────────────────────────

_default_registry = DirectionRegistry()


def get_default_registry() -> DirectionRegistry:
    return _default_registry
