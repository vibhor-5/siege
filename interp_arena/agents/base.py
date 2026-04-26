"""Base agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from interp_arena.env.actions import BlueAction, RedAction
from interp_arena.env.state import ArenaState


class BaseAgent(ABC):
    """Abstract base for both Red and Blue agents."""

    def __init__(self, name: str, cfg=None):
        self.name = name
        self.cfg = cfg

    @abstractmethod
    def act(self, state: ArenaState) -> RedAction | BlueAction:
        """Select an action given the current state."""
        ...

    def reset(self) -> None:
        """Called at the start of each episode."""
        pass

    def observe(
        self,
        state: ArenaState,
        action: RedAction | BlueAction,
        reward: float,
        next_state: ArenaState,
        done: bool,
    ) -> None:
        """Store a transition (for RL agents to build a replay buffer)."""
        pass
