"""Baseline agent that picks a uniformly random valid action."""

from __future__ import annotations

from typing import Optional

import numpy as np

from agents.base import Agent
from game.game import Action, BlockBlastGame


class RandomAgent(Agent):
    """Chooses uniformly at random among the currently valid actions."""

    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def act(self, game: BlockBlastGame) -> Action:
        actions = game.get_valid_actions()
        if not actions:
            raise RuntimeError("No valid actions available — game is over.")
        return actions[int(self._rng.integers(0, len(actions)))]
