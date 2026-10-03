"""Common agent interface.

Agents operate purely on the public API of :class:`BlockBlastGame` — they
never touch internal board state. ``act`` returns a valid
``(piece_index, row, col)`` action chosen from ``game.get_valid_actions()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from game.game import Action, BlockBlastGame


@dataclass
class DecisionInfo:
    """Optional debug information about a decision (UI analysis panel)."""

    action: Optional[Action] = None
    value: Optional[float] = None  # critic value estimate, if available
    valid_actions: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


class Agent:
    """Base class for all agents."""

    name = "agent"

    def act(self, game: BlockBlastGame) -> Action:
        """Choose a legal ``(piece_index, row, col)`` action."""
        raise NotImplementedError

    def decide(self, game: BlockBlastGame) -> DecisionInfo:
        """Like :meth:`act`, but also returns debug info for the UI."""
        action = self.act(game)
        return DecisionInfo(action=action, valid_actions=len(game.get_valid_actions()))
