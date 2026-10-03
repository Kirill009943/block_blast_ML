"""Hand-crafted heuristic baseline agent.

For every valid action the agent simulates the placement on a copy of the
board and scores the resulting position with a linear heuristic:

* ``+line_clear`` per cleared line (plus ``combo`` per extra simultaneous
  line) — clearing lines is how you score.
* ``+future_move`` per placement remaining for the other pieces — keeping
  options open means surviving longer.
* ``-hole`` per hole (empty cell trapped under occupied cells).
* ``-region`` per disconnected empty region — fragmented boards are hard
  to fill.
* ``-bumpiness`` per unit of height difference between adjacent columns.
* ``-max_height`` for the tallest stack — flat, low boards are safer.

Weights live in :class:`HeuristicWeights` so they can be tuned in one
place. This agent is a benchmark only — it contains no learned component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agents.base import Agent, DecisionInfo
from game.game import Action, BlockBlastGame
from game.utils import (
    bumpiness,
    column_heights,
    count_empty_regions,
    count_holes,
    count_valid_placements,
    simulate_placement,
)


@dataclass(frozen=True)
class HeuristicWeights:
    line_clear: float = 12.0
    combo: float = 8.0  # per additional simultaneous line
    future_move: float = 0.15
    hole: float = 1.0
    region: float = 0.3
    bumpiness: float = 0.1
    max_height: float = 0.2


class HeuristicAgent(Agent):
    """Picks the valid action with the best heuristic score."""

    name = "heuristic"

    def __init__(self, weights: Optional[HeuristicWeights] = None):
        self.weights = weights or HeuristicWeights()
        self._last_score: Optional[float] = None

    def _evaluate(self, game: BlockBlastGame, action: Action) -> float:
        piece_idx, row, col = action
        piece = game.pieces[piece_idx]
        assert piece is not None
        w = self.weights

        new_board, lines = simulate_placement(game.board, piece, row, col)

        remaining = [p for i, p in enumerate(game.pieces) if p is not None and i != piece_idx]
        future_moves = sum(count_valid_placements(new_board, p) for p in remaining)
        if not remaining:
            # a fresh set of three random pieces will be drawn — always playable
            future_moves = 1

        heights = column_heights(new_board)
        score = (
            w.line_clear * lines
            + w.combo * max(0, lines - 1)
            + w.future_move * future_moves
            - w.hole * count_holes(new_board)
            - w.region * count_empty_regions(new_board)
            - w.bumpiness * bumpiness(new_board)
            - w.max_height * int(heights.max())
        )
        return score

    def decide(self, game: BlockBlastGame) -> DecisionInfo:
        actions = game.get_valid_actions()
        if not actions:
            raise RuntimeError("No valid actions available — game is over.")
        scored = [(self._evaluate(game, a), a) for a in actions]
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_action = scored[0]
        self._last_score = best_score
        return DecisionInfo(
            action=best_action,
            value=best_score,
            valid_actions=len(actions),
            extra={"top_candidates": scored[:5]},
        )

    def act(self, game: BlockBlastGame) -> Action:
        action = self.decide(game).action
        assert action is not None
        return action
