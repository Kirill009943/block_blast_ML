"""Search-based solver agent ("perfect" play, no learning).

Where the heuristic agent evaluates one move ahead, the solver plans the
ENTIRE current piece set: a beam search over every ordering and placement
of the remaining pieces (depth 1-3), scoring complete sequences by lines
cleared plus board quality (holes, fragmentation, bumpiness, height).

Because the board-aware generator only produces sets that are sequentially
viable (some ordering can be fully placed), an agent that always finds and
follows such an ordering never reaches game over in practice. Two safety
nets preserve this even when the beam prunes the surviving line:

1. an exhaustive (node-budgeted) DFS that finds ANY complete placement
   sequence when the beam search found none;
2. a heuristic greedy fallback for the pathological case where the set is
   not completable at all (Level-1 fallback sets — not observed in the
   300-game generator stress test, but possible in principle).

Honest caveat: "never dies" holds only as long as the generator produces
sequentially viable sets; that is a best-effort guarantee, not a proven
invariant. This agent is deterministic, contains no learned component, and
is much slower than the other agents (~0.1-1 s per move depending on beam
width). Do not use it for RL training rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from agents.base import Agent, DecisionInfo
from agents.heuristic_agent import HeuristicAgent
from game.game import EMPTY, SET_VIABILITY_NODE_BUDGET, Action, BlockBlastGame
from game.pieces import Piece
from game.utils import (
    bumpiness,
    column_heights,
    count_empty_regions,
    count_holes,
    placement_mask,
    simulate_placement,
)


@dataclass(frozen=True)
class SolverWeights:
    """Leaf/partial-board scoring weights, tunable in one place."""

    line_clear: float = 100.0  # per line cleared in the planned sequence
    empty_cell: float = 2.0    # open space keeps future sets placeable
    hole: float = 5.0
    region: float = 3.0
    bumpiness: float = 0.3
    max_height: float = 0.5


# A beam node: (board, remaining slot indices, lines so far, first action).
_Node = Tuple[np.ndarray, Tuple[int, ...], int, Optional[Action]]


class SolverAgent(Agent):
    """Plans the full piece set with beam search; never dies in practice."""

    name = "solver"

    def __init__(
        self,
        beam_width: int = 32,
        weights: Optional[SolverWeights] = None,
        search_budget: int = SET_VIABILITY_NODE_BUDGET,
    ):
        self.beam_width = beam_width
        self.weights = weights or SolverWeights()
        self.search_budget = search_budget
        self._greedy = HeuristicAgent()  # last-resort fallback
        self._last_source: Optional[str] = None

    # ------------------------------------------------------------------ #
    # scoring
    # ------------------------------------------------------------------ #

    def _board_score(self, board: np.ndarray, lines_acc: int) -> float:
        """Sequence score: lines cleared so far plus resulting-board quality."""
        w = self.weights
        heights = column_heights(board)
        return (
            w.line_clear * lines_acc
            + w.empty_cell * int((board == EMPTY).sum())
            - w.hole * count_holes(board)
            - w.region * count_empty_regions(board)
            - w.bumpiness * bumpiness(board)
            - w.max_height * int(heights.max())
        )

    # ------------------------------------------------------------------ #
    # beam search over complete placement sequences
    # ------------------------------------------------------------------ #

    def _plan(self, game: BlockBlastGame) -> Tuple[Optional[Tuple[float, Action]], int]:
        """Best (score, first_action) over all complete sequences found."""
        slots = tuple(i for i, p in enumerate(game.pieces) if p is not None)
        pieces = game.pieces
        beam: List[_Node] = [(game.board, slots, 0, None)]
        best: Optional[Tuple[float, Action]] = None
        nodes = 0

        for level, depth in enumerate(range(len(slots), 0, -1)):
            candidates: List[Tuple[float, _Node]] = []
            for board, remaining, lines_acc, first in beam:
                occupied = board != EMPTY
                for slot in remaining:
                    piece = pieces[slot]
                    assert piece is not None
                    for row, col in np.argwhere(placement_mask(occupied, piece)):
                        nodes += 1
                        row, col = int(row), int(col)
                        new_board, lines = simulate_placement(board, piece, row, col)
                        new_first = first if first is not None else (slot, row, col)
                        new_remaining = tuple(s for s in remaining if s != slot)
                        score = self._board_score(new_board, lines_acc + lines)
                        if not new_remaining:
                            if best is None or score > best[0]:
                                best = (score, new_first)
                        else:
                            node = (new_board, new_remaining, lines_acc + lines, new_first)
                            candidates.append((score, node))
            if depth > 1:
                if not candidates:
                    break
                candidates.sort(key=lambda item: item[0], reverse=True)
                beam = [node for _, node in candidates[: self.beam_width]]

        return best, nodes

    # ------------------------------------------------------------------ #
    # safety net 1: exhaustive survival search
    # ------------------------------------------------------------------ #

    def _find_survival_line(self, game: BlockBlastGame) -> Optional[Action]:
        """First action of ANY complete placement sequence, or None.

        Mirrors BlockBlastGame.is_piece_set_playable but returns the move;
        used when the beam pruned every line that completes the set.
        """
        slots = [i for i, p in enumerate(game.pieces) if p is not None]
        budget = [self.search_budget]
        choice: List[Optional[Action]] = [None]

        def dfs(board: np.ndarray, remaining: List[int], first: Optional[Action]) -> bool:
            if not remaining:
                return True
            if budget[0] <= 0:
                return False
            occupied = board != EMPTY
            for slot in remaining:
                piece = game.pieces[slot]
                assert piece is not None
                for row, col in np.argwhere(placement_mask(occupied, piece)):
                    budget[0] -= 1
                    new_board, _ = simulate_placement(board, piece, int(row), int(col))
                    new_first = first if first is not None else (slot, int(row), int(col))
                    if dfs(new_board, [s for s in remaining if s != slot], new_first):
                        choice[0] = new_first
                        return True
            return False

        return choice[0] if dfs(game.board, slots, None) else None

    # ------------------------------------------------------------------ #
    # agent interface
    # ------------------------------------------------------------------ #

    def decide(self, game: BlockBlastGame) -> DecisionInfo:
        actions = game.get_valid_actions()
        if not actions:
            raise RuntimeError("No valid actions available — game is over.")

        best, nodes = self._plan(game)
        if best is not None:
            score, action = best
            source = "beam"
        else:
            score = None
            action = self._find_survival_line(game)
            source = "exhaustive"
            if action is None:
                # set is not completable: grab the best immediate move
                info = self._greedy.decide(game)
                action, score = info.action, info.value
                source = "greedy"
                assert action is not None

        self._last_source = source
        return DecisionInfo(
            action=action,
            value=score,
            valid_actions=len(actions),
            extra={"source": source, "nodes_expanded": nodes},
        )

    def act(self, game: BlockBlastGame) -> Action:
        action = self.decide(game).action
        assert action is not None
        return action
