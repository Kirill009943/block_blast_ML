"""Pure Block Blast game engine.

No console I/O, no rendering, no ML. Used by the human UI, the Gymnasium
environment and every agent. All state mutations go through this class;
agents must never manipulate the board directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import itertools

import numpy as np

from game.pieces import NUM_COLORS, PIECE_TYPES, Cell, Piece, rotate_cells
from game.utils import placement_mask

BOARD_SIZE = 8
EMPTY = 0

Action = Tuple[int, int, int]  # (piece_index, row, col)

# Board-aware piece generation limits. Generation is rejection sampling
# from the ORIGINAL random piece distribution — the generator never picks
# "easy" pieces on purpose, it only rejects impossible ones.
MAX_GENERATION_ATTEMPTS = 100  # per candidate piece, before type fallback
MAX_SET_ATTEMPTS = 50  # per 3-piece set, before relaxing Level-2 viability
SET_VIABILITY_NODE_BUDGET = 5000  # DFS budget for set viability checks


@dataclass(frozen=True)
class GameConfig:
    """Tunable game/scoring parameters in one place.

    Defaults match classic Block Blast: points for every placed cell, 10
    points per cleared line, and a bonus for clearing several lines at once.
    Set ``points_per_cell=0, combo_bonus=0`` to reproduce the original
    console game's pure ``10 * lines`` scoring.
    """

    board_size: int = BOARD_SIZE
    pieces_per_round: int = 3
    points_per_cell: int = 1
    points_per_line: int = 10
    combo_bonus: int = 10  # extra points per additional simultaneous line


@dataclass(frozen=True)
class MoveResult:
    """Outcome of a :meth:`BlockBlastGame.place_piece` call."""

    success: bool
    cells_placed: int = 0
    rows_cleared: int = 0
    cols_cleared: int = 0
    points_gained: int = 0
    pieces_refreshed: bool = False
    game_over: bool = False

    @property
    def lines_cleared(self) -> int:
        return self.rows_cleared + self.cols_cleared


class BlockBlastGame:
    """8x8 Block Blast game.

    The board is a ``numpy.int8`` grid: 0 = empty, 1..8 = color id.
    Piece generation is driven by a seeded ``numpy.random.Generator`` so a
    fixed seed reproduces the exact piece sequence.
    """

    def __init__(self, seed: Optional[int] = None, config: Optional[GameConfig] = None):
        self.config = config or GameConfig()
        self._rng = np.random.default_rng(seed)
        self.board = np.zeros((self.config.board_size, self.config.board_size), dtype=np.int8)
        self.score = 0
        self.moves = 0
        self.total_lines_cleared = 0
        self.pieces: List[Optional[Piece]] = []
        self._valid_cache: Optional[List[Action]] = None
        self._playable_types_cache: Optional[List[int]] = None
        # generation diagnostics (resettable via reset_generation_stats)
        self.generation_stats: Dict[str, int] = {
            "sets_generated": 0,
            "candidate_attempts": 0,
            "candidates_rejected": 0,
            "max_attempts_single_piece": 0,
            "type_fallbacks": 0,
            "set_attempts": 0,
            "sets_regenerated": 0,
            "viable_sets": 0,
            "level1_fallbacks": 0,
            "game_over_boards": 0,
        }
        self.reset()

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset board, score and pieces. The RNG is NOT reseeded."""
        self.board[:] = EMPTY
        self.score = 0
        self.moves = 0
        self.total_lines_cleared = 0
        self.pieces = self.generate_piece_set()
        self._valid_cache = None
        self._playable_types_cache = None

    def reset_generation_stats(self) -> None:
        for key in self.generation_stats:
            self.generation_stats[key] = 0

    def _draw_pieces(self) -> List[Optional[Piece]]:
        """Backward-compatible alias for :meth:`generate_piece_set`."""
        return self.generate_piece_set()

    def reseed(self, seed: Optional[int]) -> None:
        """Replace the RNG. Call before :meth:`reset` for reproducibility."""
        self._rng = np.random.default_rng(seed)

    def _create_piece(self, type_id: int, rotation: int = 0) -> Piece:
        """Create a random-colored piece with the requested rotation.

        Rotation is applied in 90-degree steps; cells are normalized so the
        bounding box always starts at (0, 0).
        """
        color = int(self._rng.integers(1, NUM_COLORS + 1))
        name, cells = PIECE_TYPES[type_id]
        cells = rotate_cells(cells, rotation)

        rotation_name = (rotation % 4) * 90
        display_name = (
            name if rotation % 4 == 0
            else f"{name}_rot{rotation_name}"
        )

        return Piece(
            type_id=type_id,
            name=display_name,
            cells=cells,
            color=color,
        )

    # ------------------------------------------------------------------ #
    # board-aware piece generation
    #
    # The generator produces RANDOM pieces subject to playability
    # constraints — never "helpful" pieces. Two guarantee levels:
    #
    # Level 1 (hard guarantee): every generated piece has at least one
    #   legal placement on the current board. Achieved by rejection
    #   sampling from the original distribution, falling back to a uniform
    #   choice among the piece types that are playable at all.
    #
    # Level 2 (best-effort preference): the whole set is sequentially
    #   viable — some ordering of the pieces can actually be placed in
    #   sequence (with line clears in between). Achieved by regenerating
    #   the set up to MAX_SET_ATTEMPTS times; if no viable set is found,
    #   the last Level-1-valid set is kept (counted as a fallback).
    #
    # If NO piece type fits the board, the board is genuinely dead:
    # nothing is generated and the game is over.
    # ------------------------------------------------------------------ #

    def has_valid_move_for_piece(self, piece: Piece) -> bool:
        """True if ``piece`` has at least one legal placement on the board."""
        return bool(placement_mask(self.board != EMPTY, piece).any())

    def get_playable_piece_types(self) -> List[int]:
        """Piece type ids with at least one valid placement on the board.

        A type counts as playable when ANY of its rotations fits. Cached per
        board state; invalidated together with the valid-action cache on
        every mutation.
        """
        if self._playable_types_cache is not None:
            return self._playable_types_cache
        occupied = self.board != EMPTY
        playable = []
        for type_id, (name, cells) in enumerate(PIECE_TYPES):
            seen = set()
            for rotation in range(4):
                rotated = rotate_cells(cells, rotation)
                if rotated in seen:  # symmetric shapes repeat orientations
                    continue
                seen.add(rotated)
                probe = Piece(type_id=type_id, name=name, cells=rotated, color=1)
                if placement_mask(occupied, probe).any():
                    playable.append(type_id)
                    break
        self._playable_types_cache = playable
        return playable

    def _mask_for_cells(
        self,
        masks: Dict[Tuple[Cell, ...], np.ndarray],
        cells: Tuple[Cell, ...],
        occupied: np.ndarray,
    ) -> np.ndarray:
        """Placement mask for a normalized ``cells`` shape, lazily cached.

        Keyed by the cells themselves, so different rotations of the same
        type cache independently while identical shapes share one entry.
        """
        mask = masks.get(cells)
        if mask is None:
            probe = Piece(type_id=-1, name="probe", cells=cells, color=1)
            mask = placement_mask(occupied, probe)
            masks[cells] = mask
        return mask

    def _mask_for_type(
        self, masks: Dict[Tuple[Cell, ...], np.ndarray], type_id: int, occupied: np.ndarray
    ) -> np.ndarray:
        """Rotation-0 placement mask for ``type_id`` (see :meth:`_mask_for_cells`)."""
        return self._mask_for_cells(masks, PIECE_TYPES[type_id][1], occupied)


    def generate_playable_piece(
        self, masks: Optional[Dict[Tuple[Cell, ...], np.ndarray]] = None
    ) -> Optional[Piece]:
        """One random playable piece with a random rotation, or None if the
        board is dead.

        Type and rotation are sampled from the original distribution; a
        candidate is rejected only when that exact rotated shape cannot be
        placed on the current board. ``masks`` optionally provides a (lazily
        filled) per-shape placement-mask cache for the current board.
        """
        stats = self.generation_stats
        occupied = self.board != EMPTY
        if masks is None:
            masks = {}
        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            type_id = int(self._rng.integers(0, len(PIECE_TYPES)))
            rotation = int(self._rng.integers(0, 4))
            stats["candidate_attempts"] += 1
            cells = rotate_cells(PIECE_TYPES[type_id][1], rotation)
            if self._mask_for_cells(masks, cells, occupied).any():
                stats["max_attempts_single_piece"] = max(
                    stats["max_attempts_single_piece"], attempt
                )
                return self._create_piece(type_id, rotation)
            stats["candidates_rejected"] += 1
        # fallback: never produce an impossible piece. Uniform choice among
        # every distinct playable (type, rotation) shape; playability is
        # tested on the cached masks, the RNG is only drawn for the pick.
        playable: List[Tuple[int, Tuple[Cell, ...]]] = []
        seen = set()
        for type_id in range(len(PIECE_TYPES)):
            for rotation in range(4):
                cells = rotate_cells(PIECE_TYPES[type_id][1], rotation)
                if cells in seen:
                    continue
                seen.add(cells)
                if self._mask_for_cells(masks, cells, occupied).any():
                    playable.append((type_id, cells))
        if not playable:
            return None
        stats["type_fallbacks"] += 1
        type_id, cells = playable[int(self._rng.integers(0, len(playable)))]
        color = int(self._rng.integers(1, NUM_COLORS + 1))
        return Piece(
            type_id=type_id,
            name=PIECE_TYPES[type_id][0],
            cells=cells,
            color=color,
        )

    def generate_piece_set(self, count: Optional[int] = None) -> List[Optional[Piece]]:
        """A set of ``count`` playable pieces (default: pieces_per_round).

        Prefers sequentially viable sets (Level 2) via
        :meth:`is_piece_set_playable`; always guarantees Level 1. Returns
        all-None slots when no piece type fits the board at all — that is
        a genuine game-over board, not a generation failure.
        """
        count = count or self.config.pieces_per_round
        stats = self.generation_stats
        occupied = self.board != EMPTY
        masks: Dict[Tuple[Cell, ...], np.ndarray] = {}
        # dead-board check with early exit: on a live board the first type
        # almost always fits, so this usually costs a single mask
        board_alive = any(
            self._mask_for_type(masks, t, occupied).any()
            for t in range(len(PIECE_TYPES))
        )
        if not board_alive:
            stats["game_over_boards"] += 1
            return [None] * count

        stats["sets_generated"] += 1
        best_set: List[Optional[Piece]] = []
        for attempt in range(1, MAX_SET_ATTEMPTS + 1):
            stats["set_attempts"] += 1
            candidate = [self.generate_playable_piece(masks) for _ in range(count)]
            best_set = candidate
            if self.is_piece_set_playable(candidate, masks=masks):
                stats["viable_sets"] += 1
                if attempt > 1:
                    stats["sets_regenerated"] += 1
                return candidate
        # fallback: keep the last Level-1-valid set (still never impossible)
        stats["level1_fallbacks"] += 1
        return best_set

    def is_piece_set_playable(
        self,
        pieces: Sequence[Optional[Piece]],
        board: Optional[np.ndarray] = None,
        node_budget: int = SET_VIABILITY_NODE_BUDGET,
        masks: Optional[Dict[Tuple[Cell, ...], np.ndarray]] = None,
    ) -> bool:
        """True if some ordering of ``pieces`` can be placed sequentially.

        DFS over all 3! orderings; at each level the piece is placed on a
        simulated board (lines cleared) before the next piece is tried.
        A node budget caps the search: when it is exhausted the set is
        ACCEPTED (best-effort — Level 1 remains the hard guarantee), so
        generation never stalls on pathological boards.
        """
        from game.utils import simulate_placement

        base = self.board if board is None else board
        pieces = [p for p in pieces if p is not None]
        if not pieces:
            return True

        # Level 1 pre-check: skip DFS work when a piece already fails alone.
        # Keyed by cells, not type_id: rotations of the same type differ.
        base_occupied = base != EMPTY
        pre: Dict[Tuple[Cell, ...], np.ndarray] = {}
        if masks is not None and board is None:
            pre = masks  # reuse the generation-time cache (same board)
        for piece in pieces:
            if not self._mask_for_cells(pre, piece.cells, base_occupied).any():
                return False

        nodes = [0]
        unknown = [False]

        def dfs(index: int, b: np.ndarray, order: Tuple[Piece, ...]) -> bool:
            if index == len(order):
                return True
            if nodes[0] >= node_budget:
                unknown[0] = True
                return False
            piece = order[index]
            occupied = b != EMPTY
            for row, col in np.argwhere(placement_mask(occupied, piece)):
                nodes[0] += 1
                new_board, _ = simulate_placement(b, piece, int(row), int(col))
                if dfs(index + 1, new_board, order):
                    return True
                if unknown[0]:
                    return False
            return False

        for order in itertools.permutations(pieces):
            if dfs(0, base, order):
                return True
            if unknown[0]:
                return True  # budget exhausted: accept (best-effort)
        return False

    # ------------------------------------------------------------------ #
    # state access
    # ------------------------------------------------------------------ #

    def get_state(self) -> Dict:
        """Return a snapshot of the full game state (copies, safe to keep)."""
        return {
            "board": self.board.copy(),
            "pieces": list(self.pieces),
            "used": [p is None for p in self.pieces],
            "score": self.score,
            "moves": self.moves,
            "lines_cleared": self.total_lines_cleared,
            "game_over": self.is_game_over(),
        }

    def get_score(self) -> int:
        return self.score

    def occupancy(self) -> np.ndarray:
        """Binary (float32) view of the board: 1 where a cell is occupied."""
        return (self.board != EMPTY).astype(np.float32)

    # ------------------------------------------------------------------ #
    # rules
    # ------------------------------------------------------------------ #

    def can_place(self, piece: Piece, row: int, col: int) -> bool:
        """True if ``piece`` fits at top-left position ``(row, col)``."""
        size = self.config.board_size
        for r, c in piece.cells:
            rr, cc = row + r, col + c
            if rr < 0 or rr >= size or cc < 0 or cc >= size:
                return False
            if self.board[rr, cc] != EMPTY:
                return False
        return True

    def get_valid_actions(self) -> List[Action]:
        """Every currently legal ``(piece_index, row, col)`` action.

        The result is cached and invalidated on every board/piece change;
        do not mutate the returned list.
        """
        if self._valid_cache is not None:
            return self._valid_cache
        occupied = self.board != EMPTY
        actions: List[Action] = []
        for idx, piece in enumerate(self.pieces):
            if piece is None:
                continue
            for row, col in np.argwhere(placement_mask(occupied, piece)):
                actions.append((idx, int(row), int(col)))
        self._valid_cache = actions
        return actions

    def place_piece(self, piece_index: int, row: int, col: int) -> MoveResult:
        """Place piece ``piece_index`` at ``(row, col)``.

        Returns a :class:`MoveResult`; ``success`` is False (and nothing
        changes) for invalid indices, used pieces or illegal positions.
        """
        if not 0 <= piece_index < len(self.pieces):
            return MoveResult(success=False)
        piece = self.pieces[piece_index]
        if piece is None or not self.can_place(piece, row, col):
            return MoveResult(success=False)

        for r, c in piece.cells:
            self.board[row + r, col + c] = piece.color
        self.pieces[piece_index] = None

        rows, cols = self.clear_lines()
        lines = rows + cols
        points = (
            piece.size * self.config.points_per_cell
            + lines * self.config.points_per_line
            + self.config.combo_bonus * (lines * (lines - 1) // 2)
        )
        self.score += points
        self.moves += 1
        self.total_lines_cleared += lines

        refreshed = False
        if all(p is None for p in self.pieces):
            self.pieces = self.generate_piece_set()
            refreshed = True

        self._valid_cache = None  # board and pieces changed
        self._playable_types_cache = None

        return MoveResult(
            success=True,
            cells_placed=piece.size,
            rows_cleared=rows,
            cols_cleared=cols,
            points_gained=points,
            pieces_refreshed=refreshed,
            game_over=self.is_game_over(),
        )

    def clear_lines(self) -> Tuple[int, int]:
        """Clear all complete rows and columns simultaneously.

        Detection happens before any clearing so a cell shared by a full row
        and a full column counts for both. Returns ``(rows, cols)`` cleared.
        """
        occupied = self.board != EMPTY
        full_rows = np.where(occupied.all(axis=1))[0]
        full_cols = np.where(occupied.all(axis=0))[0]
        if len(full_rows):
            self.board[full_rows, :] = EMPTY
        if len(full_cols):
            self.board[:, full_cols] = EMPTY
        return int(len(full_rows)), int(len(full_cols))

    def has_any_valid_move(self) -> bool:
        return len(self.get_valid_actions()) > 0

    def is_game_over(self) -> bool:
        """True when none of the remaining pieces can be placed anywhere."""
        return not self.has_any_valid_move()

    # ------------------------------------------------------------------ #
    # convenience
    # ------------------------------------------------------------------ #

    def available_pieces(self) -> List[Tuple[int, Piece]]:
        """``(slot_index, piece)`` pairs for slots not yet used."""
        return [(i, p) for i, p in enumerate(self.pieces) if p is not None]

    def __repr__(self) -> str:
        symbols = "." + "ABCDEFGH"
        rows = [" ".join(symbols[cell] for cell in row) for row in self.board]
        return f"BlockBlastGame(score={self.score})\n" + "\n".join(rows)
