"""Board analysis helpers shared by the heuristic agent and reward shaping.

All functions take a raw board (numpy int8, 0 = empty) or an occupancy
mask, never a game object, so they stay free of game-logic duplication.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from game.pieces import Piece


def occupancy(board: np.ndarray) -> np.ndarray:
    """Boolean mask: True where a cell is occupied."""
    return board != 0


def column_heights(board: np.ndarray) -> np.ndarray:
    """Height of the occupied stack in each column (0 for empty columns)."""
    occ = occupancy(board)
    size = board.shape[0]
    heights = np.zeros(board.shape[1], dtype=np.int32)
    for c in range(board.shape[1]):
        rows = np.where(occ[:, c])[0]
        heights[c] = size - rows[0] if len(rows) else 0
    return heights


def count_holes(board: np.ndarray) -> int:
    """Empty cells that have at least one occupied cell above them."""
    occ = occupancy(board)
    holes = 0
    for c in range(board.shape[1]):
        seen_block = False
        for r in range(board.shape[0]):
            if occ[r, c]:
                seen_block = True
            elif seen_block:
                holes += 1
    return holes


def count_empty_regions(board: np.ndarray) -> int:
    """Number of connected empty regions (4-connectivity flood fill).

    Higher values mean a more fragmented board, which is harder to fill.
    """
    empty = board == 0
    visited = np.zeros_like(empty, dtype=bool)
    regions = 0
    size = board.shape[0]
    for start_r in range(size):
        for start_c in range(size):
            if not empty[start_r, start_c] or visited[start_r, start_c]:
                continue
            regions += 1
            stack = [(start_r, start_c)]
            visited[start_r, start_c] = True
            while stack:
                r, c = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if (
                        0 <= rr < size
                        and 0 <= cc < size
                        and empty[rr, cc]
                        and not visited[rr, cc]
                    ):
                        visited[rr, cc] = True
                        stack.append((rr, cc))
    return regions


def bumpiness(board: np.ndarray) -> int:
    """Sum of absolute differences between adjacent column heights."""
    heights = column_heights(board)
    return int(np.abs(np.diff(heights)).sum())


def count_valid_placements(board: np.ndarray, piece: Piece) -> int:
    """How many top-left positions ``piece`` can legally occupy on ``board``."""
    occ = occupancy(board)
    size = board.shape[0]
    count = 0
    for row in range(0, size - piece.height + 1):
        for col in range(0, size - piece.width + 1):
            if not any(occ[row + r, col + c] for r, c in piece.cells):
                count += 1
    return count


def simulate_placement(
    board: np.ndarray, piece: Piece, row: int, col: int
) -> Tuple[np.ndarray, int]:
    """Return ``(new_board, lines_cleared)`` after placing ``piece``.

    Assumes the placement is valid; the input board is not modified.
    """
    new_board = board.copy()
    for r, c in piece.cells:
        new_board[row + r, col + c] = piece.color
    occ = new_board != 0
    full_rows = np.where(occ.all(axis=1))[0]
    full_cols = np.where(occ.all(axis=0))[0]
    if len(full_rows):
        new_board[full_rows, :] = 0
    if len(full_cols):
        new_board[:, full_cols] = 0
    return new_board, int(len(full_rows) + len(full_cols))


def placement_mask(occupied: np.ndarray, piece: Piece) -> np.ndarray:
    """Boolean grid of valid top-left positions for ``piece`` (vectorized).

    ``occupied`` is a boolean board (True = cell taken). The result has
    shape ``(size - piece.height + 1, size - piece.width + 1)``; entry
    ``(row, col)`` is True when the piece fits at that top-left position.
    Roughly 10-50x faster than checking positions in a Python loop with
    numpy scalar indexing.
    """
    size_h = occupied.shape[0] - piece.height + 1
    size_w = occupied.shape[1] - piece.width + 1
    if size_h <= 0 or size_w <= 0:
        return np.zeros((max(size_h, 0), max(size_w, 0)), dtype=bool)
    out = np.ones((size_h, size_w), dtype=bool)
    for r, c in piece.cells:
        out &= ~occupied[r:r + size_h, c:c + size_w]
    return out
