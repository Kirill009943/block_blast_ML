"""Piece shape definitions for Block Blast.

Shapes are taken from the original console implementation and kept
unchanged. Each shape is a tuple of relative ``(row, col)`` cells with the
top-left cell of the bounding box at ``(0, 0)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

Cell = Tuple[int, int]

# (name, cells) — cells are relative (row, col) offsets.
PIECE_TYPES: Tuple[Tuple[str, Tuple[Cell, ...]], ...] = (
    # ─────────────────────────────────────────────
    # 1–2 cells
    # ─────────────────────────────────────────────
    ("one", ((0, 0),)),                              # single cell

    ("domino_v", ((0, 0), (1, 0))),                 # 2 vertical
    ("domino_h", ((0, 0), (0, 1))),                 # 2 horizontal

    # ─────────────────────────────────────────────
    # 3 cells
    # ─────────────────────────────────────────────
    ("line3_v", ((0, 0), (1, 0), (2, 0))),          # 3 vertical
    ("line3_h", ((0, 0), (0, 1), (0, 2))),          # 3 horizontal

    ("corner_l", ((0, 0), (1, 0), (0, 1))),         # L-tromino
    ("corner_j", ((0, 0), (1, 0), (1, 1))),         # J-tromino

    # ─────────────────────────────────────────────
    # 4 cells
    # ─────────────────────────────────────────────
    ("line4_v", ((0, 0), (1, 0), (2, 0), (3, 0))), # 4 vertical
    ("line4_h", ((0, 0), (0, 1), (0, 2), (0, 3))), # 4 horizontal

    ("square2", (
        (0, 0), (1, 0),
        (0, 1), (1, 1),
    )),                                             # 2x2 square

    ("tee", (
        (0, 0),
        (1, 0), (2, 0),
        (1, 1),
    )),                                             # T

    ("zigzag", (
        (0, 0), (1, 0),
        (2, 0), (2, 1),
    )),                                             # L-like zigzag

    ("zigzag2", (
        (0, 0), (0, 1),
        (1, 1), (2, 1),
    )),                                             # mirrored zigzag

    ("ess", (
        (0, 0), (1, 0),
        (1, 1), (2, 1),
    )),                                             # S

    ("wide_l", (
        (0, 0), (1, 0),
        (1, 1), (0, 2),
    )),                                             # wide L

    # ─────────────────────────────────────────────
    # 5 cells
    # ─────────────────────────────────────────────
    ("line5_v", (
        (0, 0), (1, 0), (2, 0),
        (3, 0), (4, 0),
    )),                                             # 5 vertical

    ("line5_h", (
        (0, 0), (0, 1), (0, 2),
        (0, 3), (0, 4),
    )),                                             # 5 horizontal

    ("plus", (
        (0, 1),
        (1, 0), (1, 1), (1, 2),
        (2, 1),
    )),                                             # plus / cross

    ("big_t", (
        (0, 0), (0, 1), (0, 2),
        (1, 1),
        (2, 1),
    )),                                             # large T

    ("big_l", (
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0), (3, 1),
    )),                                             # large L

    ("big_j", (
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 0), (3, 1),
    )),                                             # large J

    ("penta_l", (
        (0, 0),
        (1, 0),
        (2, 0),
        (2, 1),
        (2, 2),
    )),                                             # pentomino L

    # ─────────────────────────────────────────────
    # 6 cells
    # ─────────────────────────────────────────────
    ("line6_v", (
        (0, 0), (1, 0), (2, 0),
        (3, 0), (4, 0), (5, 0),
    )),                                             # 6 vertical

    ("line6_h", (
        (0, 0), (0, 1), (0, 2),
        (0, 3), (0, 4), (0, 5),
    )),                                             # 6 horizontal

    ("rect2x3", (
        (0, 0), (1, 0),
        (0, 1), (1, 1),
        (0, 2), (1, 2),
    )),                                             # 2x3 rectangle

    ("rect3x2", (
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
    )),                                             # 3x2 rectangle

    ("big_plus", (
        (0, 1),
        (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 1),
    )),                                             # wider plus

    # ─────────────────────────────────────────────
    # 7–9 cells
    # ─────────────────────────────────────────────
    ("square3", (
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2),
    )),                                             # 3x3 square

    ("rect3x3_hole", (
        (0, 0), (0, 1), (0, 2),
        (1, 0),         (1, 2),
        (2, 0), (2, 1), (2, 2),
    )),                                             # 3x3 ring

    ("u_shape", (
        (0, 0),         (0, 2),
        (1, 0),         (1, 2),
        (2, 0), (2, 1), (2, 2),
    )),                                             # U

    ("u_shape_up", (
        (0, 0), (0, 1), (0, 2),
        (1, 0),         (1, 2),
        (2, 0),         (2, 1), (2, 2),
    )),                                             # inverted U

    ("big_square_l", (
        (0, 0), (0, 1),
        (1, 0),
        (2, 0),
        (2, 1),
    )),                                             # larger L

    # ─────────────────────────────────────────────
    # 8–9 cells
    # ─────────────────────────────────────────────
    ("rect3x3_left", (
        (0, 0), (0, 1),
        (1, 0), (1, 1),
        (2, 0), (2, 1),
        (3, 0), (3, 1),
    )),                                             # 4x2 rectangle

    ("rect3x3_wide", (
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2),
    )),                                             # 3x3 full block
)

NUM_COLORS = 8  # board cell values 1..8 are color ids; 0 means empty


def rotate_cells(cells: Tuple[Cell, ...], rotation: int) -> Tuple[Cell, ...]:
    """Return ``cells`` rotated in 90-degree steps, normalized to the origin.

    The result is sorted and shifted so the bounding box starts at (0, 0),
    making it directly usable for placement checks. Symmetric shapes can
    produce identical results for different rotations.
    """
    rotated = tuple(cells)
    for _ in range(rotation % 4):
        rotated = tuple((-c, r) for r, c in rotated)
    min_r = min(r for r, _ in rotated)
    min_c = min(c for _, c in rotated)
    return tuple(sorted((r - min_r, c - min_c) for r, c in rotated))


@dataclass(frozen=True)
class Piece:
    """An immutable placeable piece.

    Attributes:
        type_id: index into :data:`PIECE_TYPES`.
        name: human-readable shape name.
        cells: relative ``(row, col)`` cells of the shape.
        color: color id in ``1..NUM_COLORS`` written onto the board.
    """

    type_id: int
    name: str
    cells: Tuple[Cell, ...]
    color: int

    @property
    def size(self) -> int:
        """Number of cells in the piece."""
        return len(self.cells)

    @property
    def height(self) -> int:
        return max(r for r, _ in self.cells) + 1

    @property
    def width(self) -> int:
        return max(c for _, c in self.cells) + 1

    def mask(self, board_size: int = 8) -> np.ndarray:
        """Binary ``board_size x board_size`` mask of the piece at origin (0, 0).

        Used as an observation channel for the RL agent: the piece is drawn
        in the top-left corner of an otherwise empty grid. All zeros would be
        ambiguous, so callers use a zero channel for "slot already used".
        """
        grid = np.zeros((board_size, board_size), dtype=np.float32)
        for r, c in self.cells:
            grid[r, c] = 1.0
        return grid
