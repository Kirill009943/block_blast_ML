"""Observation profiles for the Block Blast RL environment.

``basic`` (4 channels, the permanent default):

* ch 0:   board occupancy (1 = occupied)
* ch 1-3: one binary mask per piece slot, drawn at the piece's natural
  origin; all zeros when the slot is used.

``enhanced`` (8 channels) adds four cheap, spatial, Markovian channels —
everything is a function of (board, piece slots) only:

* ch 4: normalized column heights (height / board_size), broadcast down
  each column.
* ch 5: holes mask — empty cells with an occupied cell somewhere above.
* ch 6: line-completion potential — for each cell, the larger of its row's
  and its column's occupied fraction; high values mark cells that are
  close to completing a line.
* ch 7: legal-placement density — for each cell, how many currently legal
  placements would cover it (normalized to [0, 1]); reuses the engine's
  cached valid-action list, so it costs almost nothing.

``enhanced_piece_legal`` (10 channels) keeps the first seven channels from
``enhanced`` but replaces the merged legal-placement density with one
legal-placement coverage map per piece slot:

* ch 7-9: legal-placement coverage for piece slot 0, 1 and 2 respectively.

Selected with ``--observation-profile``. ``basic`` stays the default until
experiments show the enhanced representation actually helps.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from game.game import BOARD_SIZE, BlockBlastGame
from game.utils import column_heights, count_holes

NUM_PIECE_SLOTS = 3
OBSERVATION_CHANNELS: Dict[str, int] = {
    "basic": 4,
    "enhanced": 8,
    "enhanced_piece_legal": 10,
}
DEFAULT_OBSERVATION_PROFILE = "basic"


def observation_channels(profile: str) -> int:
    try:
        return OBSERVATION_CHANNELS[profile]
    except KeyError:
        known = ", ".join(sorted(OBSERVATION_CHANNELS))
        raise ValueError(f"Unknown observation profile {profile!r}. Known: {known}") from None


def profile_for_channels(channels: int) -> str:
    """Reverse lookup, used by the RL agent to match a loaded model."""
    for name, count in OBSERVATION_CHANNELS.items():
        if count == channels:
            return name
    raise ValueError(f"No observation profile with {channels} channels")


def build_observation(game: BlockBlastGame, profile: str = "basic") -> np.ndarray:
    """Encode an engine state as the observation for ``profile``.

    Shared by the env, the RL agent and the UI so there is exactly one
    implementation of the state encoding.
    """
    channels = observation_channels(profile)
    obs = np.zeros((channels, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    occupied = game.board != 0
    obs[0] = occupied
    for slot in range(NUM_PIECE_SLOTS):
        piece = game.pieces[slot]
        if piece is not None:
            obs[1 + slot] = piece.mask(BOARD_SIZE)

    if profile in ("enhanced", "enhanced_piece_legal"):
        size = BOARD_SIZE
        heights = column_heights(game.board).astype(np.float32) / size
        obs[4] = np.broadcast_to(heights, (size, size))

        # holes: empty cells with an occupied cell somewhere above (vectorized)
        above = np.maximum.accumulate(occupied, axis=0)
        above_exclusive = np.roll(above, 1, axis=0)
        above_exclusive[0] = False
        obs[5] = above_exclusive & ~occupied

        row_fill = occupied.mean(axis=1, dtype=np.float32)[:, None]
        col_fill = occupied.mean(axis=0, dtype=np.float32)[None, :]
        obs[6] = np.maximum(row_fill, col_fill)

        # legal-placement coverage via one bincount over all covering cells
        # (numpy scalar indexing in a Python loop would be ~25x slower).
        # The legacy enhanced profile stores one merged map for checkpoint
        # compatibility. enhanced_piece_legal stores one slot-specific map
        # to preserve the piece-index/action-index relationship explicitly.
        valid = game.get_valid_actions()  # cached
        if profile == "enhanced":
            rows = []
            cols = []
            for piece_index, row, col in valid:
                piece = game.pieces[piece_index]
                if piece is None:
                    continue
                for r, c in piece.cells:
                    rows.append(row + r)
                    cols.append(col + c)
            if rows:
                density = np.bincount(
                    np.asarray(rows) * size + np.asarray(cols),
                    minlength=size * size,
                ).reshape(size, size).astype(np.float32)
                if density.max() > 0:
                    density /= density.max()
                obs[7] = density
        else:
            for slot in range(NUM_PIECE_SLOTS):
                rows = []
                cols = []
                piece = game.pieces[slot]
                if piece is None:
                    continue
                for piece_index, row, col in valid:
                    if piece_index != slot:
                        continue
                    for r, c in piece.cells:
                        rows.append(row + r)
                        cols.append(col + c)
                if rows:
                    coverage = np.bincount(
                        np.asarray(rows) * size + np.asarray(cols),
                        minlength=size * size,
                    ).reshape(size, size).astype(np.float32)
                    if coverage.max() > 0:
                        coverage /= coverage.max()
                    obs[7 + slot] = coverage

    return obs
