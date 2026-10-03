"""Spatial-symmetry data augmentation for behavioral cloning.

Block Blast is symmetric under board reflections, so every recorded sample
(observation, action, mask) yields up to 4 equivalent training samples:
identity, horizontal flip, vertical flip and 180-degree rotation.

A transform is only valid when it consistently maps ALL THREE of:

* the board channel (flipped directly),
* the piece channels (flipped, then renormalized to the origin — piece
  masks are always drawn with their bounding box at (0, 0)),
* the action and the 192-bit action mask (placement coordinates move with
  the mirror; a placement at ``col`` of a ``w``-wide piece maps to
  ``8 - col - w`` under a horizontal flip).

SAFETY: only the ``basic`` (4-channel) observation profile is supported.
The enhanced profile's channel 4 (column heights) and channel 5 (holes)
are NOT mirror-symmetric (heights are measured from the top edge), so
flipping those arrays would produce wrong states. Passing an 8-channel
observation raises ``ValueError``.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from game.game import BOARD_SIZE

TRANSFORMS = ("id", "hflip", "vflip", "rot180")

_SIZE = BOARD_SIZE


def _flip_grid(grid: np.ndarray, transform: str) -> np.ndarray:
    """Mirror an (8, 8) grid."""
    if transform in ("hflip", "rot180"):
        grid = grid[:, ::-1]
    if transform in ("vflip", "rot180"):
        grid = grid[::-1, :]
    return grid


def _renormalize(channel: np.ndarray) -> np.ndarray:
    """Shift a piece channel so its bounding box starts at (0, 0)."""
    rows, cols = np.nonzero(channel)
    if not len(rows):
        return channel
    out = np.zeros_like(channel)
    out[: rows.max() - rows.min() + 1, : cols.max() - cols.min() + 1] = channel[
        rows.min() : rows.max() + 1, cols.min() : cols.max() + 1
    ]
    return out


def piece_dimensions(obs: np.ndarray) -> Tuple[Tuple[int, int], ...]:
    """``(height, width)`` of the piece in each slot, from the observation.

    Empty (used) slots yield ``(1, 1)``; their mask rows are all-False and
    no action references them, so the value never matters.
    """
    dims = []
    for slot in range(3):
        channel = obs[1 + slot]
        rows, cols = np.nonzero(channel)
        if not len(rows):
            dims.append((1, 1))
        else:
            dims.append((int(rows.max()) + 1, int(cols.max()) + 1))
    return tuple(dims)


def transform_observation(obs: np.ndarray, transform: str) -> np.ndarray:
    """Transform a ``basic``-profile (4, 8, 8) observation."""
    if transform not in TRANSFORMS:
        raise ValueError(f"Unknown transform {transform!r}. Known: {TRANSFORMS}")
    if obs.shape[0] != 4:
        raise ValueError(
            "Augmentation only supports the basic (4-channel) observation "
            "profile: enhanced channels 4-5 are not mirror-symmetric."
        )
    if transform == "id":
        return obs.copy()
    out = np.empty_like(obs)
    out[0] = _flip_grid(obs[0], transform)
    for slot in range(3):
        out[1 + slot] = _renormalize(_flip_grid(obs[1 + slot], transform))
    return out


def transform_mask(mask: np.ndarray, dims: Tuple[Tuple[int, int], ...], transform: str) -> np.ndarray:
    """Transform a (192,) action mask given per-slot piece dimensions.

    Per slot the 8x8 placement grid only has meaningful entries in the
    top-left ``(9 - h) x (9 - w)`` region; mirroring the placement grid
    means reversing that region, not the whole 8x8 grid.
    """
    grids = mask.reshape(3, _SIZE, _SIZE)
    out = np.zeros_like(grids)
    for slot in range(3):
        height, width = dims[slot]
        region = grids[slot][: _SIZE - height + 1, : _SIZE - width + 1]
        out[slot][: _SIZE - height + 1, : _SIZE - width + 1] = _flip_grid(region, transform)
    return out.reshape(-1)


def transform_action(
    action: int, dims: Tuple[Tuple[int, int], ...], transform: str
) -> int:
    """Map a discrete action id through the symmetry."""
    slot = action // (_SIZE * _SIZE)
    cell = action % (_SIZE * _SIZE)
    row, col = cell // _SIZE, cell % _SIZE
    height, width = dims[slot]
    if transform in ("hflip", "rot180"):
        col = _SIZE - col - width
    if transform in ("vflip", "rot180"):
        row = _SIZE - row - height
    return slot * _SIZE * _SIZE + row * _SIZE + col


def transform_sample(
    obs: np.ndarray, action: int, mask: np.ndarray, transform: str
) -> Tuple[np.ndarray, int, np.ndarray]:
    """Transform one (observation, action, mask) sample consistently."""
    dims = piece_dimensions(obs)
    return (
        transform_observation(obs, transform),
        transform_action(action, dims, transform),
        transform_mask(mask, dims, transform),
    )


def augment_dataset(
    data: Dict[str, np.ndarray], transforms: Tuple[str, ...] = TRANSFORMS
) -> Dict[str, np.ndarray]:
    """Stack transformed copies of every sample along the sample axis.

    Per-sample fields that do not depend on geometry (rewards, sources,
    game ids, ...) are tiled unchanged; geometry fields are transformed.
    """
    observations = data["observations"]
    masks = data["action_masks"]
    actions = data["actions"]
    obs_parts, mask_parts, action_parts = [], [], []
    for transform in transforms:
        if transform == "id":
            obs_parts.append(observations)
            mask_parts.append(masks)
            action_parts.append(actions)
            continue
        new_obs = np.empty_like(observations)
        new_masks = np.empty_like(masks)
        new_actions = np.empty_like(actions)
        for i in range(len(actions)):
            new_obs[i], new_actions[i], new_masks[i] = transform_sample(
                observations[i], int(actions[i]), masks[i], transform
            )
        obs_parts.append(new_obs)
        mask_parts.append(new_masks)
        action_parts.append(new_actions)
    out = dict(data)
    out["observations"] = np.concatenate(obs_parts)
    out["action_masks"] = np.concatenate(mask_parts)
    out["actions"] = np.concatenate(action_parts)
    for key in (
        "rewards", "score_before", "lines_cleared", "game_over",
        "sources", "game_ids", "steps", "suggested_actions",
    ):
        if key in data:
            out[key] = np.tile(data[key], len(transforms))
    return out
