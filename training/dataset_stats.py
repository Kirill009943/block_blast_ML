"""Inspect a demonstration dataset: sizes, sources, bias.

    python -m training.dataset_stats
    python -m training.dataset_stats --data results/demos

Reports per-source game/sample counts and average game length, plus the
action distribution — per piece slot and the most frequent placements —
which is the quickest way to spot dataset bias (e.g. an agent that only
ever uses slot 0, or never plays the top rows).
"""

from __future__ import annotations

import argparse
from typing import Optional

import numpy as np

from environment.block_blast_env import decode_action
from game.pieces import PIECE_TYPES, rotate_cells
from training.demos import DemoDataset

DEFAULT_DATASET_DIR = "results/demos"

SCORE_BINS = ((0, 100), (100, 200), (200, 500), (500, None))
OCCUPANCY_BINS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01))


def _shape_lookup() -> dict:
    lookup = {}
    for type_id, (name, cells) in enumerate(PIECE_TYPES):
        for rotation in range(4):
            rotated = rotate_cells(cells, rotation)
            lookup.setdefault(rotated, f"{name}:{type_id}")
    return lookup


def _piece_key(channel: np.ndarray, lookup: dict) -> str:
    rows, cols = np.nonzero(channel)
    if not len(rows):
        return "used"
    min_r, min_c = int(rows.min()), int(cols.min())
    cells = tuple(sorted((int(r) - min_r, int(c) - min_c) for r, c in zip(rows, cols)))
    return lookup.get(cells, f"unknown:{cells}")


def _count_bins(values: np.ndarray, bins) -> list[tuple[str, int]]:
    out = []
    for low, high in bins:
        if high is None:
            keep = values >= low
            label = f"{low}+"
        else:
            keep = (values >= low) & (values < high)
            label = f"{low}-{high}"
        out.append((label, int(keep.sum())))
    return out


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demonstration dataset statistics.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--top-actions", type=int, default=10,
                        help="how many most-frequent actions to list")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    dataset = DemoDataset(args.data)
    stats = dataset.stats()

    print(f"Dataset: {args.data}")
    print(f"Total games:   {stats['total_games']:,}")
    print(f"Total samples: {stats['total_samples']:,}")
    if not stats["sources"]:
        print("(empty dataset)")
        return

    print("\nPer source:")
    print(f"  {'source':<18}{'games':>8}{'samples':>12}{'avg length':>12}")
    for name, entry in sorted(stats["sources"].items()):
        print(f"  {name:<18}{entry['games']:>8,}{entry['samples']:>12,}"
              f"{entry['avg_length']:>12.1f}")

    combined = sum((e["action_hist"] for e in stats["sources"].values()),
                   np.zeros(192, dtype=np.int64))
    total = max(int(combined.sum()), 1)

    data = dataset.load()
    observations = data["observations"]
    actions = data["actions"].astype(np.int64)
    scores = data["score_before"].astype(np.int32)
    board_occupancy = observations[:, 0].reshape(len(observations), -1).mean(axis=1)

    print("\nSamples by score-before range:")
    for label, count in _count_bins(scores, SCORE_BINS):
        print(f"  {label:<12}{count:>10,} ({100 * count / total:5.1f}%)")

    print("\nSamples by board occupancy:")
    for label, count in _count_bins(board_occupancy, OCCUPANCY_BINS):
        print(f"  {label:<12}{count:>10,} ({100 * count / total:5.1f}%)")

    unique_states = len({np.ascontiguousarray(obs).tobytes() for obs in observations})
    print(f"\nUnique observations: {unique_states:,} "
          f"({100 * unique_states / total:.1f}% of samples)")

    print("\nAction distribution by piece slot:")
    for slot in range(3):
        share = combined[slot * 64:(slot + 1) * 64].sum() / total
        print(f"  slot {slot}: {100 * share:5.1f}%")

    lookup = _shape_lookup()
    print("\nPiece-shape distribution by slot (top 8):")
    for slot in range(3):
        counts = {}
        for obs in observations:
            key = _piece_key(obs[1 + slot], lookup)
            counts[key] = counts.get(key, 0) + 1
        top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:8]
        print(f"  slot {slot}:")
        for key, count in top:
            print(f"    {key:<22}{count:>10,} ({100 * count / total:5.1f}%)")

    grid = combined.reshape(3, 8, 8).sum(axis=0)  # placements per board cell
    print("\nPlacement rows (share of all actions):")
    print("  " + " ".join(f"{100 * share / total:4.1f}" for share in grid.sum(axis=1)))
    print("Placement cols (share of all actions):")
    print("  " + " ".join(f"{100 * share / total:4.1f}" for share in grid.sum(axis=0)))

    print(f"\nTop {args.top_actions} actions (piece, row, col):")
    top = combined.argsort()[::-1][: args.top_actions]
    for action in top:
        slot, row, col = decode_action(int(action))
        print(f"  ({slot}, {row}, {col}): {combined[action]:>8,} "
              f"({100 * combined[action] / total:.2f}%)")


if __name__ == "__main__":
    main()
