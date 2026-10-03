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

from training.demos import DemoDataset

DEFAULT_DATASET_DIR = "results/demos"


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

    print("\nAction distribution by piece slot:")
    for slot in range(3):
        share = combined[slot * 64:(slot + 1) * 64].sum() / total
        print(f"  slot {slot}: {100 * share:5.1f}%")

    grid = combined.reshape(3, 8, 8).sum(axis=0)  # placements per board cell
    print("\nPlacement rows (share of all actions):")
    print("  " + " ".join(f"{100 * share / total:4.1f}" for share in grid.sum(axis=1)))
    print("Placement cols (share of all actions):")
    print("  " + " ".join(f"{100 * share / total:4.1f}" for share in grid.sum(axis=0)))

    print(f"\nTop {args.top_actions} actions (piece, row, col):")
    top = combined.argsort()[::-1][: args.top_actions]
    for action in top:
        slot = int(action) // 64
        row, col = (int(action) % 64) // 8, int(action) % 8
        print(f"  ({slot}, {row}, {col}): {combined[action]:>8,} "
              f"({100 * combined[action] / total:.2f}%)")


if __name__ == "__main__":
    main()
