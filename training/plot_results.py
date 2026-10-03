"""Plot training progress from a training log CSV.

Generates the two required graphs:
    * training steps vs average episode reward
    * training steps vs average game score

Usage:
    python -m training.plot_results --log results/training_log_run.csv
    python -m training.plot_results --log results/training_log_run.csv --window 200
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: only write image files
import matplotlib.pyplot as plt
import numpy as np


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        window = max(len(values), 1)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def load_log(csv_path: Path):
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    steps = np.array([int(r["timesteps"]) for r in rows])
    scores = np.array([float(r["score"]) for r in rows])
    rewards = np.array([float(r["reward"]) for r in rows])
    return steps, scores, rewards


def plot(csv_path: Path, window: int, out_path: Path) -> None:
    steps, scores, rewards = load_log(csv_path)
    if len(steps) == 0:
        raise SystemExit(f"No episodes found in {csv_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    def draw(ax, values, title, ylabel):
        rolled = rolling_mean(values, window)
        x = steps[len(steps) - len(rolled):]
        ax.plot(x, rolled, label=f"rolling mean (window={window})")
        ax.plot(steps, values, alpha=0.15, color="gray", label="raw")
        ax.set_xlabel("training steps")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    draw(ax1, rewards, "Training progress: reward", "episode reward")
    draw(ax2, scores, "Training progress: game score", "game score")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Block Blast training progress.")
    parser.add_argument("--log", type=str, required=True, help="training_log CSV path")
    parser.add_argument("--window", type=int, default=100, help="rolling mean window")
    parser.add_argument("--out", type=str, default=None,
                        help="output image path (default: <log>.png)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.log)
    out_path = Path(args.out) if args.out else csv_path.with_suffix(".png")
    plot(csv_path, args.window, out_path)


if __name__ == "__main__":
    main()
