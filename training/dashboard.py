"""Live training dashboard.

Watches the CSV logs produced by a running training session and redraws
the learning curves every ``--interval`` seconds. Runs as a separate
process so it never slows training down; it does not use Pygame.

Usage:
    # terminal 1: train
    python -m training.train --timesteps 5000000 --run-name myrun
    # terminal 2: watch
    python -m training.dashboard --run myrun
    python -m training.dashboard --run myrun --interval 10 --total-steps 5000000

The dashboard always saves the current figure to
``results/dashboard_<run>.png`` (watchable in any image viewer / VS Code)
and also opens an interactive matplotlib window when a display is
available. Headless environments just get the PNG.

Panels: reward, score, episode length, lines cleared, policy/value loss,
entropy, explained variance, learning rate — each with a rolling-mean
overlay. A text box shows timestep, episodes, FPS, best score, elapsed
time and ETA (when --total-steps is given).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

try:
    matplotlib.use("TkAgg")  # interactive window when available
    _INTERACTIVE = True
except Exception:
    matplotlib.use("Agg")
    _INTERACTIVE = False

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    window = max(min(window, len(values)), 1)
    return np.convolve(values, np.ones(window) / window, mode="valid")


def read_csv(path: Path) -> List[Dict[str, str]]:
    try:
        with path.open() as f:
            return list(csv.DictReader(f))
    except (FileNotFoundError, PermissionError):
        return []


def column(rows: List[Dict[str, str]], name: str) -> np.ndarray:
    values = []
    for row in rows:
        try:
            values.append(float(row[name]))
        except (KeyError, ValueError):
            continue
    return np.array(values, dtype=float)


def series(rows: List[Dict[str, str]], x_name: str, y_name: str):
    """Aligned (x, y) arrays using only rows where both fields are present."""
    xs, ys = [], []
    for row in rows:
        try:
            x = float(row[x_name])
            y = float(row[y_name])
        except (KeyError, ValueError):
            continue
        xs.append(x)
        ys.append(y)
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


class Dashboard:
    """Polls training logs and redraws the figure on an interval."""

    def __init__(self, run: str, interval: float = 5.0, window: int = 200,
                 total_steps: Optional[int] = None, out: Optional[Path] = None):
        self.run = run
        self.interval = interval
        self.window = window
        self.total_steps = total_steps
        self.episode_log = RESULTS_DIR / f"training_log_{run}.csv"
        self.metrics_log = RESULTS_DIR / f"metrics_{run}.csv"
        self.out_path = out or RESULTS_DIR / f"dashboard_{run}.png"

        self.fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        self.axes = axes.ravel()
        self.fig.suptitle(f"Block Blast training dashboard — {run}")
        self.fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        self._text = self.fig.text(0.01, 0.005, "", fontsize=9, family="monospace")

    # ------------------------------------------------------------------ #

    def _plot_series(self, ax, steps, values, title, ylabel, logy=False):
        ax.clear()
        if len(values) == 0:
            ax.set_title(f"{title} (waiting for data)")
            return
        ax.plot(steps, values, alpha=0.15, color="gray", label="raw")
        rolled = rolling_mean(values, self.window)
        x = steps[len(steps) - len(rolled):]
        ax.plot(x, rolled, label=f"rolling mean (w={self.window})")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, loc="best")
        if logy and (values > 0).any():
            ax.set_yscale("log")

    def update(self) -> bool:
        """Redraw once. Returns False when training appears finished."""
        episodes = read_csv(self.episode_log)
        metrics = read_csv(self.metrics_log)

        self._plot_series(self.axes[0], *series(episodes, "timesteps", "reward"),
                          "episode reward", "reward")
        self._plot_series(self.axes[1], *series(episodes, "timesteps", "score"),
                          "game score", "score")
        self._plot_series(self.axes[2], *series(episodes, "timesteps", "moves"),
                          "episode length", "moves")
        self._plot_series(self.axes[3], *series(episodes, "timesteps", "lines_cleared"),
                          "lines cleared", "lines")

        self._plot_series(self.axes[4], *series(metrics, "timesteps", "policy_gradient_loss"),
                          "policy loss", "loss")
        ax = self.axes[4]
        m_steps, value_loss = series(metrics, "timesteps", "value_loss")
        if len(value_loss):
            rolled = rolling_mean(value_loss, max(self.window // 10, 1))
            ax.plot(m_steps[len(m_steps) - len(rolled):], rolled,
                    label="value loss (rolling)", color="tab:red")
            ax.legend(fontsize=7, loc="best")
        self._plot_series(self.axes[5], *series(metrics, "timesteps", "entropy_loss"),
                          "entropy", "entropy loss")
        self._plot_series(self.axes[6], *series(metrics, "timesteps", "explained_variance"),
                          "explained variance", "variance")
        self._plot_series(self.axes[7], *series(metrics, "timesteps", "learning_rate"),
                          "learning rate", "lr")

        self._text.set_text(self._stats_text(episodes, metrics))
        self.fig.savefig(self.out_path, dpi=110)
        if _INTERACTIVE:
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
        return True

    def _stats_text(self, episodes, metrics) -> str:
        if not episodes:
            return f"waiting for {self.episode_log} ..."
        latest = episodes[-1]
        scores = column(episodes, "score")
        rewards = column(episodes, "reward")
        invalid = column(episodes, "invalid_termination")
        w = min(len(episodes), self.window)
        steps = int(float(latest["timesteps"]))

        fps = elapsed = 0.0
        approx_kl = entropy = ev = lr = "?"
        if metrics:
            last = metrics[-1]
            fps = float(last.get("fps") or 0)
            elapsed = float(last.get("elapsed_sec") or 0)
            approx_kl = last.get("approx_kl", "?")
            entropy = last.get("entropy_loss", "?")
            ev = last.get("explained_variance", "?")
            lr = last.get("learning_rate", "?")

        total = f" / {self.total_steps:,}" if self.total_steps else ""
        eta = ""
        if self.total_steps and fps > 0:
            eta = f" | ETA {(self.total_steps - steps) / fps / 60:.0f} min"
        elapsed_s = f"{elapsed / 60:.0f} min" if elapsed else "?"

        line1 = (f"steps {steps:,}{total} | episodes {len(episodes):,} | "
                 f"fps {fps:,.0f} | elapsed {elapsed_s}{eta}")
        line2 = (f"reward mean(last {w}) {np.mean(rewards[-w:]):.1f} | "
                 f"score mean {np.mean(scores[-w:]):.1f} best {scores.max():.0f} | "
                 f"invalid-term rate {np.mean(invalid[-w:]):.3f}")
        line3 = f"approx_kl {approx_kl} | entropy {entropy} | expl.var {ev} | lr {lr}"
        return f"{line1}\n{line2}\n{line3}"

    def run(self) -> None:
        print(f"Dashboard watching {self.episode_log} (+{self.metrics_log})")
        print(f"Saving figure to {self.out_path} every {self.interval}s "
              f"({'interactive window' if _INTERACTIVE else 'headless'})")
        while True:
            self.update()
            time.sleep(self.interval)


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Block Blast training dashboard.")
    parser.add_argument("--run", type=str, required=True, help="run name to watch")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="seconds between redraws (default: 5)")
    parser.add_argument("--window", type=int, default=200,
                        help="rolling-mean window in episodes/updates")
    parser.add_argument("--total-steps", type=int, default=None,
                        help="planned total timesteps, enables ETA")
    parser.add_argument("--once", action="store_true",
                        help="draw a single frame and exit")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    dashboard = Dashboard(args.run, interval=args.interval, window=args.window,
                          total_steps=args.total_steps)
    if args.once:
        dashboard.update()
        print(f"Saved {dashboard.out_path}")
    else:
        dashboard.run()


if __name__ == "__main__":
    main()
