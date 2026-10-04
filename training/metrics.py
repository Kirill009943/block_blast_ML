"""Training metrics callbacks: TensorBoard namespaces + CSV logs.

Two callbacks used by ``training/train.py``:

* :class:`EpisodeMetricsCallback` — per finished episode: appends a row to
  the episode CSV (score, reward, moves, lines, valid actions, termination
  cause, every reward component) and records rolling TensorBoard scalars
  under ``game/*``.
* :class:`TrainMetricsCallback` — after every PPO update: appends the
  optimizer statistics (entropy, approx KL, losses, explained variance,
  learning rate, clip fraction, fps) to a metrics CSV so the dashboard and
  plotting tools can consume them without parsing TensorBoard event files.

TensorBoard namespaces (SB3 already logs ``train/*``, ``rollout/*`` and
``eval/*``; these callbacks add the rest):

* ``game/score`` ``game/reward`` ``game/episode_length`` ``game/lines_cleared``
  ``game/valid_actions`` ``game/game_over_rate`` ``game/invalid_action_rate``
* ``game/reward_<component>`` — one scalar per reward component (mean over
  the rolling window): tells you which term the agent is optimizing.
* ``performance/fps`` ``performance/episodes``
* ``training/...`` aliases for the most-watched series so the exact tag
  names from the project spec exist too.
"""

from __future__ import annotations

import csv
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

from stable_baselines3.common.callbacks import BaseCallback

from environment.block_blast_env import REWARD_COMPONENTS

EPISODE_COLUMNS = (
    "timesteps",
    "score",
    "reward",
    "moves",
    "lines_cleared",
    "valid_actions",
    "final_valid_actions",
    "invalid_termination",
    "holes_mean",
    "regions_mean",
    "occupancy_mean",
    "future_moves_mean",
    "future_moves_delta_mean",
) + tuple(f"reward_{name}" for name in REWARD_COMPONENTS)

BOARD_METRIC_KEYS = (
    "holes_mean",
    "regions_mean",
    "occupancy_mean",
    "future_moves_mean",
    "future_moves_delta_mean",
)

TRAIN_METRIC_KEYS = (
    "train/entropy_loss",
    "train/approx_kl",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/explained_variance",
    "train/learning_rate",
    "train/clip_fraction",
    "train/loss",
    "train/n_updates",
)


class EpisodeMetricsCallback(BaseCallback):
    """Logs per-episode game statistics to CSV and TensorBoard."""

    def __init__(self, csv_path: Path, rolling_window: int = 100, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                csv.writer(f).writerow(EPISODE_COLUMNS)
        self.episodes = 0
        self._window: Deque[Dict[str, float]] = deque(maxlen=rolling_window)
        self._last_print = 0.0
        self._last_share_print = 0.0

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            self.episodes += 1
            components = info.get("episode_reward_components") or {
                name: 0.0 for name in REWARD_COMPONENTS
            }
            # an episode ending with a large negative game_over component
            # after zero moves means an invalid action terminated it
            invalid = bool(info.get("moves", 0) == 0 and info.get("episode_reward", 0) < 0)
            # "valid_actions" = mean legal actions over the episode (from
            # board_metrics). info["valid_actions"] at done is the TERMINAL
            # count — 0 by definition at game over — so it is reported
            # separately as a sanity column, not as the episode metric.
            board = info.get("board_metrics") or {}
            row = {
                "timesteps": self.num_timesteps,
                "score": info.get("score", 0),
                "reward": round(info.get("episode_reward", 0.0), 3),
                "moves": info.get("moves", 0),
                "lines_cleared": info.get("lines_cleared", 0),
                "valid_actions": round(board.get("future_moves", 0.0), 2),
                "final_valid_actions": info.get("valid_actions", 0),
                "invalid_termination": int(invalid),
                "holes_mean": round(board.get("holes", 0.0), 3),
                "regions_mean": round(board.get("regions", 0.0), 3),
                "occupancy_mean": round(board.get("occupancy_fraction", 0.0), 4),
                "future_moves_mean": round(board.get("future_moves", 0.0), 2),
                "future_moves_delta_mean": round(board.get("future_moves_delta", 0.0), 4),
            }
            row.update({f"reward_{k}": round(v, 3) for k, v in components.items()})
            with self.csv_path.open("a", newline="") as f:
                csv.writer(f).writerow([row[c] for c in EPISODE_COLUMNS])
            self._window.append(row)
            self._log_rolling()

            now = time.monotonic()
            if self.episodes % 100 == 0 and now - self._last_print > 1.0:
                self._last_print = now
                mean = self._rolling_mean()
                print(
                    f"  episodes={self.episodes} steps={self.num_timesteps} "
                    f"mean_score={mean['score']:.1f} mean_reward={mean['reward']:.1f}"
                )
            if self.episodes % 500 == 0 and now - self._last_share_print > 1.0:
                self._last_share_print = now
                print(self._component_share_report())
        return True

    def _component_share_report(self) -> str:
        """Which reward terms dominate the rolling window, in raw units.

        Each component's share is its magnitude relative to the summed
        magnitudes of ALL components, so the percentages show what the
        agent's objective is actually made of (e.g. whether one shaped term
        has drowned out line clears). Values are raw game units; the
        optimizer additionally sees them scaled by reward_scale.
        """
        n = max(len(self._window), 1)
        means = {
            name: sum(r[f"reward_{name}"] for r in self._window) / n
            for name in REWARD_COMPONENTS
        }
        total_abs = sum(abs(v) for v in means.values()) or 1.0
        ranked = sorted(means.items(), key=lambda kv: -abs(kv[1]))
        parts = [
            f"{name}={value:+.2f} ({100 * abs(value) / total_abs:.0f}%)"
            for name, value in ranked
            if abs(value) / total_abs >= 0.005  # skip noise terms
        ]
        return f"  reward mix (last {len(self._window)} episodes): " + " ".join(parts)

    def _rolling_mean(self) -> Dict[str, float]:
        keys = ("score", "reward", "moves", "lines_cleared", "valid_actions",
                "invalid_termination") + BOARD_METRIC_KEYS
        n = max(len(self._window), 1)
        return {k: sum(r[k] for r in self._window) / n for k in keys}

    def _log_rolling(self) -> None:
        mean = self._rolling_mean()
        n = max(len(self._window), 1)
        component_means = {
            name: sum(r[f"reward_{name}"] for r in self._window) / n
            for name in REWARD_COMPONENTS
        }
        game_over_rate = 1.0 - mean["invalid_termination"]
        scalars = {
            "game/score": mean["score"],
            "game/reward": mean["reward"],
            "game/episode_length": mean["moves"],
            "game/lines_cleared": mean["lines_cleared"],
            "game/valid_actions": mean["valid_actions"],
            "game/game_over_rate": game_over_rate,
            "game/invalid_action_rate": mean["invalid_termination"],
            # board-quality metrics: WHY is the agent improving/failing?
            "game/holes": mean["holes_mean"],
            "game/empty_regions": mean["regions_mean"],
            "game/occupancy_fraction": mean["occupancy_mean"],
            "game/future_legal_moves": mean["future_moves_mean"],
            "game/future_moves_delta": mean["future_moves_delta_mean"],
            # spec-compatible aliases
            "training/score": mean["score"],
            "training/reward": mean["reward"],
            "training/episode_length": mean["moves"],
            "training/lines_cleared": mean["lines_cleared"],
            "training/game_over_rate": game_over_rate,
            "training/valid_actions": mean["valid_actions"],
            "performance/episodes": self.episodes,
        }
        for name, value in component_means.items():
            scalars[f"game/reward_{name}"] = value
        for key, value in scalars.items():
            self.logger.record(key, value)


class TrainMetricsCallback(BaseCallback):
    """Dumps PPO optimizer statistics to a CSV after every update."""

    COLUMNS = ("timesteps", "elapsed_sec", "fps") + tuple(
        key.split("/", 1)[1] for key in TRAIN_METRIC_KEYS
    )

    def __init__(self, csv_path: Path):
        super().__init__()
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                csv.writer(f).writerow(self.COLUMNS)
        self._start = time.monotonic()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # logger values are refreshed by model.learn() right after this hook
        values = self.model.logger.name_to_value
        elapsed = time.monotonic() - self._start
        fps = self.num_timesteps / max(elapsed, 1e-9)
        row: List[object] = [self.num_timesteps, round(elapsed, 1), round(fps, 1)]
        for key in TRAIN_METRIC_KEYS:
            value = values.get(key, "")
            row.append(round(value, 6) if isinstance(value, float) else value)
        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow(row)
        self.logger.record("performance/fps", fps)


def format_hms(seconds: float) -> str:
    """Format a duration as ``h:mm:ss`` (days folded into hours)."""
    seconds = max(int(seconds), 0)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class ProgressTimerCallback(BaseCallback):
    """Prints progress, elapsed time, steps/sec and ETA periodically.

    The rate is measured over the CURRENT session (correct for resumed
    runs); the percentage/ETA use the model's total timestep count, which
    SB3 sets to include previously trained steps when resuming.
    """

    def __init__(self, interval_seconds: float = 30.0):
        super().__init__()
        self.interval = interval_seconds
        self._start_time = 0.0
        self._start_steps = 0
        self._last_print = 0.0

    def _on_training_start(self) -> None:
        self._start_time = time.monotonic()
        self._start_steps = self.model.num_timesteps
        self._last_print = 0.0

    def _on_step(self) -> bool:
        now = time.monotonic()
        if now - self._last_print < self.interval:
            return True
        self._last_print = now
        done = self.model.num_timesteps
        total = max(self.model._total_timesteps, 1)
        elapsed = now - self._start_time
        rate = (done - self._start_steps) / max(elapsed, 1e-9)
        remaining = (total - done) / max(rate, 1e-9)
        print(f"[timer] {done:,}/{total:,} steps ({100 * done / total:.1f}%) | "
              f"elapsed {format_hms(elapsed)} | ETA {format_hms(remaining)} | "
              f"{rate:,.0f} steps/s", flush=True)
        return True


def read_episode_log(csv_path: Path) -> List[Dict[str, str]]:
    with Path(csv_path).open() as f:
        return list(csv.DictReader(f))
def read_metrics_log(csv_path: Path) -> List[Dict[str, str]]:
    with Path(csv_path).open() as f:
        return list(csv.DictReader(f))
