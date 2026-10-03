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
            board = info.get("board_metrics") or {}
            row = {
                "timesteps": self.num_timesteps,
                "score": info.get("score", 0),
                "reward": round(info.get("episode_reward", 0.0), 3),
                "moves": info.get("moves", 0),
                "lines_cleared": info.get("lines_cleared", 0),
                "valid_actions": info.get("valid_actions", 0),
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
        return True

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


def read_episode_log(csv_path: Path) -> List[Dict[str, str]]:
    with Path(csv_path).open() as f:
        return list(csv.DictReader(f))


def read_metrics_log(csv_path: Path) -> List[Dict[str, str]]:
    with Path(csv_path).open() as f:
        return list(csv.DictReader(f))
