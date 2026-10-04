"""Demonstration dataset: recording, storage and loading for imitation learning.

Storage layout: a directory of shard files ``shard_000000.npz``,
``shard_000001.npz``, ... plus a small ``meta.json`` with the next free
shard index and game id (rebuilt by scanning the shards when missing).
Each shard holds one or more COMPLETE games. Per-sample arrays (N = number
of recorded steps in the shard):

    observations      basic: (N, C*8) uint8 bit-packed state BEFORE action
                      enhanced profiles: (N, C, 8, 8) float32
    actions           (N,) uint8       discrete action id 0..191
    action_masks      (N, 24) uint8    bit-packed legal-action mask
    rewards           (N,) float32     game points gained by the move
    score_before      (N,) int32       score before the move
    lines_cleared     (N,) uint8       lines cleared by the move
    game_over         (N,) bool        move ended the game
    sources           (N,) uint8       code from :data:`SOURCE_TO_CODE`
    game_ids          (N,) int32       unique per game, increasing
    steps             (N,) int32       move index within the game
    suggested_actions (N,) int16       AI suggestion (feedback mode), -1 = none

Basic-profile observations and masks are bit-packed with
:func:`numpy.packbits` (8x smaller than bool arrays; unpacked on load).
Enhanced observations contain continuous channels, so they are stored as
float32 arrays. Games are appended as new shards, so datasets grow without
rewriting existing data and the recorder never holds more than
``flush_games`` complete games in RAM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from environment.block_blast_env import build_action_mask, encode_action
from environment.observations import build_observation, observation_channels
from game.game import BlockBlastGame, MoveResult

SOURCES = (
    "human", "heuristic", "ppo", "solver", "random", "human_accepted_ai", "imitation",
)
SOURCE_TO_CODE: Dict[str, int] = {name: i for i, name in enumerate(SOURCES)}
CODE_TO_SOURCE: Dict[int, str] = {i: name for name, i in SOURCE_TO_CODE.items()}

META_FILE = "meta.json"


def _pack(bits: np.ndarray) -> np.ndarray:
    """Pack a (N, K) boolean array into (N, ceil(K/8)) uint8."""
    return np.packbits(np.asarray(bits, dtype=bool), axis=1)


def _unpack(packed: np.ndarray, count: int) -> np.ndarray:
    """Inverse of :func:`_pack`; trims padding bits."""
    return np.unpackbits(packed, axis=1, count=count).astype(bool)


def pack_observation(obs: np.ndarray) -> np.ndarray:
    """(C, 8, 8) observation -> (C*8,) packed row."""
    return _pack(obs.reshape(1, -1))[0]


def unpack_observations(packed: np.ndarray, channels: int) -> np.ndarray:
    """(N, C*8) packed observations -> (N, C, 8, 8) bool."""
    flat = _unpack(packed, channels * 64)
    return flat.reshape(-1, channels, 8, 8)


def scan_directory(directory: Path) -> Tuple[int, int]:
    """Return ``(next_shard_index, next_game_id)`` for a dataset directory."""
    meta_path = directory / META_FILE
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return int(meta["next_shard"]), int(meta["next_game_id"])
    shard_files = sorted(directory.glob("shard_*.npz"))
    next_shard = len(shard_files)
    next_game_id = 0
    for path in shard_files:
        with np.load(path) as data:
            if "game_ids" in data and len(data["game_ids"]):
                next_game_id = max(next_game_id, int(data["game_ids"].max()) + 1)
    return next_shard, next_game_id


class DemoRecorder:
    """Records gameplay into a demonstration dataset, one game at a time.

    Usage pattern (structurally guarantees the state is captured BEFORE the
    action mutates it)::

        recorder.begin_step(game, action)          # captures obs + mask
        result = game.place_piece(*action)         # state changes
        recorder.end_step(result)                  # stores outcome
        ...
        recorder.finish_game()                     # assigns the game id
        recorder.close()                           # final flush

    ``flush_games`` completed games are buffered in RAM, then written as
    one shard. ``close`` also finalizes a partially played game (recorded
    samples are still useful); only ``finish_game`` marks a game complete.
    """

    def __init__(
        self,
        directory: Path | str,
        source: str,
        observation_profile: str = "basic",
        flush_games: int = 25,
    ):
        if source not in SOURCE_TO_CODE:
            raise ValueError(f"Unknown source {source!r}. Known: {sorted(SOURCE_TO_CODE)}")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.source_code = SOURCE_TO_CODE[source]
        self.observation_profile = observation_profile
        self.channels = observation_channels(observation_profile)
        self.observation_storage = (
            "packed_bool" if observation_profile == "basic" else "float32"
        )
        self.flush_games = flush_games

        self._next_shard, self._next_game_id = scan_directory(self.directory)
        self._game: List[dict] = []            # current (unfinished) game
        self._shard_samples: List[dict] = []   # finished games awaiting flush
        self._shard_games = 0
        self._pending: Optional[dict] = None
        self.total_games = 0
        self.total_samples = 0

    # ------------------------------------------------------------------ #
    # recording
    # ------------------------------------------------------------------ #

    @property
    def pending_samples(self) -> int:
        """Samples recorded in the current (not yet finished) game."""
        return len(self._game)

    def begin_step(
        self,
        game: BlockBlastGame,
        action: Tuple[int, int, int],
        suggested_action: int = -1,
        source: Optional[str] = None,
    ) -> None:
        """Capture the state BEFORE ``action`` is executed.

        Raises ``ValueError`` if the action is not currently legal — only
        valid actions may be recorded. ``source`` overrides the recorder's
        default label for this single sample (feedback mode mixes ``human``
        and ``human_accepted_ai`` samples in one game).
        """
        if source is not None and source not in SOURCE_TO_CODE:
            raise ValueError(f"Unknown source {source!r}. Known: {sorted(SOURCE_TO_CODE)}")
        obs = build_observation(game, self.observation_profile)
        mask = build_action_mask(game)
        action_id = encode_action(*action)
        if not mask[action_id]:
            raise ValueError(f"Refusing to record invalid action {action}")
        stored_obs = (
            pack_observation(obs)
            if self.observation_storage == "packed_bool"
            else obs.astype(np.float32)
        )
        self._pending = {
            "observation": stored_obs,
            "action": action_id,
            "action_mask": _pack(mask[None, :])[0],
            "score_before": int(game.score),
            "step": int(game.moves),
            "suggested_action": int(suggested_action),
            "source": SOURCE_TO_CODE[source] if source is not None else None,
        }

    def end_step(self, result: MoveResult) -> None:
        """Store the outcome of the move started by :meth:`begin_step`."""
        if self._pending is None:
            raise RuntimeError("end_step called without begin_step")
        sample = self._pending
        self._pending = None
        sample.update(
            reward=float(result.points_gained),
            lines_cleared=int(result.lines_cleared),
            game_over=bool(result.game_over),
        )
        self._game.append(sample)

    def finish_game(self) -> int:
        """Close the current game and queue it for the next flush."""
        game_id = self._finalize_game()
        if self._shard_games >= self.flush_games:
            self.flush()
        return game_id

    def _finalize_game(self) -> int:
        game_id = self._next_game_id
        self._next_game_id += 1
        for sample in self._game:
            sample["game_id"] = game_id
            if sample["source"] is None:
                sample["source"] = self.source_code
        self._shard_samples.extend(self._game)
        if self._game:
            self._shard_games += 1
            self.total_games += 1
            self.total_samples += len(self._game)
        self._game = []
        self._pending = None
        return game_id

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #

    def flush(self) -> Optional[Path]:
        """Write buffered games as the next shard; returns its path."""
        if not self._shard_samples:
            return None
        samples, self._shard_samples = self._shard_samples, []
        self._shard_games = 0
        path = self.directory / f"shard_{self._next_shard:06d}.npz"
        self._next_shard += 1
        np.savez(
            path,
            observations=np.stack([s["observation"] for s in samples]),
            actions=np.asarray([s["action"] for s in samples], dtype=np.uint8),
            action_masks=np.stack([s["action_mask"] for s in samples]),
            rewards=np.asarray([s["reward"] for s in samples], dtype=np.float32),
            score_before=np.asarray([s["score_before"] for s in samples], dtype=np.int32),
            lines_cleared=np.asarray([s["lines_cleared"] for s in samples], dtype=np.uint8),
            game_over=np.asarray([s["game_over"] for s in samples], dtype=bool),
            sources=np.asarray([s["source"] for s in samples], dtype=np.uint8),
            game_ids=np.asarray([s["game_id"] for s in samples], dtype=np.int32),
            steps=np.asarray([s["step"] for s in samples], dtype=np.int32),
            suggested_actions=np.asarray([s["suggested_action"] for s in samples], dtype=np.int16),
            observation_channels=np.asarray(self.channels, dtype=np.int32),
            observation_storage=np.asarray(self.observation_storage),
        )
        (self.directory / META_FILE).write_text(json.dumps({
            "next_shard": self._next_shard,
            "next_game_id": self._next_game_id,
        }))
        return path

    def close(self) -> None:
        """Finalize the current (possibly partial) game and flush."""
        if self._game:
            self._finalize_game()
        self.flush()

    def __enter__(self) -> "DemoRecorder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class DemoDataset:
    """Read access to a directory of demonstration shards."""

    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        self.shard_files = sorted(self.directory.glob("shard_*.npz"))
        if not self.shard_files:
            raise FileNotFoundError(f"No demonstration shards in {self.directory}")

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #

    def load(self, sources: Optional[Iterable[str]] = None) -> Dict[str, np.ndarray]:
        """Concatenate all shards (optionally filtered by source name).

        Returns a dict of per-sample arrays with observations unpacked to
        ``(N, C, 8, 8)`` bool and masks to ``(N, 192)`` bool. Loading is
        deterministic: shards are processed in filename order.
        """
        wanted = None
        if sources is not None:
            wanted = {SOURCE_TO_CODE[s] for s in sources}
        parts: Dict[str, List[np.ndarray]] = {}
        channels = 0
        for path in self.shard_files:
            with np.load(path) as data:
                channels = int(data["observation_channels"])
                storage = (
                    str(data["observation_storage"].item())
                    if "observation_storage" in data
                    else "packed_bool"
                )
                keep = (
                    np.isin(data["sources"], list(wanted))
                    if wanted is not None
                    else np.ones(len(data["sources"]), dtype=bool)
                )
                if not keep.any():
                    continue
                if storage == "packed_bool":
                    observations = unpack_observations(
                        data["observations"][keep], channels
                    )
                elif storage == "float32":
                    observations = data["observations"][keep].astype(np.float32)
                else:
                    raise ValueError(
                        f"Unknown observation storage {storage!r} in {path}"
                    )
                unpacked = {
                    "observations": observations,
                    "action_masks": _unpack(data["action_masks"][keep], 192),
                }
                for key in (
                    "actions", "rewards", "score_before", "lines_cleared",
                    "game_over", "sources", "game_ids", "steps", "suggested_actions",
                ):
                    unpacked[key] = data[key][keep]
                for key, value in unpacked.items():
                    parts.setdefault(key, []).append(value)
        if not parts:
            raise ValueError(f"No samples for sources {sources} in {self.directory}")
        merged = {key: np.concatenate(values) for key, values in parts.items()}
        merged["channels"] = np.asarray(channels, dtype=np.int32)
        return merged

    # ------------------------------------------------------------------ #
    # statistics
    # ------------------------------------------------------------------ #

    def stats(self) -> Dict:
        """Per-source game/sample counts, mean game length, action histogram."""
        per_source: Dict[str, Dict] = {}
        total_games = total_samples = 0
        for path in self.shard_files:
            with np.load(path) as data:
                sources = data["sources"]
                game_ids = data["game_ids"]
                actions = data["actions"]
                for code in np.unique(sources):
                    name = CODE_TO_SOURCE[int(code)]
                    sel = sources == code
                    entry = per_source.setdefault(name, {
                        "games": set(), "samples": 0,
                        "action_hist": np.zeros(192, dtype=np.int64),
                    })
                    entry["games"].update(int(g) for g in np.unique(game_ids[sel]))
                    entry["samples"] += int(sel.sum())
                    entry["action_hist"] += np.bincount(actions[sel], minlength=192)
        for name, entry in per_source.items():
            games = len(entry["games"])
            entry["games"] = games
            entry["avg_length"] = entry["samples"] / max(games, 1)
            total_games += games
            total_samples += entry["samples"]
        return {
            "total_games": total_games,
            "total_samples": total_samples,
            "sources": per_source,
        }

    # ------------------------------------------------------------------ #
    # splitting
    # ------------------------------------------------------------------ #

    def game_split(
        self,
        n_samples: int,
        game_ids: np.ndarray,
        val_fraction: float = 0.1,
        seed: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Deterministic train/validation index split BY GAME.

        Splitting by game (not by sample) prevents near-duplicate states
        from one game leaking into both subsets. ``n_samples`` and
        ``game_ids`` come from :meth:`load`.
        """
        unique_games = np.unique(game_ids)
        rng = np.random.default_rng(seed)
        rng.shuffle(unique_games)
        n_val = max(1, int(round(len(unique_games) * val_fraction)))
        val_games = set(int(g) for g in unique_games[:n_val])
        is_val = np.isin(game_ids, list(val_games))
        train_idx = np.flatnonzero(~is_val)
        val_idx = np.flatnonzero(is_val)
        if not len(train_idx) or not len(val_idx):
            raise ValueError("Split produced an empty subset — add more games.")
        return train_idx, val_idx
