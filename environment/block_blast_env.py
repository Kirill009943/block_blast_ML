"""Gymnasium environment wrapping the pure Block Blast engine.

Observation
-----------
Selectable via ``observation_profile`` (see ``environment/observations.py``):

* ``basic``    — ``Box(0, 1, shape=(4, 8, 8), float32)``: board occupancy
  plus one binary channel per piece slot (zeros when the slot is used).
* ``enhanced`` — 8 channels: basic + column heights, holes, line-completion
  potential and legal-placement density.

Both are Markovian (functions of board + piece slots only) and feed a CNN
directly.

Actions
-------
``Discrete(192)`` = 3 pieces x 8 rows x 8 cols. Decoding::

    piece_index = action // 64
    row         = (action % 64) // 8
    col         = action % 8

Most combinations are illegal at any moment. Use :meth:`action_masks`
(consumed by sb3-contrib's ``MaskablePPO``) so the agent only ever samples
legal actions. ``step`` still defends itself: an illegal action terminates
the episode with the game-over penalty.

Rewards
-------
All weights live in :class:`~environment.rewards.RewardConfig`; named
profiles (``baseline``/``survival``/``lines``/``strategic``) live in
``environment/rewards.py``. Every component is returned per step in
``info["reward_components"]`` and summed per episode in
``info["episode_reward_components"]`` (on termination), so training logs
show exactly which term the agent is optimizing. Components are logged in
RAW game units; ``reward_scale`` (if set) multiplies only the scalar
reward the optimizer sees, so component sums may differ from the reward
by that constant factor.

Determinism: ``env.reset(seed=42)`` reseeds the engine's RNG, so the same
seed yields the same piece sequence every time.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment.observations import (
    DEFAULT_OBSERVATION_PROFILE,
    build_observation,
    observation_channels,
)
from environment.rewards import RewardConfig, get_reward_config
from game.game import BOARD_SIZE, BlockBlastGame, GameConfig
from game.utils import count_empty_regions, count_holes

NUM_PIECE_SLOTS = 3
NUM_ACTIONS = NUM_PIECE_SLOTS * BOARD_SIZE * BOARD_SIZE  # 192

REWARD_COMPONENTS = (
    "place",
    "lines",
    "combo",
    "holes",
    "fragmentation",
    "occupancy",
    "future_moves",
    "future_moves_delta",
    "fragmentation_delta",
    "survival",
    "game_over",
)


def decode_action(action: int) -> Tuple[int, int, int]:
    """Map a discrete action id to ``(piece_index, row, col)``."""
    piece_index = action // (BOARD_SIZE * BOARD_SIZE)
    cell = action % (BOARD_SIZE * BOARD_SIZE)
    return piece_index, cell // BOARD_SIZE, cell % BOARD_SIZE


def encode_action(piece_index: int, row: int, col: int) -> int:
    """Map ``(piece_index, row, col)`` to its discrete action id."""
    return piece_index * BOARD_SIZE * BOARD_SIZE + row * BOARD_SIZE + col


def build_action_mask(game: BlockBlastGame) -> np.ndarray:
    """Boolean mask over the 192 actions for an arbitrary engine state."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for piece_index, row, col in game.get_valid_actions():
        mask[encode_action(piece_index, row, col)] = True
    return mask


class BlockBlastEnv(gym.Env):
    """Headless Gymnasium env for Block Blast. Rendering is text-only."""

    metadata = {"render_modes": ["ansi"], "render_fps": 30}

    def __init__(
        self,
        reward_config: Optional[RewardConfig] = None,
        reward_profile: Optional[str] = None,
        observation_profile: str = DEFAULT_OBSERVATION_PROFILE,
        game_config: Optional[GameConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        if reward_config is not None and reward_profile is not None:
            raise ValueError("Pass either reward_config or reward_profile, not both.")
        if reward_config is None:
            reward_config = get_reward_config(reward_profile or "baseline")
        self.reward_config = reward_config
        self.reward_profile = reward_profile or "baseline"
        self.observation_profile = observation_profile
        self.game = BlockBlastGame(config=game_config)
        self.render_mode = render_mode

        channels = observation_channels(observation_profile)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(channels, BOARD_SIZE, BOARD_SIZE),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self._last_reward = 0.0
        self._episode_reward = 0.0
        self._episode_components = {name: 0.0 for name in REWARD_COMPONENTS}
        # carried board-state "before" values: this step's before is last
        # step's after, so delta components cost one analysis per step
        self._last_holes: Optional[int] = None
        self._last_regions: Optional[int] = None
        # per-episode board-metric accumulators (means reported at done)
        self._board_stats: Dict[str, float] = {}
        self._board_stats_count = 0

    # ------------------------------------------------------------------ #
    # gymnasium API
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.game.reseed(seed)
        self.game.reset()
        self._last_reward = 0.0
        self._episode_reward = 0.0
        self._episode_components = {name: 0.0 for name in REWARD_COMPONENTS}
        self._last_holes = None
        self._last_regions = None
        self._board_stats = {}
        self._board_stats_count = 0
        return self._observation(), self._info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        piece_index, row, col = decode_action(int(action))
        cfg = self.reward_config
        components = {name: 0.0 for name in REWARD_COMPONENTS}

        # "before" snapshots: carried from the previous step when possible
        need_holes = bool(cfg.holes)
        need_regions = bool(cfg.fragmentation or cfg.fragmentation_delta)
        holes_before = self._last_holes if need_holes else 0
        regions_before = self._last_regions if need_regions else 0
        if need_holes and holes_before is None:
            holes_before = count_holes(self.game.board)
        if need_regions and regions_before is None:
            regions_before = count_empty_regions(self.game.board)
        valid_before = (
            len(self.game.get_valid_actions())  # cached: free
            if cfg.future_moves_delta
            else 0
        )

        result = self.game.place_piece(piece_index, row, col)

        if not result.success:
            # Only reachable when the caller ignores action_masks().
            terminated = True
            components["game_over"] = -cfg.game_over
        else:
            lines = result.lines_cleared
            components["place"] = cfg.place_per_cell * result.cells_placed
            components["lines"] = cfg.line * lines
            components["combo"] = cfg.combo * lines * lines
            if need_holes:
                holes_after = count_holes(self.game.board)
                components["holes"] = -cfg.holes * max(0, holes_after - holes_before)
                self._last_holes = holes_after
            if need_regions:
                regions_after = count_empty_regions(self.game.board)
                if cfg.fragmentation:
                    components["fragmentation"] = -cfg.fragmentation * regions_after
                if cfg.fragmentation_delta:
                    delta = regions_after - regions_before
                    components["fragmentation_delta"] = float(np.clip(
                        -cfg.fragmentation_delta * delta,
                        -cfg.fragmentation_delta_clip,
                        cfg.fragmentation_delta_clip,
                    ))
                self._last_regions = regions_after
            if cfg.occupancy:
                components["occupancy"] = -cfg.occupancy * int((self.game.board != 0).sum())
            valid_after = len(self.game.get_valid_actions())  # cached: free
            if cfg.future_moves:
                components["future_moves"] = cfg.future_moves * valid_after
            if cfg.future_moves_delta:
                # a refresh draws a NEW random set: the count change is then
                # driven by generation, not by this placement -> skip
                raw_delta = 0 if result.pieces_refreshed else valid_after - valid_before
                components["future_moves_delta"] = float(np.clip(
                    cfg.future_moves_delta * raw_delta,
                    -cfg.future_moves_delta_clip,
                    cfg.future_moves_delta_clip,
                ))
            terminated = result.game_over
            if terminated:
                components["game_over"] = -cfg.game_over
            elif cfg.survival_per_move:
                components["survival"] = cfg.survival_per_move

        reward = float(sum(components.values()))
        if cfg.reward_scale != 1.0:
            reward *= cfg.reward_scale
        if cfg.reward_clip > 0:
            reward = float(np.clip(reward, -cfg.reward_clip, cfg.reward_clip))
        truncated = False  # game length is not time-limited
        self._last_reward = reward
        self._episode_reward += reward
        for name, value in components.items():
            self._episode_components[name] += value

        # per-episode board-state metrics (means reported at episode end)
        occupied_fraction = float((self.game.board != 0).mean())
        self._board_stats_count += 1
        for key, value in (
            ("holes", float(self._last_holes or 0)),
            ("regions", float(self._last_regions or 0)),
            ("occupancy_fraction", occupied_fraction),
            ("future_moves", float(len(self.game.get_valid_actions()))),
            ("future_moves_delta", components["future_moves_delta"]),
        ):
            self._board_stats[key] = self._board_stats.get(key, 0.0) + value

        info = self._info()
        info["reward_components"] = components
        if terminated:
            info["episode_reward_components"] = dict(self._episode_components)
            info["board_metrics"] = {
                key: value / max(self._board_stats_count, 1)
                for key, value in self._board_stats.items()
            }
        return self._observation(), reward, terminated, truncated, info

    def render(self) -> str:
        return repr(self.game)

    # ------------------------------------------------------------------ #
    # masking & helpers
    # ------------------------------------------------------------------ #

    def action_masks(self) -> np.ndarray:
        """Boolean mask over the 192 actions; True = legal right now."""
        return build_action_mask(self.game)

    @property
    def last_reward(self) -> float:
        return self._last_reward

    def _observation(self) -> np.ndarray:
        return build_observation(self.game, self.observation_profile)

    def _info(self) -> Dict[str, Any]:
        return {
            "score": self.game.score,
            "moves": self.game.moves,
            "lines_cleared": self.game.total_lines_cleared,
            "valid_actions": len(self.game.get_valid_actions()),
            "episode_reward": self._episode_reward,
        }


try:
    gym.register(
        id="BlockBlast-v0",
        entry_point="environment.block_blast_env:BlockBlastEnv",
    )
except gym.error.Error:
    pass  # already registered (module imported twice)
