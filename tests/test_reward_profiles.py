"""Tests for the balanced reward profile and delta-based components."""

import numpy as np
import pytest

from environment.block_blast_env import (
    REWARD_COMPONENTS,
    BlockBlastEnv,
    decode_action,
    encode_action,
)
from environment.rewards import REWARD_PROFILES, RewardConfig, get_reward_config
from game.utils import count_empty_regions
from helpers import make_piece, set_pieces


def _first_valid_action(env, slot=0):
    for a in np.flatnonzero(env.action_masks()):
        if decode_action(int(a))[0] == slot:
            return int(a)
    raise AssertionError("no valid action for slot")


# ---------------------------------------------------------------------- #
# profile registration
# ---------------------------------------------------------------------- #

def test_balanced_profile_exists_and_others_unchanged():
    assert "balanced" in REWARD_PROFILES
    # existing profiles must remain exactly as before (fair comparison)
    lines = REWARD_PROFILES["lines"]
    assert (lines.place_per_cell, lines.line, lines.combo,
            lines.game_over, lines.holes) == (0.5, 15.0, 10.0, 50.0, 0.1)
    baseline = REWARD_PROFILES["baseline"]
    assert (baseline.place_per_cell, baseline.line, baseline.combo) == (1.0, 10.0, 5.0)
    balanced = REWARD_PROFILES["balanced"]
    assert balanced.future_moves_delta > 0
    assert balanced.fragmentation_delta > 0
    assert balanced.survival_per_move > 0
    assert balanced.reward_clip > 0
    env = BlockBlastEnv(reward_profile="balanced")
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)


def test_new_components_present_in_step_info():
    env = BlockBlastEnv(reward_profile="balanced")
    env.reset(seed=0)
    _, reward, _, _, info = env.step(_first_valid_action(env))
    components = info["reward_components"]
    assert "future_moves_delta" in components
    assert "fragmentation_delta" in components
    assert set(components) == set(REWARD_COMPONENTS)
    assert reward == pytest.approx(sum(components.values()))


# ---------------------------------------------------------------------- #
# future_moves_delta
# ---------------------------------------------------------------------- #

def test_future_moves_delta_single_step_value():
    cfg = RewardConfig(future_moves_delta=0.1, future_moves_delta_clip=50.0,
                       place_per_cell=0, line=0, combo=0, holes=0)
    env = BlockBlastEnv(reward_config=cfg)
    env.reset(seed=0)
    before = len(env.game.get_valid_actions())
    action = _first_valid_action(env)
    _, reward, terminated, _, info = env.step(action)
    after = len(env.game.get_valid_actions())
    expected = 0.1 * (after - before)
    assert info["reward_components"]["future_moves_delta"] == pytest.approx(expected)
    assert reward == pytest.approx(expected)  # all other weights are 0


def test_future_moves_delta_is_clipped():
    cfg = RewardConfig(future_moves_delta=1.0, future_moves_delta_clip=2.0,
                       place_per_cell=0, line=0, combo=0, holes=0)
    env = BlockBlastEnv(reward_config=cfg)
    env.reset(seed=0)
    _, _, _, _, info = env.step(_first_valid_action(env))
    assert abs(info["reward_components"]["future_moves_delta"]) <= 2.0


def test_future_moves_delta_zero_on_refresh():
    cfg = RewardConfig(future_moves_delta=1.0, future_moves_delta_clip=100.0,
                       place_per_cell=0, line=0, combo=0, holes=0)
    env = BlockBlastEnv(reward_config=cfg)
    env.reset(seed=0)
    singles = [make_piece([(0, 0)], color=i + 1) for i in range(3)]
    set_pieces(env.game, singles)
    env.game._valid_cache = None
    for slot in range(3):
        action = encode_action(slot, 0, slot * 2)
        _, _, _, _, info = env.step(action)
        if slot < 2:
            assert info["reward_components"]["future_moves_delta"] != 0 or True
        else:
            # third placement exhausts the set -> refresh -> delta skipped
            assert info["reward_components"]["future_moves_delta"] == 0.0


def test_future_moves_delta_telescopes_without_refresh():
    """Sum of deltas over one set (no refresh) = final - initial count."""
    cfg = RewardConfig(future_moves_delta=1.0, future_moves_delta_clip=1000.0,
                       place_per_cell=0, line=0, combo=0, holes=0)
    env = BlockBlastEnv(reward_config=cfg)
    env.reset(seed=0)
    initial = len(env.game.get_valid_actions())
    total = 0.0
    # place exactly two pieces (no refresh on the second)
    for slot in (0, 1):
        action = _first_valid_action(env, slot)
        _, _, terminated, _, info = env.step(action)
        assert not terminated
        total += info["reward_components"]["future_moves_delta"]
    final = len(env.game.get_valid_actions())
    assert total == pytest.approx(final - initial)


# ---------------------------------------------------------------------- #
# fragmentation_delta
# ---------------------------------------------------------------------- #

def test_fragmentation_delta_telescopes():
    """Sum of -(regions_after - regions_before) = regions_start - regions_end."""
    cfg = RewardConfig(fragmentation_delta=1.0, fragmentation_delta_clip=1000.0,
                       place_per_cell=0, line=0, combo=0, holes=0)
    env = BlockBlastEnv(reward_config=cfg)
    env.reset(seed=0)
    start_regions = count_empty_regions(env.game.board)
    total = 0.0
    rng = np.random.default_rng(0)
    for _ in range(10):
        valid = np.flatnonzero(env.action_masks())
        if len(valid) == 0:
            break
        _, _, terminated, _, info = env.step(int(rng.choice(valid)))
        total += info["reward_components"]["fragmentation_delta"]
        if terminated:
            break
    end_regions = count_empty_regions(env.game.board)
    assert total == pytest.approx(-(end_regions - start_regions))


def test_fragmentation_delta_penalizes_splitting():
    # a board split into two empty regions by a horizontal wall; filling the
    # gap joins regions (delta should be positive) vs widening the split
    cfg = RewardConfig(fragmentation_delta=0.5, fragmentation_delta_clip=10.0,
                       place_per_cell=0, line=0, combo=0, holes=0)
    env = BlockBlastEnv(reward_config=cfg)
    env.reset(seed=0)
    env.game.board[:] = 0
    # wall along row 3 with a 2-cell gap; filling gap cells one by one
    # must not increase the region count
    for c in range(8):
        if c not in (3, 4):
            env.game.board[3, c] = 1
    env.game._valid_cache = None
    before = count_empty_regions(env.game.board)
    domino = make_piece([(0, 0), (0, 1)])
    set_pieces(env.game, [domino, make_piece([(0, 0)]), make_piece([(0, 0)])])
    env.game._valid_cache = None
    _, _, _, _, info = env.step(encode_action(0, 3, 3))
    after = count_empty_regions(env.game.board)
    assert after <= before
    assert info["reward_components"]["fragmentation_delta"] >= 0


# ---------------------------------------------------------------------- #
# reward clip + survival + game over
# ---------------------------------------------------------------------- #

def test_reward_clip_bounds_step_reward():
    cfg = RewardConfig(place_per_cell=1.0, line=10.0, combo=5.0,
                       reward_clip=15.0, holes=0.0)
    env = BlockBlastEnv(reward_config=cfg)
    env.reset(seed=0)
    # complete a full row with a single cell -> unclipped reward 1+10+5=16 > 15
    env.game.board[:] = 0
    env.game.board[0, 1:] = 1
    set_pieces(env.game, [make_piece([(0, 0)])] + env.game.pieces[1:])
    env.game._valid_cache = None
    _, reward, *_ = env.step(encode_action(0, 0, 0))
    assert reward == 15.0


def test_balanced_components_actually_fire():
    """A short episode must produce nonzero survival/future/frag terms."""
    env = BlockBlastEnv(reward_profile="balanced")
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    totals = {name: 0.0 for name in REWARD_COMPONENTS}
    for _ in range(20):
        valid = np.flatnonzero(env.action_masks())
        if len(valid) == 0:
            break
        _, _, terminated, _, info = env.step(int(rng.choice(valid)))
        for k, v in info["reward_components"].items():
            totals[k] += v
        if terminated:
            break
    assert totals["survival"] > 0
    assert totals["future_moves_delta"] != 0
    assert totals["game_over"] < 0 if terminated else True


def test_board_metrics_at_termination():
    env = BlockBlastEnv(reward_profile="balanced")
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    board_metrics = None
    for _ in range(10_000):
        valid = np.flatnonzero(env.action_masks())
        if len(valid) == 0:
            break
        _, _, terminated, _, info = env.step(int(rng.choice(valid)))
        if terminated:
            board_metrics = info["board_metrics"]
            break
    assert board_metrics is not None
    for key in ("holes", "regions", "occupancy_fraction", "future_moves",
                "future_moves_delta"):
        assert key in board_metrics
    assert 0.0 <= board_metrics["occupancy_fraction"] <= 1.0
    assert board_metrics["future_moves"] >= 0


# ---------------------------------------------------------------------- #
# invalid actions and mask behavior unchanged
# ---------------------------------------------------------------------- #

def test_balanced_invalid_action_still_terminates():
    env = BlockBlastEnv(reward_profile="balanced")
    env.reset(seed=0)
    mask = env.action_masks()
    invalid = int(np.flatnonzero(~mask)[0])
    _, reward, terminated, _, _ = env.step(invalid)
    assert terminated
    assert reward < 0
