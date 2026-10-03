"""Tests for the Gymnasium environment."""

import numpy as np
import pytest

from environment.block_blast_env import (
    NUM_ACTIONS,
    BlockBlastEnv,
    RewardConfig,
    decode_action,
    encode_action,
)


def test_action_encode_decode_roundtrip():
    for action in range(NUM_ACTIONS):
        assert encode_action(*decode_action(action)) == action
    assert decode_action(encode_action(1, 4, 6)) == (1, 4, 6)


def test_reset_observation_contract():
    env = BlockBlastEnv()
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs.dtype == np.float32
    assert obs.shape == (4, 8, 8)
    # fresh board: channel 0 empty, all three piece channels non-empty
    assert obs[0].sum() == 0
    assert all(obs[1 + slot].sum() > 0 for slot in range(3))
    assert info["score"] == 0
    assert info["valid_actions"] > 0


def test_step_contract():
    env = BlockBlastEnv()
    env.reset(seed=0)
    mask = env.action_masks()
    action = int(np.flatnonzero(mask)[0])

    obs, reward, terminated, truncated, info = env.step(action)

    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated is False
    # placing a piece with no line clear gives cells * place_per_cell
    piece_idx, _, _ = decode_action(action)
    assert reward > 0


def test_mask_matches_engine_valid_actions():
    env = BlockBlastEnv()
    env.reset(seed=7)
    expected = np.zeros(NUM_ACTIONS, dtype=bool)
    for piece_idx, row, col in env.game.get_valid_actions():
        expected[encode_action(piece_idx, row, col)] = True
    np.testing.assert_array_equal(env.action_masks(), expected)


def test_used_piece_channel_is_zero():
    env = BlockBlastEnv()
    env.reset(seed=0)
    # play one action from piece slot 0
    mask = env.action_masks()
    slot0_actions = [a for a in np.flatnonzero(mask) if decode_action(a)[0] == 0]
    obs, *_ = env.step(int(slot0_actions[0]))
    assert obs[1].sum() == 0  # slot 0 channel empty
    assert obs[2].sum() > 0 and obs[3].sum() > 0


def test_invalid_action_terminates_with_penalty():
    env = BlockBlastEnv(reward_config=RewardConfig(game_over=50.0))
    env.reset(seed=0)
    mask = env.action_masks()
    invalid_action = int(np.flatnonzero(~mask)[0])

    _, reward, terminated, truncated, _ = env.step(invalid_action)

    assert terminated is True
    assert reward == -50.0


def test_deterministic_seeding():
    def run(seed):
        env = BlockBlastEnv()
        obs, _ = env.reset(seed=seed)
        trajectory = [obs.copy()]
        rng = np.random.default_rng(123)
        for _ in range(20):
            mask = env.action_masks()
            valid = np.flatnonzero(mask)
            if len(valid) == 0:
                break
            action = int(rng.choice(valid))
            obs, reward, terminated, _, info = env.step(action)
            trajectory.append((obs.copy(), reward, info["score"]))
            if terminated:
                break
        return trajectory

    traj_a = run(42)
    traj_b = run(42)
    assert len(traj_a) == len(traj_b)
    for step_a, step_b in zip(traj_a, traj_b):
        if isinstance(step_a, tuple):
            np.testing.assert_array_equal(step_a[0], step_b[0])
            assert step_a[1] == step_b[1]
            assert step_a[2] == step_b[2]
        else:
            np.testing.assert_array_equal(step_a, step_b)


def test_full_episode_terminates_and_applies_game_over_penalty():
    env = BlockBlastEnv(reward_config=RewardConfig(game_over=50.0))
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    last_reward = 0.0
    terminated = False
    for _ in range(10_000):
        valid = np.flatnonzero(env.action_masks())
        if len(valid) == 0:
            break
        _, last_reward, terminated, _, info = env.step(int(rng.choice(valid)))
        if terminated:
            break
    assert terminated, "random play should eventually end the game"
    assert last_reward <= -50.0 + 4 * 1.0 + 3 * 15  # penalty clearly included
    assert info["score"] > 0


def test_line_clear_reward():
    env = BlockBlastEnv(reward_config=RewardConfig(place_per_cell=1.0, line=10.0,
                                                   combo=5.0, holes=0.0))
    env.reset(seed=0)
    # construct a board where a 1x1 piece completes a row
    from helpers import make_piece, set_pieces

    env.game.board[:] = 0
    env.game.board[0, 1:] = 1
    set_pieces(env.game, [make_piece([(0, 0)])] + env.game.pieces[1:])
    action = encode_action(0, 0, 0)
    _, reward, *_ = env.step(action)
    # 1 cell * 1.0 + 1 line * 10 + 5 * 1**2 = 16
    assert reward == pytest.approx(16.0)


def test_render_returns_text():
    env = BlockBlastEnv(render_mode="ansi")
    env.reset(seed=0)
    text = env.render()
    assert "BlockBlastGame" in text
