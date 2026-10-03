"""Tests for the instrumentation layer: reward/observation profiles,
metrics callbacks, dashboard helpers, experiments, visualization,
evaluation stats, expert datasets, reproducibility."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from environment.block_blast_env import (
    REWARD_COMPONENTS,
    BlockBlastEnv,
    encode_action,
)
from environment.observations import build_observation
from environment.rewards import REWARD_PROFILES, get_reward_config
from game import BlockBlastGame
from helpers import make_piece, set_pieces


# ---------------------------------------------------------------------- #
# reward profiles
# ---------------------------------------------------------------------- #

def test_reward_profiles_exist_and_build():
    for name in ("baseline", "survival", "lines", "strategic"):
        env = BlockBlastEnv(reward_profile=name)
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
    with pytest.raises(ValueError):
        get_reward_config("nonexistent")


def test_baseline_profile_matches_original_defaults():
    cfg = REWARD_PROFILES["baseline"]
    assert cfg.place_per_cell == 1.0
    assert cfg.line == 10.0
    assert cfg.combo == 5.0
    assert cfg.game_over == 50.0
    assert cfg.holes == 0.1


def test_reward_components_logged_per_step():
    env = BlockBlastEnv(reward_profile="baseline")
    env.reset(seed=0)
    mask = env.action_masks()
    _, reward, terminated, _, info = env.step(int(np.flatnonzero(mask)[0]))
    components = info["reward_components"]
    assert set(components) == set(REWARD_COMPONENTS)
    assert reward == pytest.approx(sum(components.values()))
    if terminated:
        assert "episode_reward_components" in info


def test_survival_profile_pays_per_move():
    env = BlockBlastEnv(reward_profile="survival")
    env.reset(seed=0)
    mask = env.action_masks()
    _, _, terminated, _, info = env.step(int(np.flatnonzero(mask)[0]))
    if not terminated:
        assert info["reward_components"]["survival"] == pytest.approx(0.2)


def test_strategic_profile_penalizes_holes_and_counts_future_moves():
    env = BlockBlastEnv(reward_profile="strategic")
    env.reset(seed=0)
    mask = env.action_masks()
    _, _, _, _, info = env.step(int(np.flatnonzero(mask)[0]))
    components = info["reward_components"]
    assert components["future_moves"] > 0  # fresh board has many options
    assert components["occupancy"] < 0  # occupied cells cost something
    assert components["fragmentation"] < 0


# ---------------------------------------------------------------------- #
# observation profiles
# ---------------------------------------------------------------------- #

def test_observation_profiles_shapes():
    basic = BlockBlastEnv(observation_profile="basic")
    enhanced = BlockBlastEnv(observation_profile="enhanced")
    assert basic.reset(seed=0)[0].shape == (4, 8, 8)
    assert enhanced.reset(seed=0)[0].shape == (8, 8, 8)


def test_enhanced_channels_are_derived_from_state():
    game = BlockBlastGame(seed=5)
    rng = np.random.default_rng(0)
    for _ in range(15):
        actions = game.get_valid_actions()
        if not actions:  # larger pieces can end the game within 15 moves
            break
        game.place_piece(*actions[int(rng.integers(0, len(actions)))])
    obs = build_observation(game, "enhanced")
    assert obs.shape == (8, 8, 8)
    # channel 4: column heights broadcast
    from game.utils import column_heights, count_holes

    np.testing.assert_allclose(obs[4][0], column_heights(game.board) / 8)
    # channel 5: holes mask matches count_holes
    assert obs[5].sum() == count_holes(game.board)
    # channel 7: legal-placement density only where placements exist
    assert 0 <= obs[7].min() and obs[7].max() <= 1.0
    if game.has_any_valid_move():
        assert obs[7].sum() > 0


def test_enhanced_observation_is_markovian_deterministic():
    game = BlockBlastGame(seed=11)
    a = build_observation(game, "enhanced")
    b = build_observation(game, "enhanced")
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------- #
# metrics callbacks (dashboard logging)
# ---------------------------------------------------------------------- #

def _tiny_train(tmp_path, timesteps=256):
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv

    from training.metrics import EpisodeMetricsCallback, TrainMetricsCallback
    from training.train import mask_fn

    def make():
        env = BlockBlastEnv()
        env = ActionMasker(env, mask_fn)
        env.reset(seed=0)
        return env

    vec = DummyVecEnv([make])
    model = MaskablePPO("MlpPolicy", vec, n_steps=64, batch_size=64, n_epochs=1,
                        verbose=0)
    ep_csv = tmp_path / "episodes.csv"
    metrics_csv = tmp_path / "metrics.csv"
    model.learn(total_timesteps=timesteps,
                callback=[EpisodeMetricsCallback(ep_csv),
                          TrainMetricsCallback(metrics_csv)])
    vec.close()
    return ep_csv, metrics_csv


def test_metrics_callbacks_write_csvs(tmp_path):
    ep_csv, metrics_csv = _tiny_train(tmp_path)
    with ep_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert rows, "episode CSV should have rows"
    for col in ("timesteps", "score", "reward", "moves", "lines_cleared",
                "valid_actions", "invalid_termination", "reward_place",
                "reward_game_over"):
        assert col in rows[0]
    with metrics_csv.open() as f:
        mrows = list(csv.DictReader(f))
    assert mrows, "metrics CSV should have rows"
    for col in ("timesteps", "fps", "entropy_loss", "approx_kl",
                "policy_gradient_loss", "value_loss", "explained_variance",
                "learning_rate"):
        assert col in mrows[0]
    # at least one row with actual train stats (non-empty entropy)
    assert any(r["entropy_loss"] for r in mrows)


def test_dashboard_helpers_align_series(tmp_path):
    from training.dashboard import rolling_mean, series

    rows = [{"timesteps": "1", "value": "2.0"},
            {"timesteps": "2", "value": ""},   # missing y -> row skipped
            {"timesteps": "3", "value": "4.0"}]
    xs, ys = series(rows, "timesteps", "value")
    assert len(xs) == len(ys) == 2
    np.testing.assert_array_equal(rolling_mean(np.array([1.0, 2.0, 3.0]), 2),
                                  np.array([1.5, 2.5]))


def test_dashboard_update_renders_png(tmp_path):
    from training.dashboard import Dashboard

    ep_csv, metrics_csv = _tiny_train(tmp_path, timesteps=192)
    dash = Dashboard("test_run")
    dash.episode_log = ep_csv
    dash.metrics_log = metrics_csv
    dash.out_path = tmp_path / "dash.png"
    dash.update()
    assert dash.out_path.exists() and dash.out_path.stat().st_size > 0


# ---------------------------------------------------------------------- #
# experiment configuration
# ---------------------------------------------------------------------- #

def test_experiment_matrix_is_valid():
    from training.config import TrainConfig
    from training.experiments import EXPERIMENTS

    names = [e["name"] for e in EXPERIMENTS]
    assert len(names) >= 3
    assert len(names) == len(set(names)), "experiment names must be unique"
    valid_fields = set(TrainConfig().to_dict())
    for exp in EXPERIMENTS:
        unknown = set(exp["overrides"]) - valid_fields
        assert not unknown, f"{exp['name']} overrides unknown fields: {unknown}"


def test_config_roundtrip():
    from training.config import TrainConfig

    cfg = TrainConfig(ent_coef=0.03, reward_profile="strategic",
                      observation_profile="enhanced", n_envs=4)
    clone = TrainConfig.from_dict(cfg.to_dict())
    assert clone.ent_coef == 0.03
    assert clone.reward_profile == "strategic"
    assert clone.observation_profile == "enhanced"
    assert clone.n_envs == 4


# ---------------------------------------------------------------------- #
# network visualization
# ---------------------------------------------------------------------- #

@pytest.fixture()
def tiny_model(tmp_path):
    sb3_contrib = pytest.importorskip("sb3_contrib")
    from training.cnn_extractor import BlockBlastCNN

    env = BlockBlastEnv()
    model = sb3_contrib.MaskablePPO(
        "MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1, verbose=0,
        policy_kwargs=dict(
            features_extractor_class=BlockBlastCNN,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
    )
    path = tmp_path / "viz_model.zip"
    model.save(str(path))
    return str(path)


def test_network_visualization_modes(tmp_path, tiny_model):
    from training.visualize_network import (
        load_model,
        visualize_activations,
        visualize_architecture,
        visualize_policy,
        visualize_weights,
    )

    model = load_model(tiny_model)
    game = BlockBlastGame(seed=0)
    outputs = [
        visualize_architecture(model, tmp_path / "arch.png"),
        visualize_weights(model, tmp_path / "weights.png"),
        visualize_activations(model, game, tmp_path / "acts.png", "basic"),
        visualize_policy(model, game, tmp_path / "policy.png", "basic"),
    ]
    for path in outputs:
        assert path.exists() and path.stat().st_size > 0


# ---------------------------------------------------------------------- #
# evaluation metrics
# ---------------------------------------------------------------------- #

def test_summarize_statistics():
    from training.evaluate import summarize

    scores = list(range(1, 101))  # 1..100
    stats = summarize(scores, [10] * 100, [2] * 100)
    assert stats["avg_score"] == pytest.approx(50.5)
    assert stats["median_score"] == pytest.approx(50.5)
    assert stats["best_score"] == 100
    assert stats["worst_score"] == 1
    assert stats["p25_score"] == pytest.approx(25.75)
    assert stats["p75_score"] == pytest.approx(75.25)
    assert stats["avg_moves"] == 10
    assert stats["avg_lines_cleared"] == 2


def test_identical_seeds_make_evaluation_reproducible():
    from training.evaluate import evaluate_agent

    a = evaluate_agent("random", 10, base_seed=777, progress=False)
    b = evaluate_agent("random", 10, base_seed=777, progress=False)
    assert a["scores"] == b["scores"]


# ---------------------------------------------------------------------- #
# expert dataset generation
# ---------------------------------------------------------------------- #

def test_expert_dataset_generation(tmp_path):
    from training.generate_expert_data import generate

    out = generate(games=3, base_seed=60_000, out=tmp_path / "ds.npz",
                   progress=False)
    data = np.load(out)
    n = len(data["actions"])
    assert n > 0
    assert data["observations"].shape == (n, 4, 8, 8)
    assert data["action_masks"].shape == (n, 192)
    assert data["game_scores"].shape == (n,)
    # every recorded expert action must be legal in its mask
    masks = data["action_masks"]
    actions = data["actions"]
    assert all(masks[i, a] for i, a in enumerate(actions))


# ---------------------------------------------------------------------- #
# reproducibility
# ---------------------------------------------------------------------- #

def test_collect_run_metadata_keys():
    from training.repro import collect_run_metadata

    meta = collect_run_metadata({"seed": 0, "total_timesteps": 100})
    for key in ("python_version", "torch_version", "stable_baselines3_version",
                "sb3_contrib_version", "gymnasium_version", "cuda_version",
                "gpu", "git_commit", "config"):
        assert key in meta
    assert meta["config"]["seed"] == 0


def test_write_config_json(tmp_path):
    from training.repro import write_config_json

    path = write_config_json(tmp_path / "config.json", {"seed": 42})
    data = json.loads(path.read_text())
    assert data["config"]["seed"] == 42
    assert "torch_version" in data
