"""Regression tests for the training-pipeline semantics.

Covers the bugs found during the training-reliability investigation:

* ``--timesteps`` vs ``--until-total`` session-step computation;
* SB3's ``learn(total_timesteps, reset_num_timesteps=False)`` semantics
  (ADDITIONAL steps — the assumption ``--resume`` is built on);
* resume hyperparameter resolution (checkpoint values kept, only
  lr/ent_coef/clip_range overridable, structural knobs ignored);
* ``reward_scale`` (optimizer-facing scaling, raw component logs);
* the ``valid_actions`` episode metric (mean in-episode legal actions,
  not the always-zero terminal count);
* the separate-value-network policy option.
"""

import csv
import dataclasses

import numpy as np
import pytest

from environment.block_blast_env import BlockBlastEnv
from environment.rewards import get_reward_config
from training.config import PPO_DEFAULTS, TrainConfig
from training.train import (
    apply_hyperparam_overrides,
    compute_session_steps,
    mask_fn,
    resolve_ppo_params,
)


# ---------------------------------------------------------------------- #
# session-step computation
# ---------------------------------------------------------------------- #

def test_session_steps_without_until_total():
    cfg = TrainConfig(total_timesteps=1_000_000)
    assert compute_session_steps(cfg, current_steps=0) == 1_000_000
    # --timesteps means ADDITIONAL steps, regardless of the model's counter
    assert compute_session_steps(cfg, current_steps=25_000_000) == 1_000_000


def test_session_steps_until_total_converts_to_remaining():
    cfg = TrainConfig(total_timesteps=1_000_000, until_total=35_000_000)
    assert compute_session_steps(cfg, current_steps=26_371_792) == 8_628_208


def test_session_steps_until_total_rejects_past_target():
    cfg = TrainConfig(until_total=10_000_000)
    with pytest.raises(ValueError, match="nothing to do"):
        compute_session_steps(cfg, current_steps=10_000_000)
    with pytest.raises(ValueError, match="nothing to do"):
        compute_session_steps(cfg, current_steps=26_000_000)


# ---------------------------------------------------------------------- #
# SB3 timestep semantics (the assumption --resume relies on)
# ---------------------------------------------------------------------- #

def _tiny_model(vec_env, **kwargs):
    from sb3_contrib import MaskablePPO

    return MaskablePPO("MlpPolicy", vec_env, n_steps=32, batch_size=32,
                       n_epochs=1, verbose=0, **kwargs)


def _tiny_vec_env():
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make():
        env = BlockBlastEnv()
        env = ActionMasker(env, mask_fn)
        env.reset(seed=0)
        return env

    return DummyVecEnv([make])


def test_sb3_learn_reset_false_trains_additional_steps():
    vec = _tiny_vec_env()
    model = _tiny_model(vec)
    model.learn(total_timesteps=64)
    assert model.num_timesteps == 64
    model.learn(total_timesteps=64, reset_num_timesteps=False)
    # additional 64 steps -> 128 total, NOT 64
    assert model.num_timesteps == 128
    vec.close()


def test_sb3_learn_reset_true_restarts_counter():
    vec = _tiny_vec_env()
    model = _tiny_model(vec)
    model.learn(total_timesteps=64)
    model.learn(total_timesteps=64, reset_num_timesteps=True)
    assert model.num_timesteps == 64
    vec.close()


def test_resume_preserves_checkpoint_hyperparameters(tmp_path):
    from sb3_contrib import MaskablePPO

    vec = _tiny_vec_env()
    model = _tiny_model(vec, ent_coef=0.05, gamma=0.9, learning_rate=1e-4)
    model.learn(total_timesteps=32)
    path = tmp_path / "ckpt.zip"
    model.save(str(path))
    optimizer_state = model.policy.optimizer.state_dict()
    assert optimizer_state["state"], "trained checkpoint should have optimizer state"
    vec.close()

    vec2 = _tiny_vec_env()
    loaded = MaskablePPO.load(str(path), env=vec2, device="cpu")
    assert loaded.num_timesteps == 32
    assert loaded.ent_coef == 0.05
    assert loaded.gamma == 0.9
    assert loaded.learning_rate == 1e-4
    assert loaded.policy.optimizer.state_dict()["state"]
    vec2.close()


def test_init_from_policy_starts_fresh_optimizer_and_timestep():
    vec_source = _tiny_vec_env()
    source = _tiny_model(vec_source)
    source.learn(total_timesteps=32)
    source_state = source.policy.state_dict()
    assert source.num_timesteps == 32
    assert source.policy.optimizer.state_dict()["state"]
    vec_source.close()

    vec_fresh = _tiny_vec_env()
    fresh = _tiny_model(vec_fresh)
    fresh.policy.load_state_dict(source_state)
    assert fresh.num_timesteps == 0
    assert fresh.policy.optimizer.state_dict()["state"] == {}
    vec_fresh.close()


# ---------------------------------------------------------------------- #
# hyperparameter resolution
# ---------------------------------------------------------------------- #

def test_resolve_fresh_run_uses_defaults_and_explicit_values():
    cfg = TrainConfig()  # everything None
    params, overrides, _ = resolve_ppo_params(cfg, resuming=False)
    assert params == PPO_DEFAULTS
    assert overrides == {}

    cfg = TrainConfig(ent_coef=0.03, learning_rate=1e-4)
    params, _, _ = resolve_ppo_params(cfg, resuming=False)
    assert params["ent_coef"] == 0.03
    assert params["learning_rate"] == 1e-4
    assert params["gamma"] == PPO_DEFAULTS["gamma"]  # untouched


def test_resolve_resume_keeps_checkpoint_and_limits_overrides():
    cfg = TrainConfig()  # nothing explicit
    params, overrides, notes = resolve_ppo_params(cfg, resuming=True)
    assert params == {} and overrides == {}
    assert all("from checkpoint" in n for n in notes)

    # lr/ent_coef/clip_range are the only adjustable knobs on resume
    cfg = TrainConfig(learning_rate=1e-4, ent_coef=0.02, clip_range=0.1)
    _, overrides, _ = resolve_ppo_params(cfg, resuming=True)
    assert overrides == {"learning_rate": 1e-4, "ent_coef": 0.02,
                         "clip_range": 0.1}

    # structural knobs must NOT be applied to a resumed model
    cfg = TrainConfig(n_steps=1024, gamma=0.9, batch_size=2048)
    _, overrides, notes = resolve_ppo_params(cfg, resuming=True)
    assert overrides == {}
    assert any("IGNORED" in n and "n_steps" in n for n in notes)
    assert any("IGNORED" in n and "gamma" in n for n in notes)


def test_apply_hyperparam_overrides_updates_schedules():
    vec = _tiny_vec_env()
    model = _tiny_model(vec)
    apply_hyperparam_overrides(model, {
        "learning_rate": 1e-4, "ent_coef": 0.07, "clip_range": 0.15,
    })
    assert model.ent_coef == 0.07
    # schedules, not just attributes: PPO re-reads these every update
    assert model.lr_schedule(0.5) == pytest.approx(1e-4)
    assert model.clip_range(0.5) == pytest.approx(0.15)
    vec.close()


# ---------------------------------------------------------------------- #
# reward_scale
# ---------------------------------------------------------------------- #

def test_reward_scale_scales_optimizer_reward_only():
    base = get_reward_config("lines")
    scaled = dataclasses.replace(base, reward_scale=0.1)

    env_raw = BlockBlastEnv(reward_config=base)
    env_scaled = BlockBlastEnv(reward_config=scaled)
    env_raw.reset(seed=3)
    env_scaled.reset(seed=3)

    action = int(np.flatnonzero(env_raw.action_masks())[0])
    _, reward_raw, _, _, info_raw = env_raw.step(action)
    _, reward_scaled, _, _, info_scaled = env_scaled.step(action)

    # same deterministic first move -> same raw components...
    assert info_raw["reward_components"] == info_scaled["reward_components"]
    raw_sum = sum(info_raw["reward_components"].values())
    assert reward_raw == pytest.approx(raw_sum)
    # ...but the optimizer-facing reward is scaled
    assert reward_scaled == pytest.approx(0.1 * raw_sum)


def test_reward_scale_default_is_noop():
    cfg = get_reward_config("strategic")
    assert cfg.reward_scale == 1.0


# ---------------------------------------------------------------------- #
# valid_actions episode metric
# ---------------------------------------------------------------------- #

def test_episode_metric_valid_actions_is_in_episode_mean(tmp_path):
    from sb3_contrib import MaskablePPO

    from training.metrics import EpisodeMetricsCallback

    vec = _tiny_vec_env()
    model = MaskablePPO("MlpPolicy", vec, n_steps=64, batch_size=64,
                        n_epochs=1, verbose=0)
    ep_csv = tmp_path / "episodes.csv"
    model.learn(total_timesteps=256, callback=[EpisodeMetricsCallback(ep_csv)])
    vec.close()

    with ep_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert rows, "expected at least one finished episode"
    assert "final_valid_actions" in rows[0]
    # the episode metric is the in-episode mean: strictly positive
    assert any(float(r["valid_actions"]) > 0 for r in rows)
    # the terminal count is 0 by definition at game over (sanity column)
    assert all(int(r["final_valid_actions"]) == 0 for r in rows)


# ---------------------------------------------------------------------- #
# separate value network option
# ---------------------------------------------------------------------- #

def test_separate_value_net_builds_independent_extractors():
    from sb3_contrib import MaskablePPO

    from training.cnn_extractor import BlockBlastCNN

    vec = _tiny_vec_env()
    model = MaskablePPO(
        "MlpPolicy", vec, n_steps=8, batch_size=8, n_epochs=1, verbose=0,
        policy_kwargs=dict(
            features_extractor_class=BlockBlastCNN,
            features_extractor_kwargs=dict(features_dim=64),
            net_arch=dict(pi=[64], vf=[64]),
            share_features_extractor=False,
        ),
    )
    assert model.policy.share_features_extractor is False
    assert model.policy.pi_features_extractor is not \
        model.policy.vf_features_extractor
    vec.close()
