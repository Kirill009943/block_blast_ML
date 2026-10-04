"""PPO training for the Block Blast RL agent.

Uses MaskablePPO from sb3-contrib so the policy only ever samples legal
(piece, row, col) placements — the action mask comes from the environment.

Timestep semantics (verified against SB3 2.9 ``base_class._setup_learn``):
``learn(total_timesteps=N, reset_num_timesteps=False)`` trains **N
additional steps** — SB3 adds the model's existing counter internally.
Therefore ``--timesteps`` ALWAYS means "steps trained in this session",
fresh or resumed. ``--until-total N`` converts an absolute target into the
right session length (and refuses to run if the model is already past N).

--resume vs --init-from
-----------------------
``--resume``     full PPO state survives: policy + optimizer state,
                 timestep counter, saved hyperparameters. Only
                 --lr/--ent-coef/--clip-range may override the checkpoint;
                 structural knobs (n_steps, batch, epochs, gamma, ...) are
                 read from the checkpoint and CLI values are ignored.
``--init-from``  copies POLICY WEIGHTS ONLY into a brand-new PPO model
                 (fresh optimizer, fresh value head if the source is a BC
                 checkpoint, timestep counter at 0). Use this to start a
                 NEW training regime from an existing policy.

Examples:
    python -m training.train                              # full run, 10M steps
    python -m training.train --timesteps 50000            # quick smoke run
    python -m training.train --timesteps 5000000 --ent-coef 0.03 --gamma 0.999
    python -m training.train --reward-profile strategic --observation-profile enhanced
    python -m training.train --resume models/strategic_interrupted.zip \\
        --until-total 35000000 --reward-profile strategic \\
        --observation-profile enhanced --run-name strategic_35M

Outputs (for run name ``<run>``):
    models/<run>_final.zip            final model
    models/<run>_interrupted.zip      model saved on Ctrl+C (stop and save)
    models/best/<run>/best_model.zip  best model by masked eval (this run)
    models/checkpoints/<run>/         periodic checkpoints (this run)
    results/training_log_<run>.csv    per-episode metrics (dashboard input)
    results/metrics_<run>.csv         per-update PPO optimizer stats
    results/tensorboard/<run>/        tensorboard logs
    results/config_<run>.json         full reproducibility info

Progress: a ``[timer]`` line prints every ``--progress-interval`` seconds
with steps done, elapsed time, steps/sec and the estimated time remaining.
Pressing Ctrl+C stops training gracefully and saves the current model to
``models/<run>_interrupted.zip`` (resume it with ``--resume``).
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from environment.block_blast_env import BlockBlastEnv
from environment.rewards import RewardConfig, get_reward_config
from training.cnn_extractor import BlockBlastCNN
from training.config import (
    PPO_DEFAULTS,
    RESUMABLE_OVERRIDES,
    TrainConfig,
    add_cli_args,
    config_from_args,
)
from training.metrics import (
    EpisodeMetricsCallback,
    ProgressTimerCallback,
    TrainMetricsCallback,
    format_hms,
)
from training.repro import write_config_json

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


def mask_fn(env: BlockBlastEnv) -> np.ndarray:
    return env.action_masks()


def resolve_reward_config(config: TrainConfig) -> RewardConfig:
    """Profile weights with the run's ``reward_scale`` applied on top."""
    cfg = get_reward_config(config.reward_profile)
    if config.reward_scale != 1.0:
        cfg = dataclasses.replace(cfg, reward_scale=config.reward_scale)
    return cfg


def make_env(config: TrainConfig, reward_config: RewardConfig, rank: int):
    """Factory for one vectorized environment instance (headless)."""

    def _init():
        env = BlockBlastEnv(
            reward_config=reward_config,
            observation_profile=config.observation_profile,
        )
        env = ActionMasker(env, mask_fn)
        env.reset(seed=config.seed + rank)
        return env

    return _init


def compute_session_steps(config: TrainConfig, current_steps: int) -> int:
    """How many steps this session must train.

    ``--until-total`` is converted to a session length here; the result is
    always what should be passed to ``model.learn(total_timesteps=...)``.
    """
    if config.until_total is None:
        return config.total_timesteps
    remaining = config.until_total - current_steps
    if remaining <= 0:
        raise ValueError(
            f"--until-total {config.until_total:,} but the model already has "
            f"{current_steps:,} steps — nothing to do."
        )
    return remaining


def resolve_ppo_params(
    config: TrainConfig, resuming: bool
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """Split PPO settings into (constructor/override params, ignored, notes).

    Returns ``(params, checkpoint_overrides, notes)`` where ``params`` are
    the concrete values for a fresh ``MaskablePPO(...)`` constructor,
    ``checkpoint_overrides`` are the values to apply onto a loaded model
    (only :data:`RESUMABLE_OVERRIDES` keys, only on resume), and ``notes``
    are human-readable lines explaining every decision.
    """
    params: Dict[str, Any] = {}
    overrides: Dict[str, Any] = {}
    notes: List[str] = []
    for key, default in PPO_DEFAULTS.items():
        value = getattr(config, key)
        if resuming:
            if value is None:
                notes.append(f"  {key}: from checkpoint")
            elif key in RESUMABLE_OVERRIDES:
                overrides[key] = value
                notes.append(f"  {key}: {value} (CLI override of the checkpoint value)")
            else:
                notes.append(
                    f"  {key}: from checkpoint (CLI value {value} IGNORED — "
                    f"structural on a resumed model)"
                )
        else:
            params[key] = default if value is None else value
    return params, overrides, notes


def apply_hyperparam_overrides(model: MaskablePPO, overrides: Dict[str, Any]) -> None:
    """Apply the safely-adjustable hyperparameters onto a loaded model.

    ``learning_rate`` and ``clip_range`` must update the schedule objects,
    not just the attributes: PPO re-reads ``lr_schedule``/``clip_range`` at
    every update (``_update_learning_rate``), so the optimizer follows.
    """
    if "learning_rate" in overrides:
        model.learning_rate = overrides["learning_rate"]
        model.lr_schedule = FloatSchedule(overrides["learning_rate"])
    if "clip_range" in overrides:
        model.clip_range = FloatSchedule(overrides["clip_range"])
    if "ent_coef" in overrides:
        model.ent_coef = overrides["ent_coef"]


def print_device_info(device: str) -> str:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: not available — training on CPU")
    print(f"PyTorch: {torch.__version__}")
    return device


def train(config: TrainConfig) -> Tuple[Path, str]:
    """Run one training session. Returns ``(final_model_path, run_name)``."""
    run_name = config.run_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    device = print_device_info(config.device)
    reward_config = resolve_reward_config(config)

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    checkpoint_dir = MODELS_DIR / "checkpoints" / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_dir = MODELS_DIR / "best" / run_name
    best_dir.mkdir(parents=True, exist_ok=True)

    config.run_name = run_name

    vec_env = SubprocVecEnv(
        [make_env(config, reward_config, i) for i in range(config.n_envs)]
    )
    vec_env = VecMonitor(vec_env)

    policy_kwargs = dict(
        features_extractor_class=BlockBlastCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        share_features_extractor=config.share_features_extractor,
    )

    resuming = config.resume is not None
    ppo_params, overrides, notes = resolve_ppo_params(config, resuming)

    if resuming:
        print(f"Resuming from: {config.resume}")
        try:
            model = MaskablePPO.load(
                config.resume, env=vec_env, device=device,
                tensorboard_log=str(RESULTS_DIR / "tensorboard"),
            )
        except ValueError as exc:
            vec_env.close()
            raise ValueError(
                f"Cannot resume {config.resume} with --observation-profile "
                f"{config.observation_profile!r}: {exc}. Pass the observation "
                f"profile the checkpoint was trained with."
            ) from None
        apply_hyperparam_overrides(model, overrides)
        print("Hyperparameters:")
        for line in notes:
            print(line)
    elif config.init_from:
        print(f"Initializing weights from: {config.init_from}")
        model = MaskablePPO(
            "MlpPolicy", vec_env, policy_kwargs=policy_kwargs,
            **ppo_params,
            seed=config.seed, device=device,
            tensorboard_log=str(RESULTS_DIR / "tensorboard"), verbose=1,
        )
        if str(config.init_from).endswith(".pt"):
            # behavioral-cloning checkpoint: policy-side keys match the PPO
            # policy exactly; value head keeps its random init
            from training.imitation_model import load_imitation_state_dict

            state, meta = load_imitation_state_dict(config.init_from, device=device)
            bc_channels = int(meta.get("observation_channels", 4))
            env_channels = int(vec_env.observation_space.shape[0])
            if bc_channels != env_channels:
                vec_env.close()
                raise ValueError(
                    f"BC checkpoint expects {bc_channels} observation channels "
                    f"but the env provides {env_channels} — match "
                    f"--observation-profile to the BC model."
                )
            model.policy.load_state_dict(state, strict=False)
        else:
            source = MaskablePPO.load(config.init_from, device=device)
            source_channels = int(source.observation_space.shape[0])
            env_channels = int(vec_env.observation_space.shape[0])
            if source_channels != env_channels:
                vec_env.close()
                raise ValueError(
                    f"--init-from model expects {source_channels} observation "
                    f"channels but --observation-profile "
                    f"{config.observation_profile!r} provides {env_channels}."
                )
            model.policy.load_state_dict(source.policy.state_dict())
    else:
        model = MaskablePPO(
            "MlpPolicy", vec_env, policy_kwargs=policy_kwargs,
            **ppo_params,
            seed=config.seed, device=device,
            tensorboard_log=str(RESULTS_DIR / "tensorboard"), verbose=1,
        )

    session_steps = compute_session_steps(config, model.num_timesteps)

    eval_env = BlockBlastEnv(
        reward_config=reward_config,
        observation_profile=config.observation_profile,
    )
    eval_env = ActionMasker(eval_env, mask_fn)
    eval_env = Monitor(eval_env)

    callbacks = [
        CheckpointCallback(
            save_freq=max(config.checkpoint_freq // config.n_envs, 1),
            save_path=str(checkpoint_dir),
            name_prefix="ppo_block_blast",
        ),
        MaskableEvalCallback(
            eval_env,
            best_model_save_path=str(best_dir),
            log_path=str(RESULTS_DIR / "eval" / run_name),
            eval_freq=max(config.eval_freq // config.n_envs, 1),
            n_eval_episodes=config.eval_episodes,
            deterministic=True,
            use_masking=True,
        ),
        EpisodeMetricsCallback(RESULTS_DIR / f"training_log_{run_name}.csv"),
        TrainMetricsCallback(RESULTS_DIR / f"metrics_{run_name}.csv"),
    ]
    if config.progress_interval > 0:
        callbacks.append(ProgressTimerCallback(config.progress_interval))

    print(f"Run: {run_name}")
    print(f"Session: +{session_steps:,} steps "
          f"(model at {model.num_timesteps:,} -> "
          f"{model.num_timesteps + session_steps:,}) | envs: {config.n_envs} | "
          f"seed: {config.seed}")
    print(f"Reward profile: {config.reward_profile} "
          f"(reward_scale={reward_config.reward_scale}) | "
          f"Observation profile: {config.observation_profile} | "
          f"shared trunk: {config.share_features_extractor}")
    effective = ppo_params if not resuming else {
        key: getattr(model, key) for key in PPO_DEFAULTS
    }
    print(f"PPO: lr={effective['learning_rate']} gamma={effective['gamma']} "
          f"gae_lambda={effective['gae_lambda']} ent_coef={effective['ent_coef']} "
          f"clip={effective['clip_range']} n_steps={effective['n_steps']} "
          f"batch={effective['batch_size']} epochs={effective['n_epochs']} "
          f"vf_coef={effective['vf_coef']} "
          f"max_grad_norm={effective['max_grad_norm']}")

    # record the EFFECTIVE configuration, after resolution
    config_record = config.to_dict()
    config_record["session_timesteps"] = session_steps
    config_record["model_timesteps_at_start"] = model.num_timesteps
    config_record["effective_ppo"] = {
        key: str(value) for key, value in effective.items()
    }
    write_config_json(RESULTS_DIR / f"config_{run_name}.json", config_record)

    started = time.time()
    start_steps = model.num_timesteps  # > 0 when resuming
    interrupted = False
    try:
        model.learn(
            total_timesteps=session_steps,
            callback=callbacks,
            tb_log_name=run_name,
            reset_num_timesteps=not resuming,
        )
    except KeyboardInterrupt:
        interrupted = True
        print(f"\nTraining interrupted (Ctrl+C) after {model.num_timesteps:,} steps")
    elapsed = time.time() - started
    trained_steps = model.num_timesteps - start_steps

    if interrupted:
        final_path = MODELS_DIR / f"{run_name}_interrupted.zip"
        model.save(str(final_path))
        print(f"  interrupted model saved: {final_path}")
        print(f"  resume with: python -m training.train "
              f"--resume {final_path} --timesteps <additional-steps>")
        print(f"  or target a total:  --resume {final_path} "
              f"--until-total <absolute-target>")
    else:
        final_path = MODELS_DIR / f"{run_name}_final.zip"
        model.save(str(final_path))
    vec_env.close()

    print(f"\nTraining {'interrupted' if interrupted else 'finished'}")
    print(f"  wall time:      {format_hms(elapsed)} ({elapsed / 60:.1f} min)")
    print(f"  steps this run: {trained_steps:,} (total: {model.num_timesteps:,})")
    print(f"  throughput:     {trained_steps / max(elapsed, 1e-9):,.0f} steps/sec")
    print(f"  model saved:    {final_path}")
    print(f"  best model:     {best_dir / 'best_model.zip'}")
    print(f"  checkpoints:    {checkpoint_dir}")
    print(f"  training log:   {RESULTS_DIR / f'training_log_{run_name}.csv'}")
    print(f"  metrics log:    {RESULTS_DIR / f'metrics_{run_name}.csv'}")
    print(f"  tensorboard:    python -m tensorboard.main --logdir results/tensorboard")
    return final_path, run_name


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO Block Blast agent.")
    add_cli_args(parser)
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    train(config_from_args(parse_args(argv)))


if __name__ == "__main__":
    main()  # guard required: SubprocVecEnv spawns processes on Windows
