"""PPO training for the Block Blast RL agent.

Uses MaskablePPO from sb3-contrib so the policy only ever samples legal
(piece, row, col) placements — the action mask comes from the environment.

Examples:
    python -m training.train                              # full run, 10M steps
    python -m training.train --timesteps 50000            # quick smoke run
    python -m training.train --timesteps 5000000 --ent-coef 0.03 --gamma 0.999
    python -m training.train --reward-profile strategic --observation-profile enhanced
    python -m training.train --resume models/checkpoints/ppo_block_blast_250000_steps.zip

Outputs (for run name ``<run>``):
    models/<run>_final.zip            final model
    models/best/best_model.zip        best model by masked eval (latest run)
    models/checkpoints/               periodic checkpoints
    results/training_log_<run>.csv    per-episode metrics (dashboard input)
    results/metrics_<run>.csv         per-update PPO optimizer stats
    results/tensorboard/<run>/        tensorboard logs
    results/experiments or results/   config.json with full reproducibility info
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from environment.block_blast_env import BlockBlastEnv
from environment.rewards import get_reward_config
from training.cnn_extractor import BlockBlastCNN
from training.config import TrainConfig, add_cli_args, config_from_args
from training.metrics import EpisodeMetricsCallback, TrainMetricsCallback
from training.repro import write_config_json

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")


def mask_fn(env: BlockBlastEnv) -> np.ndarray:
    return env.action_masks()


def make_env(config: TrainConfig, rank: int):
    """Factory for one vectorized environment instance (headless)."""

    def _init():
        env = BlockBlastEnv(
            reward_profile=config.reward_profile,
            observation_profile=config.observation_profile,
        )
        env = ActionMasker(env, mask_fn)
        env.reset(seed=config.seed + rank)
        return env

    return _init


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

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    checkpoint_dir = MODELS_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    config.run_name = run_name
    write_config_json(RESULTS_DIR / f"config_{run_name}.json", config.to_dict())

    print(f"Run: {run_name}")
    print(f"Timesteps: {config.total_timesteps:,} | envs: {config.n_envs} | "
          f"seed: {config.seed}")
    print(f"Reward profile: {config.reward_profile} | "
          f"Observation profile: {config.observation_profile}")
    print(f"PPO: lr={config.learning_rate} gamma={config.gamma} "
          f"gae_lambda={config.gae_lambda} ent_coef={config.ent_coef} "
          f"clip={config.clip_range} n_steps={config.n_steps} "
          f"batch={config.batch_size} epochs={config.n_epochs} "
          f"vf_coef={config.vf_coef} max_grad_norm={config.max_grad_norm}")

    vec_env = SubprocVecEnv([make_env(config, i) for i in range(config.n_envs)])
    vec_env = VecMonitor(vec_env)

    policy_kwargs = dict(
        features_extractor_class=BlockBlastCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    if config.resume:
        print(f"Resuming from: {config.resume}")
        model = MaskablePPO.load(config.resume, env=vec_env, device=device)
    elif config.init_from:
        print(f"Initializing weights from: {config.init_from}")
        model = MaskablePPO(
            "MlpPolicy", vec_env, policy_kwargs=policy_kwargs,
            learning_rate=config.learning_rate, n_steps=config.n_steps,
            batch_size=config.batch_size, n_epochs=config.n_epochs,
            gamma=config.gamma, gae_lambda=config.gae_lambda,
            clip_range=config.clip_range, ent_coef=config.ent_coef,
            vf_coef=config.vf_coef, max_grad_norm=config.max_grad_norm,
            seed=config.seed, device=device,
            tensorboard_log=str(RESULTS_DIR / "tensorboard"), verbose=1,
        )
        source = MaskablePPO.load(config.init_from, device=device)
        model.policy.load_state_dict(source.policy.state_dict())
    else:
        model = MaskablePPO(
            "MlpPolicy", vec_env, policy_kwargs=policy_kwargs,
            learning_rate=config.learning_rate, n_steps=config.n_steps,
            batch_size=config.batch_size, n_epochs=config.n_epochs,
            gamma=config.gamma, gae_lambda=config.gae_lambda,
            clip_range=config.clip_range, ent_coef=config.ent_coef,
            vf_coef=config.vf_coef, max_grad_norm=config.max_grad_norm,
            seed=config.seed, device=device,
            tensorboard_log=str(RESULTS_DIR / "tensorboard"), verbose=1,
        )

    eval_env = BlockBlastEnv(
        reward_profile=config.reward_profile,
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
            best_model_save_path=str(MODELS_DIR / "best"),
            log_path=str(RESULTS_DIR / "eval" / run_name),
            eval_freq=max(config.eval_freq // config.n_envs, 1),
            n_eval_episodes=config.eval_episodes,
            deterministic=True,
            use_masking=True,
        ),
        EpisodeMetricsCallback(RESULTS_DIR / f"training_log_{run_name}.csv"),
        TrainMetricsCallback(RESULTS_DIR / f"metrics_{run_name}.csv"),
    ]

    started = time.time()
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        tb_log_name=run_name,
        reset_num_timesteps=config.resume is None,
    )
    elapsed = time.time() - started

    final_path = MODELS_DIR / f"{run_name}_final.zip"
    model.save(str(final_path))
    vec_env.close()

    print("\nTraining finished")
    print(f"  wall time:      {elapsed / 60:.1f} min")
    print(f"  throughput:     {config.total_timesteps / max(elapsed, 1e-9):,.0f} steps/sec")
    print(f"  final model:    {final_path}")
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
