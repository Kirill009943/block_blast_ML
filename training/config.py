"""Central training configuration.

Every PPO hyperparameter and run option lives in :class:`TrainConfig` so
experiments are declared in one place, passed to ``training.train.train``
programmatically, and serialized to ``config.json`` for reproducibility.
CLI flags in ``train.py`` map 1:1 onto these fields.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional

from environment.rewards import REWARD_PROFILES


@dataclass
class TrainConfig:
    """All knobs for one training run. Defaults = the proven 5M baseline."""

    # run setup
    run_name: Optional[str] = None
    total_timesteps: int = 10_000_000
    n_envs: int = 8
    seed: int = 0
    device: str = "auto"
    checkpoint_freq: int = 100_000
    eval_freq: int = 50_000
    eval_episodes: int = 20
    resume: Optional[str] = None
    init_from: Optional[str] = None  # e.g. behavior-cloned policy weights
    reward_profile: str = "baseline"
    observation_profile: str = "basic"

    # PPO hyperparameters (baseline = the configuration behind models/main_final.zip)
    learning_rate: float = 3e-4
    n_steps: int = 512
    batch_size: int = 1024
    n_epochs: int = 4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register every TrainConfig field as a CLI flag."""
    run = parser.add_argument_group("run")
    run.add_argument("--timesteps", type=int, default=10_000_000,
                     help="total training timesteps (default: 10,000,000)")
    run.add_argument("--n-envs", type=int, default=8)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    run.add_argument("--checkpoint-freq", type=int, default=100_000)
    run.add_argument("--eval-freq", type=int, default=50_000)
    run.add_argument("--eval-episodes", type=int, default=20)
    run.add_argument("--resume", type=str, default=None,
                     help="checkpoint to continue training from")
    run.add_argument("--init-from", type=str, default=None,
                     help="model whose weights initialize the new run (e.g. behavior cloning)")
    run.add_argument("--reward-profile", type=str, default="baseline",
                     choices=sorted(REWARD_PROFILES))
    run.add_argument("--observation-profile", type=str, default="basic",
                     choices=["basic", "enhanced"])
    run.add_argument("--run-name", type=str, default=None)

    ppo = parser.add_argument_group("ppo")
    ppo.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float, default=3e-4)
    ppo.add_argument("--n-steps", type=int, default=512)
    ppo.add_argument("--batch-size", type=int, default=1024)
    ppo.add_argument("--n-epochs", type=int, default=4)
    ppo.add_argument("--gamma", type=float, default=0.995)
    ppo.add_argument("--gae-lambda", type=float, default=0.95)
    ppo.add_argument("--clip-range", type=float, default=0.2)
    ppo.add_argument("--ent-coef", type=float, default=0.01)
    ppo.add_argument("--vf-coef", type=float, default=0.5)
    ppo.add_argument("--max-grad-norm", type=float, default=0.5)
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        run_name=args.run_name,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        device=args.device,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        resume=args.resume,
        init_from=args.init_from,
        reward_profile=args.reward_profile,
        observation_profile=args.observation_profile,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
    )
