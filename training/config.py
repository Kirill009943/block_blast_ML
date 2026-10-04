"""Central training configuration.

Every PPO hyperparameter and run option lives in :class:`TrainConfig` so
experiments are declared in one place, passed to ``training.train.train``
programmatically, and serialized to ``config.json`` for reproducibility.
CLI flags in ``train.py`` map 1:1 onto these fields.

Hyperparameter semantics
------------------------
PPO fields are ``Optional``; ``None`` means "auto":

* fresh run / ``--init-from``  -> the built-in defaults below are used;
* ``--resume``                 -> the values stored in the checkpoint are
  kept, except the three safely-adjustable knobs (``learning_rate``,
  ``ent_coef``, ``clip_range``) which are applied to the loaded model when
  given explicitly.

Timestep semantics
------------------
``total_timesteps`` (``--timesteps``) always means **steps trained in this
session** — matching SB3's ``model.learn(total_timesteps=...)``. When
resuming a 26M-step model, ``--timesteps 9000000`` trains 9M MORE steps.
Use ``--until-total`` (``until_total``) to target an absolute total:
``--until-total 35000000`` trains exactly ``35M - model.num_timesteps``
additional steps (and refuses to run if the model is already past it).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, Optional

from environment.observations import OBSERVATION_CHANNELS
from environment.rewards import REWARD_PROFILES

# Built-in PPO defaults (= the proven 5M baseline configuration behind
# models/main_final.zip). Used whenever a field is None and no checkpoint
# supplies a value.
PPO_DEFAULTS: Dict[str, Any] = dict(
    learning_rate=3e-4,
    n_steps=512,
    batch_size=1024,
    n_epochs=4,
    gamma=0.995,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
)

# Hyperparameters that can safely change on a resumed model (they are
# re-read every PPO update). Everything else is structural (rollout-buffer
# shape, gamma baked into the buffer, ...) and comes from the checkpoint.
RESUMABLE_OVERRIDES = ("learning_rate", "ent_coef", "clip_range")


@dataclass
class TrainConfig:
    """All knobs for one training run. None PPO fields mean "auto"."""

    # run setup
    run_name: Optional[str] = None
    total_timesteps: int = 10_000_000  # steps trained in THIS session
    until_total: Optional[int] = None  # absolute total target; overrides --timesteps
    n_envs: int = 8
    seed: int = 0
    device: str = "auto"
    checkpoint_freq: int = 100_000
    eval_freq: int = 50_000
    eval_episodes: int = 20
    progress_interval: float = 30.0  # seconds between [timer] ETA lines; 0 = off
    resume: Optional[str] = None
    init_from: Optional[str] = None  # e.g. behavior-cloned policy weights
    reward_profile: str = "baseline"
    observation_profile: str = "basic"
    reward_scale: float = 1.0  # multiplies the reward the optimizer sees
    share_features_extractor: bool = True  # False = separate pi/vf CNNs

    # PPO hyperparameters (None = auto: defaults, or checkpoint on resume)
    learning_rate: Optional[float] = None
    n_steps: Optional[int] = None
    batch_size: Optional[int] = None
    n_epochs: Optional[int] = None
    gamma: Optional[float] = None
    gae_lambda: Optional[float] = None
    clip_range: Optional[float] = None
    ent_coef: Optional[float] = None
    vf_coef: Optional[float] = None
    max_grad_norm: Optional[float] = None

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
                     help="steps trained in THIS session; when resuming this "
                          "is ADDITIONAL steps (default: 10,000,000)")
    run.add_argument("--until-total", type=int, default=None,
                     help="absolute total-timestep target; overrides "
                          "--timesteps (trains until_total - model steps)")
    run.add_argument("--n-envs", type=int, default=8)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    run.add_argument("--checkpoint-freq", type=int, default=100_000)
    run.add_argument("--eval-freq", type=int, default=50_000)
    run.add_argument("--eval-episodes", type=int, default=20)
    run.add_argument("--progress-interval", type=float, default=30.0,
                     help="seconds between [timer] progress/ETA lines (0 = off)")
    run.add_argument("--resume", type=str, default=None,
                     help="checkpoint to continue training from; keeps the "
                          "checkpoint's PPO state (optimizer, timestep "
                          "counter, hyperparameters). --lr/--ent-coef/"
                          "--clip-range may override; other PPO flags are ignored.")
    run.add_argument("--init-from", type=str, default=None,
                     help="model whose POLICY WEIGHTS initialize a fresh run "
                          "(no optimizer/PPO state is copied)")
    run.add_argument("--reward-profile", type=str, default="baseline",
                     choices=sorted(REWARD_PROFILES))
    run.add_argument("--observation-profile", type=str, default="basic",
                     choices=sorted(OBSERVATION_CHANNELS))
    run.add_argument("--reward-scale", type=float, default=1.0,
                     help="multiply the reward the optimizer sees (component "
                          "logs stay in raw units); e.g. 0.05 when value_loss "
                          "is in the thousands")
    run.add_argument("--separate-value-net", action="store_true",
                     help="give policy and value their own CNN (prevents a "
                          "large value gradient from starving the policy "
                          "through the shared trunk)")
    run.add_argument("--run-name", type=str, default=None)

    ppo = parser.add_argument_group(
        "ppo",
        description="None of these has a CLI default: omitted means the "
                    "built-in default for fresh runs, or the checkpoint "
                    "value for --resume (only lr/ent-coef/clip-range may "
                    "override a resumed checkpoint).",
    )
    ppo.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float, default=None)
    ppo.add_argument("--n-steps", type=int, default=None)
    ppo.add_argument("--batch-size", type=int, default=None)
    ppo.add_argument("--n-epochs", type=int, default=None)
    ppo.add_argument("--gamma", type=float, default=None)
    ppo.add_argument("--gae-lambda", type=float, default=None)
    ppo.add_argument("--clip-range", type=float, default=None)
    ppo.add_argument("--ent-coef", type=float, default=None)
    ppo.add_argument("--vf-coef", type=float, default=None)
    ppo.add_argument("--max-grad-norm", type=float, default=None)
    return parser


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        run_name=args.run_name,
        total_timesteps=args.timesteps,
        until_total=args.until_total,
        n_envs=args.n_envs,
        seed=args.seed,
        device=args.device,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        progress_interval=args.progress_interval,
        resume=args.resume,
        init_from=args.init_from,
        reward_profile=args.reward_profile,
        observation_profile=args.observation_profile,
        reward_scale=args.reward_scale,
        share_features_extractor=not args.separate_value_net,
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
