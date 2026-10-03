"""Optional hyperparameter optimization with Optuna.

Optuna is an OPTIONAL dependency — the rest of the project does not need
it. Install with ``pip install optuna``.

    python -m training.optimize --trials 20 --timesteps 1000000

Each trial trains a short PPO run with sampled hyperparameters and scores
it by *deterministic masked evaluation score* (not training reward) on a
fixed seed set. Short-run winners are not guaranteed to win at long runs —
take the printed best config and launch a full run yourself, e.g.:

    python -m training.train --timesteps 20000000 --lr ... --gamma ... ...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

from training.config import TrainConfig

RESULTS_DIR = Path("results") / "optuna"


def suggest_config(trial, base: TrainConfig) -> TrainConfig:
    """Sample a TrainConfig around the important PPO knobs."""
    config = TrainConfig(**base.to_dict())
    config.learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    config.gamma = trial.suggest_categorical("gamma", [0.99, 0.995, 0.999])
    config.gae_lambda = trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.98])
    config.ent_coef = trial.suggest_float("ent_coef", 1e-3, 5e-2, log=True)
    config.clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3])
    config.n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024])
    config.batch_size = trial.suggest_categorical("batch_size", [512, 1024, 2048])
    return config


def objective_factory(base: TrainConfig, eval_games: int):
    def objective(trial) -> float:
        from training.evaluate import evaluate_agent
        from training.train import train

        config = suggest_config(trial, base)
        config.run_name = f"optuna_t{trial.number:03d}"
        config.eval_freq = max(config.total_timesteps + 1, 1)  # skip mid-run evals
        config.checkpoint_freq = max(config.total_timesteps + 1, 1)
        final_path, _ = train(config)
        stats = evaluate_agent(f"rl:{final_path}", eval_games, base_seed=1000,
                               progress=False)
        trial.set_user_attr("avg_score", stats["avg_score"])
        trial.set_user_attr("model", str(final_path))
        return stats["avg_score"]

    return objective


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description="Optuna HPO for Block Blast PPO.")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="SHORT training length per trial")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-games", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--study-name", type=str,
                        default=f"bb_{time.strftime('%Y%m%d_%H%M%S')}")
    args = parser.parse_args(argv)

    try:
        import optuna
    except ImportError:
        raise SystemExit(
            "Optuna is an optional dependency. Install it with: pip install optuna"
        )

    base = TrainConfig(total_timesteps=args.timesteps, n_envs=args.n_envs,
                       seed=args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{RESULTS_DIR / 'optuna.db'}"
    study = optuna.create_study(direction="maximize", study_name=args.study_name,
                                storage=storage, load_if_exists=True)
    study.optimize(objective_factory(base, args.eval_games), n_trials=args.trials)

    best = study.best_trial
    print("\nBest trial:")
    print(f"  avg eval score: {best.value:.1f}")
    for key, value in best.params.items():
        print(f"  {key}: {value}")
    print(f"  model: {best.user_attrs.get('model')}")

    out = RESULTS_DIR / f"best_config_{args.study_name}.json"
    best_config = suggest_config(best, base).to_dict()
    best_config["short_run_avg_score"] = best.value
    with out.open("w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\nSaved best config: {out}")
    print("NOTE: this won a SHORT run — verify with a long training run:")
    print(f"  python -m training.train --timesteps 20000000 "
          f"--lr {best.params['learning_rate']:.2e} --gamma {best.params['gamma']} "
          f"--gae-lambda {best.params['gae_lambda']} --ent-coef {best.params['ent_coef']:.4f} "
          f"--clip-range {best.params['clip_range']} --n-steps {best.params['n_steps']} "
          f"--batch-size {best.params['batch_size']}")


if __name__ == "__main__":
    main()
