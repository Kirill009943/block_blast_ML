"""Controlled experiment runner.

Runs a small, fixed set of meaningful experiments sequentially — each an
isolated training run with its own artifact directory — then evaluates
every model against the SAME seed set as the permanent baselines and
writes a summary.

    python -m training.experiments --timesteps 2000000
    python -m training.experiments --timesteps 5000000 --eval-games 200

Per experiment, ``results/experiments/<name>/`` contains:

* ``config.json``        — full TrainConfig + reproducibility metadata
* ``model_final.zip``    — trained model
* ``training_log.csv``   — per-episode metrics
* ``metrics.csv``        — per-update PPO stats
* ``tensorboard/``       — TB event files
* ``eval.json``          — evaluation on identical seeds
* ``learning_curves.png``

Plus ``results/experiments/summary.csv`` and ``summary.png`` comparing all
experiments and the Random/Heuristic baselines.

Scientific rule: an experiment only "worked" if its EVALUATION score on
identical seeds beats the baseline experiment's — training reward alone
proves nothing (different reward profiles are not comparable by reward).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

from training.config import TrainConfig
from training.repro import write_config_json

EXPERIMENTS_DIR = Path("results") / "experiments"

# The experiment matrix. Each entry overrides TrainConfig fields.
# Keep this small and meaningful — do not add dozens of runs.
EXPERIMENTS: List[Dict] = [
    {"name": "A_baseline", "overrides": {},
     "hypothesis": "reference point — the proven 5M configuration"},
    {"name": "B_entropy_0.03", "overrides": {"ent_coef": 0.03},
     "hypothesis": "more exploration escapes the place-anywhere local optimum"},
    {"name": "C_lr_1e-4", "overrides": {"learning_rate": 1e-4},
     "hypothesis": "lower lr stabilizes the value function (expl.var was ~0.1)"},
    {"name": "D_reward_strategic", "overrides": {"reward_profile": "strategic"},
     "hypothesis": "board-quality shaping (holes/fragmentation/future moves) "
                   "improves real score"},
    {"name": "E_obs_enhanced", "overrides": {"observation_profile": "enhanced"},
     "hypothesis": "height/holes/line-potential/placement-density channels "
                   "make line clears easier to learn"},
    {"name": "F_bc_init", "overrides": {"init_from": "models/bc_init.zip"},
     "hypothesis": "behavior-cloned policy weights bootstrap PPO past the "
                   "weak early phase"},
]


def run_experiment(name: str, overrides: Dict, base: TrainConfig,
                   eval_games: int, eval_seed: int) -> Dict:
    """Train + evaluate one experiment; save artifacts; return eval stats."""
    from training.evaluate import evaluate_agent
    from training.plot_results import plot
    from training.train import RESULTS_DIR, train

    exp_dir = EXPERIMENTS_DIR / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(**base.to_dict())
    for key, value in overrides.items():
        setattr(config, key, value)
    config.run_name = f"exp_{name}"

    write_config_json(exp_dir / "config.json",
                      {"experiment": name, "hypothesis": overrides.get("hypothesis", ""),
                       "overrides": {k: v for k, v in overrides.items()
                                     if k != "hypothesis"},
                       **config.to_dict()})

    print(f"\n{'=' * 70}\nExperiment {name}: {overrides.get('hypothesis', '')}\n{'=' * 70}")
    started = time.time()
    final_path, run_name = train(config)
    train_minutes = (time.time() - started) / 60

    stats = evaluate_agent(f"rl:{final_path}", eval_games, base_seed=eval_seed)
    stats["train_minutes"] = round(train_minutes, 1)

    # collect artifacts
    shutil.copy(final_path, exp_dir / "model_final.zip")
    for src_name, dst_name in [
        (f"training_log_{run_name}.csv", "training_log.csv"),
        (f"metrics_{run_name}.csv", "metrics.csv"),
    ]:
        src = RESULTS_DIR / src_name
        if src.exists():
            shutil.copy(src, exp_dir / dst_name)
    tb_src = RESULTS_DIR / "tensorboard" / run_name
    if tb_src.exists():
        shutil.copytree(tb_src, exp_dir / "tensorboard", dirs_exist_ok=True)
    with (exp_dir / "eval.json").open("w") as f:
        json.dump(stats, f, indent=2)
    try:
        plot(exp_dir / "training_log.csv", 200, exp_dir / "learning_curves.png")
    except Exception as exc:
        print(f"  (learning-curve plot failed: {exc})")

    print(f"Experiment {name}: avg eval score {stats['avg_score']:.1f} "
          f"(median {stats['median_score']:.0f}) — artifacts in {exp_dir}")
    return stats


def summarize(rows: List[Dict]) -> None:
    """Write summary.csv + summary.png across experiments and baselines."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["name", "avg_score", "median_score", "std_score", "best_score",
               "worst_score", "p25_score", "p75_score", "avg_moves",
               "avg_lines_cleared", "train_minutes"]
    with (EXPERIMENTS_DIR / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row.get(c, "") for c in columns])
    print(f"\nSaved {EXPERIMENTS_DIR / 'summary.csv'}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [r["name"] for r in rows]
    means = [r["avg_score"] for r in rows]
    stds = [r["std_score"] for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 5))
    colors = ["#6b7280" if r.get("baseline") else "#1e3a8a" for r in rows]
    ax.bar(labels, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    ax.set_ylabel("avg evaluation score (identical seeds)")
    ax.set_title("Experiment comparison (+/- 1 std)")
    ax.tick_params(axis="x", rotation=25)
    for i, m in enumerate(means):
        ax.text(i, m, f"{m:.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(EXPERIMENTS_DIR / "summary.png", dpi=140)
    plt.close(fig)
    print(f"Saved {EXPERIMENTS_DIR / 'summary.png'}")


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled PPO experiments.")
    parser.add_argument("--timesteps", type=int, default=2_000_000,
                        help="training steps per experiment")
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=1000,
                        help="same base seed for every agent/experiment")
    parser.add_argument("--only", type=str, nargs="*", default=None,
                        help="run only these experiment names")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    base = TrainConfig(total_timesteps=args.timesteps, n_envs=args.n_envs,
                       seed=args.seed)

    rows: List[Dict] = []

    # permanent baselines on the same seed set
    from training.evaluate import evaluate_agent

    for spec in ("random", "heuristic"):
        stats = evaluate_agent(spec, args.eval_games, base_seed=args.eval_seed,
                               progress=False)
        rows.append({"name": f"baseline_{spec}", "baseline": True, **stats})
        print(f"baseline {spec}: avg score {stats['avg_score']:.1f}")

    selected = EXPERIMENTS
    if args.only:
        wanted = set(args.only)
        selected = [e for e in EXPERIMENTS if e["name"] in wanted]
        missing = wanted - {e["name"] for e in selected}
        if missing:
            raise SystemExit(f"Unknown experiments: {sorted(missing)}")

    for exp in selected:
        init_from = exp["overrides"].get("init_from")
        if init_from and not Path(init_from).exists():
            print(f"\nSkipping {exp['name']}: {init_from} not found. Generate it with:")
            print("  python -m training.generate_expert_data --games 1000")
            print("  python -m training.behavior_cloning --epochs 10")
            continue
        stats = run_experiment(exp["name"], exp["overrides"], base,
                               args.eval_games, args.eval_seed)
        rows.append({"name": exp["name"], "baseline": False, **stats})

    summarize(rows)

    baseline_score = next(r["avg_score"] for r in rows if r["name"] == "A_baseline") \
        if any(r["name"] == "A_baseline" for r in rows) else None
    print("\nResults (avg evaluation score, identical seeds):")
    for row in rows:
        verdict = ""
        if baseline_score and not row.get("baseline"):
            diff = row["avg_score"] - baseline_score
            verdict = f"  ({'+' if diff >= 0 else ''}{diff:.1f} vs A_baseline)"
        print(f"  {row['name']:24s} {row['avg_score']:8.1f}{verdict}")


if __name__ == "__main__":
    main()
