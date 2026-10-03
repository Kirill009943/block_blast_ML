"""Evaluate and compare Block Blast agents.

Single agent:
    python -m training.evaluate --agent random --games 200
    python -m training.evaluate --agent heuristic --games 100
    python -m training.evaluate --agent solver --games 5 --max-moves 2000
    python -m training.evaluate --agent rl:models/main_final.zip --games 200

Comparison (identical seed sets for every agent):
    python -m training.evaluate --compare random heuristic rl:models/main_final.zip --games 200

Reports mean/median/std/min/max/25th/75th percentile of the score, average
episode length and average lines cleared. Appends a row per agent to
``results/evaluation_results.csv``, writes a JSON dump per run, and in
compare mode saves ``results/evaluation_comparison.png``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from agents.base import Agent
from game import BlockBlastGame

RESULTS_DIR = Path("results")
CSV_PATH = RESULTS_DIR / "evaluation_results.csv"


def build_agent(spec: str, seed: int) -> Agent:
    """Create an agent: ``random`` | ``heuristic`` | ``solver`` | ``rl:<path>`` | ``imitation:<path>``."""
    if spec == "random":
        from agents.random_agent import RandomAgent

        return RandomAgent(seed=seed)
    if spec == "heuristic":
        from agents.heuristic_agent import HeuristicAgent

        return HeuristicAgent()
    if spec == "solver":
        from agents.solver_agent import SolverAgent

        return SolverAgent()
    if spec.startswith("rl:"):
        from agents.rl_agent import RLAgent

        return RLAgent(spec[3:])
    if spec.startswith("imitation:"):
        from agents.imitation_agent import ImitationAgent

        return ImitationAgent(spec[len("imitation:"):], seed=seed)
    raise ValueError(
        f"Unknown agent spec: {spec!r} "
        "(use random | heuristic | solver | rl:<path> | imitation:<path>)"
    )


def short_name(spec: str) -> str:
    """Compact label for plots: ``rl:models/main_final.zip`` -> ``main_final``."""
    if spec.startswith("rl:"):
        return Path(spec[3:]).stem
    return spec


def summarize(scores: List[int], moves: List[int], lines: List[int]) -> Dict:
    """Aggregate statistics for a set of finished games."""
    arr = np.asarray(scores, dtype=float)
    return {
        "games": len(scores),
        "avg_score": float(arr.mean()),
        "median_score": float(np.median(arr)),
        "std_score": float(arr.std()),
        "best_score": int(arr.max()),
        "worst_score": int(arr.min()),
        "p25_score": float(np.percentile(arr, 25)),
        "p75_score": float(np.percentile(arr, 75)),
        "avg_moves": float(np.mean(moves)),
        "avg_lines_cleared": float(np.mean(lines)),
    }


def evaluate_agent(spec: str, games: int, base_seed: int = 1000,
                   progress: bool = True, max_moves: int = 0) -> Dict:
    """Play ``games`` full games and return aggregate statistics + raw data.

    ``max_moves`` > 0 caps each game (needed for agents that effectively
    never reach game over, such as the solver).
    """
    scores: List[int] = []
    moves: List[int] = []
    lines: List[int] = []

    started = time.time()
    for i in range(games):
        game = BlockBlastGame(seed=base_seed + i)
        agent = build_agent(spec, seed=base_seed + i)
        while not game.is_game_over():
            if max_moves and game.moves >= max_moves:
                break
            game.place_piece(*agent.act(game))
        scores.append(game.score)
        moves.append(game.moves)
        lines.append(game.total_lines_cleared)
        if progress and (i + 1) % max(games // 10, 1) == 0:
            print(f"  {spec}: {i + 1}/{games} games played")

    stats = summarize(scores, moves, lines)
    stats.update({
        "agent": spec,
        "base_seed": base_seed,
        "elapsed_seconds": round(time.time() - started, 2),
        "scores": scores,
    })
    return stats


def print_report(stats: Dict) -> None:
    print("\nEvaluation")
    print("----------")
    print(f"Agent:                {stats['agent']}")
    print(f"Games:                {stats['games']}")
    print(f"Average score:        {stats['avg_score']:.1f}")
    print(f"Median score:         {stats['median_score']:.0f}")
    print(f"Std deviation:        {stats['std_score']:.1f}")
    print(f"Best score:           {stats['best_score']}")
    print(f"Worst score:          {stats['worst_score']}")
    print(f"25th percentile:      {stats['p25_score']:.0f}")
    print(f"75th percentile:      {stats['p75_score']:.0f}")
    print(f"Average moves:        {stats['avg_moves']:.1f}")
    print(f"Average lines cleared:{stats['avg_lines_cleared']: .1f}")


def save_results(stats: Dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    agent_tag = stats["agent"].replace(":", "_").replace("/", "_").replace("\\", "_")

    json_path = RESULTS_DIR / f"eval_{agent_tag}_{timestamp}.json"
    with json_path.open("w") as f:
        json.dump(stats, f, indent=2)

    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(
                ["timestamp", "agent", "games", "avg_score", "median_score",
                 "std_score", "best_score", "worst_score", "p25_score",
                 "p75_score", "avg_moves", "avg_lines_cleared"]
            )
        writer.writerow(
            [timestamp, stats["agent"], stats["games"],
             round(stats["avg_score"], 2), round(stats["median_score"], 2),
             round(stats["std_score"], 2), stats["best_score"],
             stats["worst_score"], round(stats["p25_score"], 2),
             round(stats["p75_score"], 2), round(stats["avg_moves"], 2),
             round(stats["avg_lines_cleared"], 2)]
        )
    print(f"Saved: {json_path}")
    print(f"Appended: {CSV_PATH}")


def plot_comparison(all_stats: List[Dict], out_path: Path) -> Path:
    """Box plot + bar chart comparing agents on identical seed sets."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [short_name(s["agent"]) for s in all_stats]
    scores = [s["scores"] for s in all_stats]
    means = [s["avg_score"] for s in all_stats]
    moves = [s["avg_moves"] for s in all_stats]
    lines = [s["avg_lines_cleared"] for s in all_stats]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    axes[0].boxplot(scores, labels=labels, showmeans=True)
    axes[0].set_title("score distribution (identical seeds)")
    axes[0].set_ylabel("score")
    axes[0].tick_params(axis="x", rotation=20)

    x = np.arange(len(labels))
    axes[1].bar(x, means, yerr=[s["std_score"] for s in all_stats],
                capsize=4, color="#1e3a8a", alpha=0.85)
    axes[1].set_xticks(x, labels, rotation=20)
    axes[1].set_title("mean score (+/- 1 std)")
    for xi, m in zip(x, means):
        axes[1].text(xi, m, f"{m:.0f}", ha="center", va="bottom", fontsize=9)

    width = 0.35
    axes[2].bar(x - width / 2, moves, width, label="avg moves", color="#0e7490")
    axes[2].bar(x + width / 2, lines, width, label="avg lines cleared", color="#b45309")
    axes[2].set_xticks(x, labels, rotation=20)
    axes[2].set_title("game length / line clears")
    axes[2].legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Block Blast agents.")
    parser.add_argument("--agent", type=str, default=None,
                        help="random | heuristic | solver | rl:<path-to-model.zip>")
    parser.add_argument("--compare", type=str, nargs="+", default=None,
                        help="evaluate several agents on identical seeds")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1000,
                        help="base seed; game i uses seed base+i (reproducible)")
    parser.add_argument("--max-moves", type=int, default=0,
                        help="cap moves per game (0 = no cap; needed for the solver)")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> List[Dict]:
    args = parse_args(argv)
    specs = args.compare or ([args.agent] if args.agent else None)
    if not specs:
        raise SystemExit("Pass --agent <spec> or --compare <spec> [<spec> ...]")

    all_stats = []
    for spec in specs:
        stats = evaluate_agent(spec, args.games, base_seed=args.seed,
                               progress=not args.quiet, max_moves=args.max_moves)
        print_report(stats)
        save_results(stats)
        all_stats.append(stats)

    if args.compare:
        out = plot_comparison(all_stats, RESULTS_DIR / "evaluation_comparison.png")
        print(f"\nSaved comparison plot: {out}")
    return all_stats


if __name__ == "__main__":
    main()
