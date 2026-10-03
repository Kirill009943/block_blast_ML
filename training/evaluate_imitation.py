"""Evaluate an imitation-learning agent against the usual benchmarks.

    python -m training.evaluate_imitation --model models/imitation/bc_all.pt
    python -m training.evaluate_imitation --model models/imitation/bc_all.pt \
        --games 200 --compare-with rl:models/main_final.zip

Always compares on identical seed sets against RandomAgent and
HeuristicAgent (plus any extra specs passed via --compare-with, e.g. the
PPO model). Gameplay score is the metric that matters — high training
accuracy does not imply strong play.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from training.evaluate import evaluate_agent


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a behavioral-cloning agent.")
    parser.add_argument("--model", type=str, required=True,
                        help="path to the .pt imitation checkpoint")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000,
                        help="base seed; identical for every agent")
    parser.add_argument("--compare-with", type=str, nargs="*", default=[],
                        help="extra agent specs, e.g. rl:models/main_final.zip")
    parser.add_argument("--max-moves", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true",
                        help="also evaluate the stochastic (sampling) variant")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> List[dict]:
    args = parse_args(argv)
    specs = ["random", "heuristic", f"imitation:{args.model}", *args.compare_with]

    all_stats = []
    for spec in specs:
        stats = evaluate_agent(spec, args.games, base_seed=args.seed,
                               max_moves=args.max_moves)
        all_stats.append(stats)

    if args.stochastic:
        from agents.imitation_agent import ImitationAgent

        spec = f"imitation:{args.model}"
        # evaluate_agent rebuilds agents per game; a stochastic variant is
        # evaluated here directly on the same seed set
        from game.game import BlockBlastGame

        scores, moves_list, lines_list = [], [], []
        for i in range(args.games):
            game = BlockBlastGame(seed=args.seed + i)
            agent = ImitationAgent(args.model, deterministic=False, seed=args.seed + i)
            while not game.is_game_over():
                if args.max_moves and game.moves >= args.max_moves:
                    break
                game.place_piece(*agent.act(game))
            scores.append(game.score)
            moves_list.append(game.moves)
            lines_list.append(game.total_lines_cleared)
        from training.evaluate import summarize

        stats = summarize(scores, moves_list, lines_list)
        stats["agent"] = f"{spec} (stochastic)"
        all_stats.append(stats)

    print("\nComparison (identical seeds)")
    print("----------------------------")
    print(f"  {'agent':<44}{'avg':>9}{'median':>9}{'best':>8}{'moves':>8}{'lines':>8}")
    for stats in all_stats:
        print(f"  {stats['agent']:<44}{stats['avg_score']:>9.1f}"
              f"{stats['median_score']:>9.0f}{stats['best_score']:>8}"
              f"{stats['avg_moves']:>8.1f}{stats['avg_lines_cleared']:>8.1f}")
    return all_stats


if __name__ == "__main__":
    main()
