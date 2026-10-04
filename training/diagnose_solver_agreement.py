"""Compare a learned model against the solver on identical states.

Controlled agreement:
    state_t -> solver action and model distribution are recorded
    state_t -> solver action is applied
    state_(t+1) repeats

This separates "cannot imitate the solver on solver states" from
"matches solver states but drifts after its own mistakes". Use
``--autonomous`` to also evaluate the model on its own trajectories.

Examples:
    python -m training.diagnose_solver_agreement --model rl:models/main_final.zip
    python -m training.diagnose_solver_agreement --model imitation:models/imitation/bc_all.pt --games 20 --trace-out results/agreement.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.solver_agent import SolverAgent
from environment.block_blast_env import decode_action, encode_action
from game.game import Action, BlockBlastGame
from training.evaluate import build_agent, summarize

SCORE_BUCKETS: Tuple[Tuple[str, int, Optional[int]], ...] = (
    ("0-100", 0, 100),
    ("100-200", 100, 200),
    ("200-500", 200, 500),
    ("500+", 500, None),
)


def score_bucket(score: int) -> str:
    for label, low, high in SCORE_BUCKETS:
        if score >= low and (high is None or score < high):
            return label
    return SCORE_BUCKETS[0][0]


def serializable_pieces(game: BlockBlastGame) -> List[Optional[Dict[str, Any]]]:
    pieces = []
    for piece in game.pieces:
        if piece is None:
            pieces.append(None)
        else:
            pieces.append({
                "type_id": piece.type_id,
                "name": piece.name,
                "cells": [list(cell) for cell in piece.cells],
                "color": piece.color,
            })
    return pieces


def model_action_probabilities(agent, game: BlockBlastGame) -> np.ndarray:
    if not hasattr(agent, "action_probabilities"):
        raise TypeError(
            "The diagnostic requires a learned agent with action_probabilities(), "
            "for example rl:<zip> or imitation:<pt>."
        )
    result = agent.action_probabilities(game)
    probs = result[0]
    return np.asarray(probs, dtype=np.float64)


def top_actions(probs: np.ndarray, k: int = 5) -> List[Tuple[int, float, Action]]:
    finite = np.isfinite(probs)
    candidates = np.flatnonzero(finite & (probs > 0))
    order = candidates[np.argsort(probs[candidates])[::-1]][:k]
    return [(int(a), float(probs[a]), decode_action(int(a))) for a in order]


def controlled_agreement(
    model_spec: str,
    games: int,
    base_seed: int,
    max_moves: int,
    beam_width: int,
    keep_trace: bool,
) -> Dict[str, Any]:
    agent = build_agent(model_spec, seed=base_seed)
    totals = {
        label: {"n": 0, "top1": 0, "top3": 0, "top5": 0, "prob_sum": 0.0}
        for label, _, _ in SCORE_BUCKETS
    }
    traces: List[Dict[str, Any]] = []

    for i in range(games):
        game = BlockBlastGame(seed=base_seed + i)
        solver = SolverAgent(beam_width=beam_width)
        game_trace: List[Dict[str, Any]] = []

        while not game.is_game_over() and game.moves < max_moves:
            solver_action = solver.act(game)
            solver_id = encode_action(*solver_action)
            probs = model_action_probabilities(agent, game)
            top5 = top_actions(probs, 5)
            top_ids = [action_id for action_id, _, _ in top5]

            bucket = score_bucket(game.score)
            entry = totals[bucket]
            entry["n"] += 1
            entry["top1"] += int(top_ids[:1] == [solver_id])
            entry["top3"] += int(solver_id in top_ids[:3])
            entry["top5"] += int(solver_id in top_ids[:5])
            entry["prob_sum"] += float(probs[solver_id])

            if keep_trace:
                game_trace.append({
                    "step": game.moves,
                    "score": game.score,
                    "board": game.board.tolist(),
                    "pieces": serializable_pieces(game),
                    "legal_actions": len(game.get_valid_actions()),
                    "solver_action": list(solver_action),
                    "model_top5": [
                        {
                            "rank": rank,
                            "action": list(action),
                            "probability": probability,
                        }
                        for rank, (_, probability, action) in enumerate(top5, start=1)
                    ],
                    "solver_in_top1": top_ids[:1] == [solver_id],
                    "solver_in_top3": solver_id in top_ids[:3],
                    "solver_in_top5": solver_id in top_ids[:5],
                    "solver_probability": float(probs[solver_id]),
                })

            result = game.place_piece(*solver_action)
            if not result.success:
                raise RuntimeError(f"Solver produced invalid action {solver_action}")

        if keep_trace:
            traces.append({
                "seed": base_seed + i,
                "final_score": game.score,
                "moves": game.moves,
                "steps": game_trace,
            })

    by_bucket = {}
    for label, entry in totals.items():
        n = max(entry["n"], 1)
        by_bucket[label] = {
            "samples": entry["n"],
            "top1": entry["top1"] / n,
            "top3": entry["top3"] / n,
            "top5": entry["top5"] / n,
            "mean_solver_action_probability": entry["prob_sum"] / n,
        }
    overall_n = sum(entry["n"] for entry in totals.values())
    overall = {
        "samples": overall_n,
        "top1": sum(e["top1"] for e in totals.values()) / max(overall_n, 1),
        "top3": sum(e["top3"] for e in totals.values()) / max(overall_n, 1),
        "top5": sum(e["top5"] for e in totals.values()) / max(overall_n, 1),
        "mean_solver_action_probability": (
            sum(e["prob_sum"] for e in totals.values()) / max(overall_n, 1)
        ),
    }
    return {"overall": overall, "by_score": by_bucket, "traces": traces}


def autonomous_stats(
    model_spec: str,
    games: int,
    base_seed: int,
    max_moves: int,
) -> Dict[str, Any]:
    scores: List[int] = []
    moves: List[int] = []
    lines: List[int] = []
    game_over = 0
    for i in range(games):
        game = BlockBlastGame(seed=base_seed + i)
        agent = build_agent(model_spec, seed=base_seed + i)
        while not game.is_game_over() and game.moves < max_moves:
            result = game.place_piece(*agent.act(game))
            if not result.success:
                raise RuntimeError(f"{model_spec} produced an invalid action")
        scores.append(game.score)
        moves.append(game.moves)
        lines.append(game.total_lines_cleared)
        game_over += int(game.is_game_over())
    stats = summarize(scores, moves, lines)
    stats["game_over_rate"] = game_over / max(games, 1)
    return stats


def print_agreement(report: Dict[str, Any]) -> None:
    print("\nSolver/model agreement on solver-controlled states")
    print("--------------------------------------------------")
    overall = report["overall"]
    print(f"samples: {overall['samples']:,}")
    print(f"top-1:   {overall['top1']:.3f}")
    print(f"top-3:   {overall['top3']:.3f}")
    print(f"top-5:   {overall['top5']:.3f}")
    print(f"p(solver action): {overall['mean_solver_action_probability']:.4f}")
    print("\nBy score range:")
    print(f"  {'range':<10}{'n':>8}{'top1':>8}{'top3':>8}{'top5':>8}{'p_solver':>12}")
    for label, stats in report["by_score"].items():
        print(f"  {label:<10}{stats['samples']:>8,}{stats['top1']:>8.3f}"
              f"{stats['top3']:>8.3f}{stats['top5']:>8.3f}"
              f"{stats['mean_solver_action_probability']:>12.4f}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a learned model against SolverAgent on identical states."
    )
    parser.add_argument("--model", required=True,
                        help="rl:<zip> or imitation:<pt>")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-moves", type=int, default=500)
    parser.add_argument("--beam-width", type=int, default=32)
    parser.add_argument("--trace-out", type=str, default=None,
                        help="optional JSON path for per-step traces")
    parser.add_argument("--autonomous", action="store_true",
                        help="also evaluate the model on its own trajectories")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    report = controlled_agreement(
        model_spec=args.model,
        games=args.games,
        base_seed=args.seed,
        max_moves=args.max_moves,
        beam_width=args.beam_width,
        keep_trace=bool(args.trace_out),
    )
    print_agreement(report)

    if args.autonomous:
        stats = autonomous_stats(args.model, args.games, args.seed, args.max_moves)
        report["autonomous"] = stats
        print("\nAutonomous model play on the same seeds")
        print("---------------------------------------")
        print(f"mean score:       {stats['avg_score']:.1f}")
        print(f"median score:     {stats['median_score']:.0f}")
        print(f"max score:        {stats['best_score']}")
        print(f"std score:        {stats['std_score']:.1f}")
        print(f"mean lines:       {stats['avg_lines_cleared']:.1f}")
        print(f"mean length:      {stats['avg_moves']:.1f}")
        print(f"game-over rate:   {stats['game_over_rate']:.3f}")

    if args.trace_out:
        out = Path(args.trace_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
        print(f"\nTrace written: {out}")
    return report


if __name__ == "__main__":
    main()
