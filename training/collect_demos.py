"""Autonomous demonstration collection (headless, no Pygame).

    python -m training.collect_demos --agent heuristic --games 1000
    python -m training.collect_demos --agent solver --games 50 --max-moves 5000
    python -m training.collect_demos --agent rl:models/main_final.zip --games 200

Plays full games with any agent spec known to
:func:`training.evaluate.build_agent` and records every decision into the
shared demonstration dataset (see :mod:`training.demos`). The recorder
flushes completed games to disk every ``--flush-games`` games, so memory
use stays flat no matter how many games are collected. Game ids continue
from the existing dataset, so repeated runs accumulate in one directory.

The source label is inferred from the agent spec (``rl:`` -> ``ppo``,
``imitation:`` -> ``imitation``) unless overridden with ``--source``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

from training.demos import DemoRecorder
from training.evaluate import build_agent

DEFAULT_DATASET_DIR = "results/demos"


def source_for_spec(spec: str) -> str:
    """Map an agent spec to a dataset source label."""
    head = spec.split(":", 1)[0]
    if head == "rl":
        return "ppo"
    return head


def collect(
    spec: str,
    games: int,
    dataset_dir: str = DEFAULT_DATASET_DIR,
    base_seed: int = 20_000,
    max_moves: int = 0,
    observation_profile: str = "basic",
    source: Optional[str] = None,
    flush_games: int = 25,
    progress: bool = True,
) -> dict:
    """Play ``games`` with the agent from ``spec`` and record every move."""
    source = source or source_for_spec(spec)
    totals = {"games": 0, "samples": 0, "moves": 0, "score": 0, "lines": 0}
    started = time.time()
    with DemoRecorder(
        dataset_dir,
        source=source,
        observation_profile=observation_profile,
        flush_games=flush_games,
    ) as recorder:
        for i in range(games):
            from game.game import BlockBlastGame

            game = BlockBlastGame(seed=base_seed + i)
            agent = build_agent(spec, seed=base_seed + i)
            while not game.is_game_over():
                if max_moves and game.moves >= max_moves:
                    break
                action = agent.act(game)
                recorder.begin_step(game, action)
                result = game.place_piece(*action)
                recorder.end_step(result)
            recorder.finish_game()
            totals["games"] += 1
            totals["moves"] += game.moves
            totals["score"] += game.score
            totals["lines"] += game.total_lines_cleared
            if progress and (i + 1) % max(games // 10, 1) == 0:
                elapsed = time.time() - started
                print(f"  {spec}: {i + 1}/{games} games "
                      f"({recorder.total_samples:,} samples, {elapsed:.0f}s)")
    totals["samples"] = recorder.total_samples
    totals["elapsed"] = time.time() - started
    return totals


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect demonstration games.")
    parser.add_argument("--agent", type=str, required=True,
                        help="random | heuristic | solver | rl:<path> | imitation:<path>")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--seed", type=int, default=20_000,
                        help="base seed; game i uses seed base+i")
    parser.add_argument("--max-moves", type=int, default=0,
                        help="cap moves per game (0 = no cap; use for solver)")
    parser.add_argument("--observation-profile", type=str, default="basic")
    parser.add_argument("--source", type=str, default=None,
                        help="override the source label inferred from --agent")
    parser.add_argument("--flush-games", type=int, default=25,
                        help="games buffered in RAM before writing a shard")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    print(f"Collecting {args.games} games with {args.agent!r} -> {args.dataset_dir}")
    totals = collect(
        args.agent,
        args.games,
        dataset_dir=args.dataset_dir,
        base_seed=args.seed,
        max_moves=args.max_moves,
        observation_profile=args.observation_profile,
        source=args.source,
        flush_games=args.flush_games,
    )
    games = max(totals["games"], 1)
    print(f"\nDone: {totals['games']} games, {totals['samples']:,} samples")
    print(f"  avg moves/game: {totals['moves'] / games:.1f}")
    print(f"  avg score:      {totals['score'] / games:.1f}")
    print(f"  avg lines:      {totals['lines'] / games:.1f}")
    print(f"  elapsed:        {totals['elapsed']:.1f}s")


if __name__ == "__main__":
    main()
