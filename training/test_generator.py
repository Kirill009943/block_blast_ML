"""Generator stress test: proves the board-aware generator never produces
unplayable pieces, and reports its overhead.

    python -m training.test_generator --games 1000
    python -m training.test_generator --games 200 --density-steps 12

Two suites:

* ``games`` — full games played with random valid moves; every piece set
  generated during play is checked for Level-1 validity (each piece has a
  legal placement) and Level-2 viability (some ordering can be placed
  sequentially, unless it was an accepted fallback).
* ``density`` — synthetic boards filled to increasing densities; 200 sets
  are generated per density level and checked the same way.

Exit code is non-zero if any unplayable piece is generated, so this can
run in CI.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import numpy as np

from game import BOARD_SIZE, BlockBlastGame


def check_set(game: BlockBlastGame) -> dict:
    """Validate the current piece set of ``game`` against both levels."""
    pieces = [p for p in game.pieces if p is not None]
    result = {"pieces": len(pieces), "unplayable": 0, "viable": True}
    for piece in pieces:
        if not game.has_valid_move_for_piece(piece):
            result["unplayable"] += 1
    if pieces:
        result["viable"] = game.is_piece_set_playable(game.pieces)
    return result


def run_game_suite(games: int, base_seed: int) -> dict:
    """Validate each set right after it is generated.

    (After the player moves, remaining pieces may legitimately become
    stranded — the guarantee holds at generation time, not forever.)
    """
    totals = {
        "sets": 0, "unplayable_pieces": 0, "nonviable_sets": 0,
    }
    started = time.time()
    for i in range(games):
        game = BlockBlastGame(seed=base_seed + i)
        rng = np.random.default_rng(i)
        fresh_set = True  # reset() just generated a set
        while not game.is_game_over():
            if fresh_set:
                check = check_set(game)
                totals["sets"] += 1
                totals["unplayable_pieces"] += check["unplayable"]
                totals["nonviable_sets"] += 0 if check["viable"] else 1
                fresh_set = False
            actions = game.get_valid_actions()
            if not actions:
                break
            result = game.place_piece(*actions[int(rng.integers(0, len(actions)))])
            fresh_set = result.pieces_refreshed
    totals["elapsed"] = time.time() - started
    return totals


def run_stats_pass(games: int, base_seed: int) -> dict:
    """Replay games accumulating the per-game generation_stats dicts."""
    agg = {
        "sets_generated": 0, "candidate_attempts": 0, "candidates_rejected": 0,
        "max_attempts_single_piece": 0, "type_fallbacks": 0, "set_attempts": 0,
        "sets_regenerated": 0, "viable_sets": 0, "level1_fallbacks": 0,
        "game_over_boards": 0,
    }
    for i in range(games):
        game = BlockBlastGame(seed=base_seed + i)
        rng = np.random.default_rng(i)  # once per game, matches run_game_suite
        while not game.is_game_over():
            actions = game.get_valid_actions()
            if not actions:
                break
            game.place_piece(*actions[int(rng.integers(0, len(actions)))])
        for key, value in game.generation_stats.items():
            if key == "max_attempts_single_piece":
                agg[key] = max(agg[key], value)
            else:
                agg[key] += value
    return agg


def run_density_suite(max_fill: int, sets_per_density: int, seed: int) -> None:
    print("\nDensity suite (synthetic boards)")
    print("--------------------------------")
    rng = np.random.default_rng(seed)
    for filled in range(0, max_fill + 1, 8):
        unplayable = nonviable = dead = 0
        for s in range(sets_per_density):
            game = BlockBlastGame(seed=seed + s)
            cells = rng.choice(BOARD_SIZE * BOARD_SIZE, size=filled, replace=False)
            game.board[:] = 0
            for cell in cells:
                game.board[cell // BOARD_SIZE, cell % BOARD_SIZE] = 1
            game._valid_cache = None
            game._playable_types_cache = None
            game.pieces = game.generate_piece_set()
            if all(p is None for p in game.pieces):
                dead += 1
                continue
            check = check_set(game)
            unplayable += check["unplayable"]
            nonviable += 0 if check["viable"] else 1
        print(f"  filled {filled:2d}/64: unplayable={unplayable} "
              f"nonviable={nonviable} dead-boards={dead} (of {sets_per_density})")
        if unplayable:
            raise SystemExit("FAIL: unplayable piece generated")


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress-test the piece generator.")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=70_000)
    parser.add_argument("--density-sets", type=int, default=200,
                        help="sets per density level in the density suite")
    parser.add_argument("--skip-density", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    print(f"Game suite: {args.games} games with random play")
    started = time.time()
    totals = run_game_suite(args.games, args.seed)
    stats = run_stats_pass(args.games, args.seed)
    elapsed = time.time() - started

    attempts = stats["candidate_attempts"]
    sets = max(stats["sets_generated"], 1)
    print(f"\nGames tested:                {args.games}")
    print(f"Sets checked during play:    {totals['sets']}")
    print(f"Sets generated:              {stats['sets_generated']}")
    print(f"Unplayable pieces:           {totals['unplayable_pieces']}")
    print(f"Non-viable sets (in play):   {totals['nonviable_sets']}")
    print(f"Sequentially viable sets:    {stats['viable_sets']} "
          f"({100 * stats['viable_sets'] / sets:.1f}%)")
    print(f"Level-1 fallbacks used:      {stats['level1_fallbacks']}")
    print(f"Type fallbacks used:         {stats['type_fallbacks']}")
    print(f"Dead boards (game over):     {stats['game_over_boards']}")
    print(f"Candidate rejections:        {stats['candidates_rejected']}")
    print(f"Avg attempts per candidate:  {attempts / max(stats['sets_generated'] * 3, 1):.2f}")
    print(f"Max attempts (single piece): {stats['max_attempts_single_piece']}")
    print(f"Sets requiring regeneration: {stats['sets_regenerated']} "
          f"({100 * stats['sets_regenerated'] / sets:.1f}%)")
    print(f"Elapsed: {elapsed:.1f}s")

    if not args.skip_density:
        run_density_suite(64, args.density_sets, args.seed + 1)

    if totals["unplayable_pieces"]:
        print("\nFAIL: generator produced unplayable pieces")
        sys.exit(1)
    print("\nPASS: no unplayable piece was ever generated")


if __name__ == "__main__":
    main()
