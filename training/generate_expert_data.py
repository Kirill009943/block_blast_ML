"""Generate an expert dataset by recording HeuristicAgent games.

For every decision the heuristic makes we store:

* ``observations`` — the (C, 8, 8) observation before the move
* ``action_masks`` — the (192,) boolean legality mask
* ``actions``       — the expert's chosen action id
* ``game_scores``   — final score of the game each sample came from

The dataset can be used for behavior cloning (see
``training/behavior_cloning.py``): train the policy network with masked
cross-entropy against the expert action, then use the result as PPO
initialization via ``python -m training.train --init-from <model>``.
Whether that actually helps evaluation score is an empirical question —
run it as an experiment and compare on identical seeds.

Usage:
    python -m training.generate_expert_data --games 1000
    python -m training.generate_expert_data --games 10000 --out results/expert_dataset.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np

from agents.heuristic_agent import HeuristicAgent
from environment.block_blast_env import build_action_mask, encode_action
from environment.observations import build_observation, observation_channels
from game import BlockBlastGame


def generate(games: int, base_seed: int = 50_000, observation_profile: str = "basic",
             out: Path = Path("results/expert_dataset.npz"),
             progress: bool = True) -> Path:
    channels = observation_channels(observation_profile)
    observations = []
    masks = []
    actions = []
    scores = []

    started = time.time()
    for i in range(games):
        game = BlockBlastGame(seed=base_seed + i)
        agent = HeuristicAgent()
        while not game.is_game_over():
            obs = build_observation(game, observation_profile)
            mask = build_action_mask(game)
            piece_idx, row, col = agent.act(game)
            observations.append(obs)
            masks.append(mask)
            actions.append(encode_action(piece_idx, row, col))
            game.place_piece(piece_idx, row, col)
        scores.extend([game.score] * (len(actions) - len(scores)))
        if progress and (i + 1) % max(games // 10, 1) == 0:
            rate = (i + 1) / max(time.time() - started, 1e-9)
            print(f"  {i + 1}/{games} games ({rate:.1f} games/s, "
                  f"{len(actions)} samples so far)")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        observations=np.stack(observations).astype(np.float32),
        action_masks=np.stack(masks),
        actions=np.asarray(actions, dtype=np.int64),
        game_scores=np.asarray(scores, dtype=np.int32),
    )
    print(f"\nSaved {len(actions):,} samples from {games} games to {out}")
    print(f"  observations: ({len(actions)}, {channels}, 8, 8) float32")
    print(f"  action_masks: ({len(actions)}, 192) bool")
    print(f"  actions:      ({len(actions)},) int64")
    print(f"  avg expert score: {np.mean(scores):.1f}")
    return out


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record HeuristicAgent games as an expert dataset.")
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=50_000,
                        help="base seed (kept far from evaluation seeds)")
    parser.add_argument("--observation-profile", type=str, default="basic",
                        choices=["basic", "enhanced"])
    parser.add_argument("--out", type=str, default="results/expert_dataset.npz")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    generate(args.games, base_seed=args.seed,
             observation_profile=args.observation_profile, out=Path(args.out))


if __name__ == "__main__":
    main()
