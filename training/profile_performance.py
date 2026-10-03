"""Profile the environment and training loop.

Measures, without changing any behavior:

* environment step time (place + obs + info)
* action-mask generation time
* board-analysis time (holes / regions — the reward-shaping helpers)
* neural-network inference time (single obs and batched)
* IPC overhead: DummyVecEnv vs SubprocVecEnv stepping throughput

Usage:
    python -m training.profile_performance
    python -m training.profile_performance --steps 20000 --model models/main_final.zip
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np

from environment.block_blast_env import BlockBlastEnv
from game.utils import count_empty_regions, count_holes


def time_it(label: str, fn, iterations: int) -> float:
    fn()  # warmup
    started = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - started
    per_call_us = elapsed / iterations * 1e6
    print(f"  {label:44s} {per_call_us:9.1f} us/call "
          f"({iterations / elapsed:,.0f} calls/s)")
    return per_call_us


def profile_env(steps: int) -> None:
    print("\n[1] Environment components (baseline reward profile)")
    env = BlockBlastEnv()
    env.reset(seed=0)
    rng = np.random.default_rng(0)

    def random_action():
        return int(rng.choice(np.flatnonzero(env.action_masks())))

    def do_step():
        action = random_action()
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            env.reset()

    def mask_only():
        env.action_masks()

    def obs_only():
        env._observation()

    def holes_only():
        count_holes(env.game.board)

    def regions_only():
        count_empty_regions(env.game.board)

    time_it("env.step (incl. place, obs, info)", do_step, steps)
    time_it("action_masks()", mask_only, steps)
    time_it("build_observation (basic)", obs_only, steps)
    time_it("count_holes", holes_only, steps)
    time_it("count_empty_regions", regions_only, steps)

    env_enhanced = BlockBlastEnv(observation_profile="enhanced")
    env_enhanced.reset(seed=0)
    time_it("build_observation (enhanced)",
            lambda: env_enhanced._observation(), steps)


def profile_nn(steps: int, model_path: Optional[str]) -> None:
    if not model_path or not Path(model_path).exists():
        print("\n[2] Neural network inference — skipped (no model)")
        return
    print("\n[2] Neural network inference")
    import torch

    from agents.rl_agent import RLAgent
    from game import BlockBlastGame

    agent = RLAgent(model_path)
    game = BlockBlastGame(seed=0)
    time_it("agent.action_probabilities (single state)",
            lambda: agent.action_probabilities(game), min(steps, 5000))

    obs = np.zeros((256,) + tuple(agent.model.observation_space.shape), dtype=np.float32)
    obs_t = torch.as_tensor(obs, device=agent.model.device)

    def batch_forward():
        with torch.no_grad():
            agent.model.policy.extract_features(obs_t)

    time_it("policy.extract_features (batch 256)", batch_forward, 200)


def profile_vecenv(steps: int, n_envs: int) -> None:
    print("\n[3] VecEnv IPC overhead")
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    def make(rank):
        def _init():
            env = BlockBlastEnv()
            env.reset(seed=rank)
            return env
        return _init

    rng = np.random.default_rng(0)
    for cls in (DummyVecEnv, SubprocVecEnv):
        vec = cls([make(i) for i in range(n_envs)])
        vec.reset()
        # random-but-legal actions need masks; use action 0..191 uniform and
        # let resets absorb terminations — for throughput comparison only
        actions = np.array([rng.integers(0, 192) for _ in range(n_envs)])
        vec.step(actions)  # warmup
        started = time.perf_counter()
        total = 0
        while total < steps:
            vec.step(actions)
            total += n_envs
        elapsed = time.perf_counter() - started
        print(f"  {cls.__name__:44s} {total / elapsed:9,.0f} env steps/s")
        vec.close()


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile env + training loop.")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--model", type=str, default="models/main_final.zip")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    profile_env(args.steps)
    profile_nn(args.steps, args.model)
    profile_vecenv(args.steps * 2, args.n_envs)


if __name__ == "__main__":
    main()
