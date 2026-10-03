"""Behavior cloning from the expert dataset.

Trains the MaskablePPO policy network to imitate the heuristic agent with
masked cross-entropy (illegal actions get -inf logits, so probability mass
is only distributed over legal moves). Only the policy side is trained —
the value head stays at its random initialization and is learned by PPO
afterwards.

Usage:
    python -m training.generate_expert_data --games 1000
    python -m training.behavior_cloning --dataset results/expert_dataset.npz --epochs 10
    python -m training.train --init-from models/bc_init.zip --timesteps 5000000

Whether BC initialization actually helps *evaluation score* is an
empirical question — compare against the baseline on identical seeds
(e.g. via ``training/experiments.py``) before believing it.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from environment.block_blast_env import NUM_ACTIONS, BlockBlastEnv
from training.cnn_extractor import BlockBlastCNN


def load_dataset(path: Path):
    data = np.load(path)
    return (data["observations"], data["action_masks"], data["actions"])


def behavior_clone(dataset: Path, out: Path, epochs: int = 10, batch_size: int = 512,
                   lr: float = 1e-3, device: str = "auto",
                   observation_profile: str = "basic", eval_games: int = 0,
                   progress: bool = True) -> Path:
    from sb3_contrib import MaskablePPO

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    observations, masks, actions = load_dataset(dataset)
    n = len(actions)
    print(f"Dataset: {n:,} samples | device: {device}")

    env = BlockBlastEnv(observation_profile=observation_profile)
    policy_kwargs = dict(
        features_extractor_class=BlockBlastCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )
    model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs,
                        n_steps=128, batch_size=64, device=device, verbose=0)
    policy = model.policy

    obs_t = torch.as_tensor(observations, device=device)
    mask_t = torch.as_tensor(masks, device=device)
    act_t = torch.as_tensor(actions, device=device)

    # only train the feature extractor + policy branch
    params = (list(policy.features_extractor.parameters())
              + list(policy.mlp_extractor.policy_net.parameters())
              + list(policy.action_net.parameters()))
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    started = time.time()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total_loss = total_correct = total_seen = 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_obs, batch_mask, batch_act = obs_t[idx], mask_t[idx], act_t[idx]
            features = policy.extract_features(batch_obs)
            latent_pi, _ = policy.mlp_extractor(features)
            logits = policy.action_net(latent_pi)
            logits = logits.masked_fill(~batch_mask, float("-inf"))
            loss = loss_fn(logits, batch_act)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss) * len(idx)
            total_correct += int((logits.argmax(dim=-1) == batch_act).sum())
            total_seen += len(idx)
        if progress:
            print(f"epoch {epoch + 1}/{epochs}  loss {total_loss / total_seen:.4f}  "
                  f"top-1 accuracy {total_correct / total_seen:.3f}  "
                  f"({time.time() - started:.0f}s)")

    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    print(f"Saved BC-initialized model: {out}")

    if eval_games:
        from training.evaluate import evaluate_agent

        stats = evaluate_agent(f"rl:{out}", eval_games, base_seed=1000)
        print(f"BC policy evaluation: avg score {stats['avg_score']:.1f} "
              f"over {eval_games} games (heuristic reference: ~705)")
    return out


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Behavior cloning from expert data.")
    parser.add_argument("--dataset", type=str, default="results/expert_dataset.npz")
    parser.add_argument("--out", type=str, default="models/bc_init.zip")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--observation-profile", type=str, default="basic")
    parser.add_argument("--eval-games", type=int, default=0,
                        help="optionally evaluate the cloned policy afterwards")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    behavior_clone(Path(args.dataset), Path(args.out), epochs=args.epochs,
                   batch_size=args.batch_size, lr=args.lr, device=args.device,
                   observation_profile=args.observation_profile,
                   eval_games=args.eval_games)


if __name__ == "__main__":
    main()
