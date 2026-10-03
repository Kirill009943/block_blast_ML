"""Behavioral cloning on the demonstration dataset.

    python -m training.train_imitation --source human
    python -m training.train_imitation --source heuristic --epochs 20
    python -m training.train_imitation --source all --weight-human 5.0 --augment
    python -m training.train_imitation --source all --export-ppo models/imitation/bc_ppo.zip

Trains :class:`training.imitation_model.ImitationPolicy` with masked
cross-entropy: illegal actions get -inf logits before the loss, so no
probability mass is ever assigned to them (the recorded action is always
legal — the recorder enforces it). With ``--source all`` per-source sample
weights keep a small human dataset from being overwhelmed by a large
autonomous one (default: human 5x).

Metrics per epoch: train loss, validation loss, plain accuracy, masked
accuracy, top-3 and top-5 accuracy. The checkpoint with the best
validation masked accuracy is saved (``.pt``); ``--export-ppo`` also
writes a MaskablePPO ``.zip`` usable via ``training.train --init-from``.

High training accuracy does NOT imply good play — always evaluate with
``training.evaluate_imitation`` before drawing conclusions.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from training.demos import CODE_TO_SOURCE, DemoDataset

DEFAULT_DATASET_DIR = "results/demos"
ALL_SOURCES = "all"


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Behavioral cloning training.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--source", type=str, default=ALL_SOURCES,
                        help="human | heuristic | ppo | solver | random | "
                             "imitation | human_accepted_ai | all")
    parser.add_argument("--weight-human", type=float, default=5.0)
    parser.add_argument("--weight-heuristic", type=float, default=1.0)
    parser.add_argument("--weight-ppo", type=float, default=1.0)
    parser.add_argument("--augment", action="store_true",
                        help="4x data via board symmetries (basic profile only)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default=None,
                        help="checkpoint path (default models/imitation/bc_<source>.pt)")
    parser.add_argument("--export-ppo", type=str, default=None,
                        help="also write a MaskablePPO .zip for --init-from")
    parser.add_argument("--eval-games", type=int, default=0,
                        help="optionally evaluate the cloned policy afterwards")
    return parser.parse_args(argv)


def source_weights(args: argparse.Namespace) -> Dict[int, float]:
    """Per-source-code sample weight (unlisted sources weigh 1.0)."""
    from training.demos import SOURCE_TO_CODE

    return {
        SOURCE_TO_CODE["human"]: args.weight_human,
        SOURCE_TO_CODE["human_accepted_ai"]: args.weight_human,
        SOURCE_TO_CODE["heuristic"]: args.weight_heuristic,
        SOURCE_TO_CODE["ppo"]: args.weight_ppo,
    }


def masked_topk_correct(
    logits: torch.Tensor, targets: torch.Tensor, k: int
) -> int:
    """How many targets are within the top-k (already-masked) logits."""
    topk = logits.topk(k, dim=-1).indices
    return int((topk == targets[:, None]).any(dim=1).sum())


def evaluate_split(
    policy: nn.Module,
    obs: np.ndarray,
    masks: np.ndarray,
    actions: np.ndarray,
    weights: np.ndarray,
    device: str,
    batch_size: int,
) -> Dict[str, float]:
    """Loss / accuracy / masked accuracy / top-3 / top-5 on a fixed split."""
    policy.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    totals = {"loss": 0.0, "weight": 0.0, "acc": 0, "masked_acc": 0, "top3": 0, "top5": 0}
    with torch.no_grad():
        for start in range(0, len(actions), batch_size):
            sl = slice(start, start + batch_size)
            batch_obs = torch.as_tensor(obs[sl], dtype=torch.float32, device=device)
            batch_mask = torch.as_tensor(masks[sl], device=device)
            batch_act = torch.as_tensor(actions[sl], dtype=torch.long, device=device)
            batch_w = torch.as_tensor(weights[sl], dtype=torch.float32, device=device)
            logits = policy(batch_obs)
            masked = logits.masked_fill(~batch_mask, float("-inf"))
            loss = loss_fn(masked, batch_act)
            totals["loss"] += float((loss * batch_w).sum())
            totals["weight"] += float(batch_w.sum())
            totals["acc"] += int((logits.argmax(-1) == batch_act).sum())
            totals["masked_acc"] += int((masked.argmax(-1) == batch_act).sum())
            totals["top3"] += masked_topk_correct(masked, batch_act, 3)
            totals["top5"] += masked_topk_correct(masked, batch_act, 5)
    n = len(actions)
    return {
        "loss": totals["loss"] / max(totals["weight"], 1e-9),
        "accuracy": totals["acc"] / n,
        "masked_accuracy": totals["masked_acc"] / n,
        "top3": totals["top3"] / n,
        "top5": totals["top5"] / n,
    }


def train(args: argparse.Namespace) -> Path:
    from training.augment import augment_dataset
    from training.imitation_model import (
        ImitationPolicy,
        export_to_ppo,
        save_checkpoint,
    )

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.default_rng(args.seed)

    sources = None if args.source == ALL_SOURCES else [args.source]
    dataset = DemoDataset(args.data)
    data = dataset.load(sources=sources)
    channels = int(data["channels"])
    print(f"Dataset: {len(data['actions']):,} samples "
          f"({channels} channels) | device: {device}")

    if args.augment:
        data = augment_dataset(data)
        print(f"Augmented: {len(data['actions']):,} samples (4 symmetries)")

    weights = np.ones(len(data["actions"]), dtype=np.float32)
    for code, weight in source_weights(args).items():
        weights[data["sources"] == code] = weight

    train_idx, val_idx = dataset.game_split(
        len(data["actions"]), data["game_ids"],
        val_fraction=args.val_fraction, seed=args.seed,
    )
    print(f"Split: {len(train_idx):,} train / {len(val_idx):,} val (by game)")

    obs = data["observations"].astype(np.float32)
    masks = data["action_masks"]
    actions = data["actions"].astype(np.int64)

    policy = ImitationPolicy(observation_channels=channels).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    out = Path(args.out or f"models/imitation/bc_{args.source}.pt")
    best_val = -1.0
    rng = np.random.default_rng(args.seed)
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        policy.train()
        perm = rng.permutation(len(train_idx))
        train_loss = weight_sum = 0.0
        for start in range(0, len(perm), args.batch_size):
            idx = train_idx[perm[start:start + args.batch_size]]
            batch_obs = torch.as_tensor(obs[idx], device=device)
            batch_mask = torch.as_tensor(masks[idx], device=device)
            batch_act = torch.as_tensor(actions[idx], device=device)
            batch_w = torch.as_tensor(weights[idx], device=device)
            masked = policy.masked_logits(batch_obs, batch_mask)
            loss = (loss_fn(masked, batch_act) * batch_w).sum() / batch_w.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += float(loss) * float(batch_w.sum())
            weight_sum += float(batch_w.sum())

        val = evaluate_split(policy, obs[val_idx], masks[val_idx],
                             actions[val_idx], weights[val_idx], device,
                             args.batch_size)
        marker = ""
        if val["masked_accuracy"] > best_val:
            best_val = val["masked_accuracy"]
            save_checkpoint(policy, out, config=vars(args),
                            metrics={"val": val, "epoch": epoch})
            marker = "  <- saved"
        print(f"epoch {epoch:2d}/{args.epochs}  "
              f"train loss {train_loss / max(weight_sum, 1e-9):.4f}  "
              f"val loss {val['loss']:.4f}  "
              f"acc {val['accuracy']:.3f}  masked acc {val['masked_accuracy']:.3f}  "
              f"top-3 {val['top3']:.3f}  top-5 {val['top5']:.3f}"
              f"  ({time.time() - started:.0f}s){marker}")

    print(f"Best validation masked accuracy: {best_val:.3f}")
    print(f"Saved checkpoint: {out}")

    if args.export_ppo:
        from training.imitation_model import load_policy

        best_policy, _ = load_policy(out, device=device)
        zip_path = export_to_ppo(best_policy, args.export_ppo, device=device)
        print(f"Exported PPO-initializable model: {zip_path}")

    if args.eval_games:
        from training.evaluate import evaluate_agent

        stats = evaluate_agent(f"imitation:{out}", args.eval_games,
                               base_seed=1000, progress=True)
        print(f"Imitation policy evaluation: avg score {stats['avg_score']:.1f} "
              f"over {args.eval_games} games")
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
