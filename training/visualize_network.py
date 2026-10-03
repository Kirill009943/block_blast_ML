"""Neural-network diagnostic visualizations.

Two modes (select with ``--mode``):

* ``architecture`` — layer graph of the policy network: input -> Conv1 ->
  Conv2 -> Conv3 -> Flatten -> FC 256 -> policy/value heads, with tensor
  shapes, parameter counts, activations and trainable status.
  Saved to ``results/network_architecture.png``.

* ``weights`` / ``activations`` / ``policy`` — learned-data diagnostics:
  conv filter grids, feature maps for a real game state, dense-layer
  weight magnitude, and the 192 masked action probabilities drawn as three
  8x8 heatmaps (one per piece slot) with illegal actions masked out and
  the selected action highlighted. Saved to ``results/activations/`` and
  ``results/policy_maps/``.

These are diagnostic views of what the network computes — they do not
prove the network "understands" the game.

Usage:
    python -m training.visualize_network --model models/main_final.zip --mode all
    python -m training.visualize_network --model models/main_final.zip --mode policy --seed 5
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # file output only; safe on headless machines
import matplotlib.pyplot as plt
import numpy as np
import torch

from environment.block_blast_env import (
    NUM_ACTIONS,
    build_action_mask,
    decode_action,
    encode_action,
)
from environment.observations import build_observation, profile_for_channels
from game import BOARD_SIZE, BlockBlastGame

RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------- #
# model helpers
# ---------------------------------------------------------------------- #

def load_model(model_path: str, device: str = "cpu"):
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(model_path, device=device)
    model.policy.eval()
    return model


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def masked_probs(model, game: BlockBlastGame, profile: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """(192,) masked action probabilities + legality mask + value estimate.

    When the game is over (no legal actions) probabilities are all zero.
    """
    obs = build_observation(game, profile)
    mask = build_action_mask(game)
    obs_t = torch.as_tensor(obs[None], dtype=torch.float32, device=model.device)
    with torch.no_grad():
        features = model.policy.extract_features(obs_t)
        latent_pi, _ = model.policy.mlp_extractor(features)
        logits = model.policy.action_net(latent_pi)[0]
        value = float(model.policy.predict_values(obs_t)[0, 0])
        if mask.any():
            mask_t = torch.as_tensor(mask, device=model.device)
            logits = logits.masked_fill(~mask_t, float("-inf"))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        else:
            probs = np.zeros(NUM_ACTIONS, dtype=np.float32)
    return probs, mask, value


# ---------------------------------------------------------------------- #
# Mode A: architecture graph
# ---------------------------------------------------------------------- #

def visualize_architecture(model, out_path: Path) -> Path:
    """Draw the layer graph with shapes, params and activation functions."""
    policy = model.policy
    extractor = policy.features_extractor
    in_shape = tuple(model.observation_space.shape)

    # walk the network with a dummy input to record real tensor shapes
    layers: List[Dict] = []

    def add(name, module, shape, activation):
        layers.append({
            "name": name,
            "shape": shape,
            "params": count_parameters(module) if module is not None else 0,
            "activation": activation,
            "trainable": (module is not None and
                          any(p.requires_grad for p in module.parameters())),
        })

    dummy = torch.zeros((1,) + in_shape, device=model.device)
    x = dummy
    add("Input", None, in_shape, "-")
    conv_index = 0
    for module in extractor.cnn:
        if isinstance(module, torch.nn.Conv2d):
            conv_index += 1
            x = module(x)
            add(f"Conv{conv_index}", module, tuple(x.shape[1:]), "ReLU")
    flat = x.flatten(1)
    add("Flatten", None, (flat.shape[1],), "-")
    fc = extractor.linear[0]
    features = extractor.linear(flat)
    add("FC 256", fc, (features.shape[1],), "ReLU")

    latent_pi, latent_vf = policy.mlp_extractor(features)
    pi_params = count_parameters(policy.mlp_extractor.policy_net) + \
        count_parameters(policy.action_net)
    vf_params = count_parameters(policy.mlp_extractor.value_net) + \
        count_parameters(policy.value_net)
    logits = policy.action_net(latent_pi)
    value = policy.value_net(latent_vf)
    add("Policy head", policy.action_net, (logits.shape[1],), "softmax(masked)")
    layers[-1]["params"] = pi_params
    layers[-1]["name"] = "Policy head (192 actions)"
    add("Value head", policy.value_net, (value.shape[1],), "-")
    layers[-1]["params"] = vf_params

    total = count_parameters(policy)

    fig, ax = plt.subplots(figsize=(10, 2.0 + 1.1 * len(layers)))
    ax.axis("off")
    ax.set_title(f"BlockBlastCNN + MaskablePPO heads — {total:,} trainable parameters")
    box_w = 0.52
    x_center = 0.5
    step = 0.9 / len(layers)
    box_h = step * 0.72
    for i, layer in enumerate(layers):
        y = 0.96 - i * step - box_h / 2
        color = "#dbeafe" if layer["trainable"] or layer["name"] == "Input" else "#e5e7eb"
        rect = plt.Rectangle((x_center - box_w / 2, y - box_h / 2), box_w, box_h,
                             facecolor=color, edgecolor="#1e3a8a", lw=1.4)
        ax.add_patch(rect)
        trainable = "trainable" if layer["trainable"] else "fixed"
        text = (f"{layer['name']}\n"
                f"shape {layer['shape']} | params {layer['params']:,} | "
                f"{layer['activation']} | {trainable}")
        ax.text(x_center, y, text, ha="center", va="center", fontsize=9, family="monospace")
        if i < len(layers) - 1:
            ax.annotate("", xy=(x_center, y - box_h / 2 - step * 0.26),
                        xytext=(x_center, y - box_h / 2),
                        arrowprops=dict(arrowstyle="->", color="#1e3a8a", lw=1.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------- #
# Mode B1: conv filters + dense weights
# ---------------------------------------------------------------------- #

def visualize_weights(model, out_path: Path) -> Path:
    """Conv filter grids and dense-layer weight magnitude."""
    extractor = model.policy.features_extractor
    convs = [m for m in extractor.cnn if isinstance(m, torch.nn.Conv2d)]
    fc = extractor.linear[0]

    fig = plt.figure(figsize=(18, 10))
    grid = fig.add_gridspec(len(convs) + 1, 1, height_ratios=[1, 1, 1, 0.8])

    for row, conv in enumerate(convs):
        weight = conv.weight.detach().cpu().numpy()  # (out, in, kh, kw)
        n_out, n_in = weight.shape[0], weight.shape[1]
        show = min(n_out, 32)
        ax = fig.add_subplot(grid[row])
        # tile: each row = one output filter, columns = its input channels
        tiles = []
        for f in range(show):
            tiles.append(np.concatenate([weight[f, c] for c in range(n_in)], axis=1))
        image = np.concatenate(tiles, axis=0)
        vmax = np.abs(image).max()
        ax.imshow(image, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"Conv{row + 1} filters (first {show}/{n_out} outputs x "
                     f"{n_in} input channels, 3x3 each) — blue/red = neg/pos")
        ax.set_xticks([])
        ax.set_yticks([])

    ax = fig.add_subplot(grid[len(convs)])
    dense = fc.weight.detach().cpu().numpy().ravel()
    ax.hist(dense, bins=100, color="#1e3a8a", alpha=0.8)
    ax.set_title(f"FC 256 weight distribution (mean {dense.mean():.4f}, "
                 f"std {dense.std():.4f}, {dense.size:,} weights)")
    ax.set_ylabel("count")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------- #
# Mode B2: feature maps for a real game state
# ---------------------------------------------------------------------- #

def visualize_activations(model, game: BlockBlastGame, out_path: Path,
                          profile: str) -> Path:
    """Forward a real game state and show the CNN feature maps."""
    obs = build_observation(game, profile)
    obs_t = torch.as_tensor(obs[None], dtype=torch.float32, device=model.device)

    extractor = model.policy.features_extractor
    convs = [m for m in extractor.cnn if isinstance(m, torch.nn.Conv2d)]
    activations: List[np.ndarray] = []
    hooks = []

    def hook(_, __, output):
        activations.append(output[0].detach().cpu().numpy())

    for conv in convs:
        hooks.append(conv.register_forward_hook(hook))
    with torch.no_grad():
        model.policy.extract_features(obs_t)
    for h in hooks:
        h.remove()

    n_rows = len(convs) + 1
    fig, axes = plt.subplots(n_rows, 9, figsize=(18, 2.2 * n_rows))
    # row 0: the input channels
    for c in range(obs.shape[0]):
        ax = axes[0, c]
        ax.imshow(obs[c], cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"input ch{c}", fontsize=8)
        ax.axis("off")
    for c in range(obs.shape[0], 9):
        axes[0, c].axis("off")
    axes[0, 0].set_ylabel("input", fontsize=9)

    for row, acts in enumerate(activations, start=1):
        for c in range(9):
            ax = axes[row, c]
            if c < min(8, acts.shape[0]):
                ax.imshow(acts[c], cmap="magma")
                ax.set_title(f"map {c}", fontsize=8)
            ax.axis("off")
        axes[row, 0].set_ylabel(f"conv{row}", fontsize=9)

    fig.suptitle(f"Feature maps for game state (score={game.score}, "
                 f"moves={game.moves})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------- #
# Mode B3: policy heatmaps (3 grids of 8x8)
# ---------------------------------------------------------------------- #

def visualize_policy(model, game: BlockBlastGame, out_path: Path,
                     profile: str) -> Path:
    """Draw the 192 masked action probabilities as three 8x8 heatmaps."""
    probs, mask, value = masked_probs(model, game, profile)
    selected = int(np.argmax(probs))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.2),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    for slot in range(3):
        ax = axes[slot]
        grid = np.full((BOARD_SIZE, BOARD_SIZE), np.nan)
        for cell in range(BOARD_SIZE * BOARD_SIZE):
            action = slot * 64 + cell
            r, c = cell // BOARD_SIZE, cell % BOARD_SIZE
            if mask[action]:
                grid[r, c] = probs[action]
        # illegal positions stay NaN -> shown as light gray
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="#d1d5db")
        ax.imshow(grid, cmap=cmap)
        piece = game.pieces[slot]
        label = piece.name if piece is not None else "used"
        ax.set_title(f"Piece slot {slot}: {label}\n"
                     f"max prob {np.nanmax(grid) if not np.isnan(grid).all() else 0:.3f}")
        ax.set_xticks(range(BOARD_SIZE))
        ax.set_yticks(range(BOARD_SIZE))
        # mark selected action
        s_piece, s_row, s_col = decode_action(selected)
        if s_piece == slot:
            rect = plt.Rectangle((s_col - 0.5, s_row - 0.5), 1, 1,
                                 fill=False, edgecolor="red", lw=3)
            ax.add_patch(rect)

    ax = axes[3]
    ax.imshow(game.occupancy(), cmap="Greys", vmin=0, vmax=1)
    ax.set_title(f"Board (score={game.score}, moves={game.moves})")
    ax.set_xticks(range(BOARD_SIZE))
    ax.set_yticks(range(BOARD_SIZE))

    s_piece, s_row, s_col = decode_action(selected)
    fig.suptitle(
        f"Masked action probabilities | selected: piece {s_piece} at "
        f"({s_row}, {s_col}) with p={probs[selected]:.3f} | "
        f"value estimate {value:.2f} | legal actions {int(mask.sum())} | "
        f"gray = illegal",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #

def run_all(model_path: str, mode: str, seed: int, device: str = "cpu") -> List[Path]:
    model = load_model(model_path, device=device)
    profile = profile_for_channels(int(model.observation_space.shape[0]))
    # play a few random valid moves so the state is non-trivial; if the
    # random game ends, retry with fewer moves to keep a playable state
    rng = np.random.default_rng(seed)
    game = BlockBlastGame(seed=seed)
    for target in (12, 8, 5, 3):
        game = BlockBlastGame(seed=seed)
        for _ in range(target):
            actions = game.get_valid_actions()
            if not actions:
                break
            game.place_piece(*actions[int(rng.integers(0, len(actions)))])
        if not game.is_game_over():
            break

    outputs: List[Path] = []
    if mode in ("architecture", "all"):
        outputs.append(visualize_architecture(model, RESULTS_DIR / "network_architecture.png"))
    if mode in ("weights", "all"):
        outputs.append(visualize_weights(model, RESULTS_DIR / "activations" / "conv_weights.png"))
    if mode in ("activations", "all"):
        outputs.append(visualize_activations(
            model, game, RESULTS_DIR / "activations" / f"feature_maps_seed{seed}.png", profile))
    if mode in ("policy", "all"):
        outputs.append(visualize_policy(
            model, game, RESULTS_DIR / "policy_maps" / f"policy_seed{seed}.png", profile))
    return outputs


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the Block Blast PPO network.")
    parser.add_argument("--model", type=str, default="models/main_final.zip")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["architecture", "weights", "activations", "policy", "all"])
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the example game state")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = parse_args(argv)
    outputs = run_all(args.model, args.mode, args.seed, device=args.device)
    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
