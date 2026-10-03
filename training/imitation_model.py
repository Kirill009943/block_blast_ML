"""Behavioral-cloning policy network, checkpoint format and PPO conversion.

The network deliberately mirrors the policy side of the MaskablePPO agent
used by ``training/train.py`` — same CNN feature extractor, same
``pi=[256, 256]`` trunk, same 192-logit action head — and names its
submodules so the state-dict keys are IDENTICAL to the PPO policy keys
(``features_extractor.*``, ``mlp_extractor.policy_net.*``,
``action_net.*``). A BC checkpoint therefore loads straight into a PPO
policy with ``strict=False`` (the value head and log_std keep their random
init and are learned by PPO afterwards) — no network duplication and no
weight renaming.

Checkpoint format (``.pt``, via ``torch.save``)::

    state_dict            policy weights (PPO-compatible keys)
    observation_channels  4 (basic) or 8 (enhanced)
    features_dim          CNN feature size (256)
    config                training config used (dict)
    metrics               final training metrics (dict)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from gymnasium import spaces
from torch import nn

from environment.block_blast_env import NUM_ACTIONS
from game.game import BOARD_SIZE
from training.cnn_extractor import BlockBlastCNN

FEATURES_DIM = 256
NET_ARCH_PI = (256, 256)


class _PolicyBranch(nn.Module):
    """Matches the key layout of sb3's ``MlpExtractor.policy_net``."""

    def __init__(self, features_dim: int, net_arch_pi: Tuple[int, ...]):
        super().__init__()
        layers = []
        last = features_dim
        for width in net_arch_pi:
            layers += [nn.Linear(last, width), nn.ReLU()]
            last = width
        self.policy_net = nn.Sequential(*layers)


class ImitationPolicy(nn.Module):
    """CNN policy: (N, C, 8, 8) observation -> (N, 192) action logits."""

    def __init__(
        self,
        observation_channels: int = 4,
        features_dim: int = FEATURES_DIM,
        net_arch_pi: Tuple[int, ...] = NET_ARCH_PI,
    ):
        super().__init__()
        observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(observation_channels, BOARD_SIZE, BOARD_SIZE),
        )
        self.features_extractor = BlockBlastCNN(
            observation_space, features_dim=features_dim
        )
        self.mlp_extractor = _PolicyBranch(features_dim, net_arch_pi)
        self.action_net = nn.Linear(net_arch_pi[-1], NUM_ACTIONS)
        self.observation_channels = observation_channels

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        features = self.features_extractor(observations)
        latent = self.mlp_extractor.policy_net(features)
        return self.action_net(latent)

    def masked_logits(
        self, observations: torch.Tensor, action_mask: torch.Tensor
    ) -> torch.Tensor:
        """Logits with illegal actions set to -inf (no probability mass)."""
        logits = self.forward(observations)
        return logits.masked_fill(~action_mask, float("-inf"))


# ---------------------------------------------------------------------- #
# checkpoint I/O
# ---------------------------------------------------------------------- #

def save_checkpoint(
    policy: ImitationPolicy,
    path: Path | str,
    config: Optional[Dict] = None,
    metrics: Optional[Dict] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "observation_channels": policy.observation_channels,
            "features_dim": FEATURES_DIM,
            "config": config or {},
            "metrics": metrics or {},
        },
        path,
    )
    return path


def load_imitation_state_dict(
    path: Path | str, device: str = "cpu"
) -> Tuple[Dict, Dict]:
    """Return ``(state_dict, checkpoint_meta)`` for a ``.pt`` checkpoint."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    return checkpoint["state_dict"], checkpoint


def load_policy(path: Path | str, device: str = "auto") -> Tuple[ImitationPolicy, Dict]:
    """Load an ImitationPolicy from a ``.pt`` checkpoint."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    state, meta = load_imitation_state_dict(path, device)
    policy = ImitationPolicy(
        observation_channels=int(meta.get("observation_channels", 4)),
        features_dim=int(meta.get("features_dim", FEATURES_DIM)),
    )
    policy.load_state_dict(state)
    policy.to(device)
    policy.eval()
    return policy, meta


def export_to_ppo(
    policy: ImitationPolicy,
    out: Path | str,
    device: str = "auto",
) -> Path:
    """Write the BC policy as a MaskablePPO ``.zip`` for ``--init-from``.

    The policy-side weights are transferred; the value head stays at its
    random initialization (PPO learns it during fine-tuning).
    """
    from sb3_contrib import MaskablePPO

    from environment.block_blast_env import BlockBlastEnv
    from environment.observations import profile_for_channels

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    env = BlockBlastEnv(
        observation_profile=profile_for_channels(policy.observation_channels)
    )
    policy_kwargs = dict(
        features_extractor_class=BlockBlastCNN,
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
        net_arch=dict(pi=list(NET_ARCH_PI), vf=list(NET_ARCH_PI)),
    )
    model = MaskablePPO("MlpPolicy", env, policy_kwargs=policy_kwargs,
                        n_steps=128, batch_size=64, device=device, verbose=0)
    model.policy.load_state_dict(policy.state_dict(), strict=False)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out))
    return out
