"""RL agent wrapping a trained MaskablePPO model.

The agent has no hard-coded strategy: it loads the neural network produced
by ``training/train.py`` and acts purely from what PPO learned. Action
masks are applied at inference time, so it only ever picks legal moves.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from agents.base import Agent, DecisionInfo
from environment.block_blast_env import (
    build_action_mask,
    decode_action,
)
from environment.observations import build_observation, profile_for_channels
from game.game import Action, BlockBlastGame


class RLAgent(Agent):
    """Plays Block Blast using a saved MaskablePPO policy."""

    name = "rl"

    def __init__(self, model_path: Union[str, Path], device: str = "auto"):
        from sb3_contrib import MaskablePPO  # deferred: needs torch

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = MaskablePPO.load(str(self.model_path), device=device)
        # match the observation profile the model was trained with
        channels = int(self.model.observation_space.shape[0])
        self.observation_profile = profile_for_channels(channels)
        self._last_value: Optional[float] = None

    def _predict(self, game: BlockBlastGame, deterministic: bool = True) -> int:
        obs = build_observation(game, self.observation_profile)
        mask = build_action_mask(game)
        action, _ = self.model.predict(obs, action_masks=mask, deterministic=deterministic)
        return int(action)

    def act(self, game: BlockBlastGame) -> Action:
        return decode_action(self._predict(game))

    def value_estimate(self, game: BlockBlastGame) -> float:
        """The critic's estimated value of the current state."""
        import torch

        obs = build_observation(game, self.observation_profile)
        obs_tensor = torch.as_tensor(obs[None], device=self.model.device)
        with torch.no_grad():
            value = self.model.policy.predict_values(obs_tensor)
        return float(value.cpu().numpy()[0, 0])

    def decide(self, game: BlockBlastGame) -> DecisionInfo:
        action_id = self._predict(game)
        value = self.value_estimate(game)
        self._last_value = value
        return DecisionInfo(
            action=decode_action(action_id),
            value=value,
            valid_actions=len(game.get_valid_actions()),
        )

    def action_probabilities(self, game: BlockBlastGame):
        """Masked action probabilities for the current state.

        Returns ``(probs, mask, value)``: a (192,) float array summing to 1
        over legal actions, the boolean legality mask, and the critic's
        value estimate. Used by the AI Brain UI and policy heatmaps.
        """
        import torch

        from environment.block_blast_env import NUM_ACTIONS

        obs = build_observation(game, self.observation_profile)
        mask = build_action_mask(game)
        obs_tensor = torch.as_tensor(obs[None], device=self.model.device)
        mask_tensor = torch.as_tensor(mask[None], device=self.model.device)
        with torch.no_grad():
            features = self.model.policy.extract_features(obs_tensor)
            latent_pi, _ = self.model.policy.mlp_extractor(features)
            logits = self.model.policy.action_net(latent_pi)[0]
            logits = logits.masked_fill(~mask_tensor[0], float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            value = self.model.policy.predict_values(obs_tensor)[0, 0]
        return probs.cpu().numpy(), mask, float(value.cpu())
