"""Agent that plays using a behavior-cloned policy (no search, no RL).

Loads an ``ImitationPolicy`` checkpoint produced by
``training/train_imitation.py`` and acts from what it learned from the
demonstrations. The valid-action mask is always applied, so the agent can
only select legal moves:

* deterministic mode — ``argmax`` over masked action probabilities;
* stochastic mode — sample from the masked softmax (more human-like
  variety; two games with the same seed can still differ across agents
  unless they share the seed).

Heavy imports (torch, the model module) are deferred to ``__init__`` so the
``agents`` package stays importable without torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from agents.base import Agent, DecisionInfo
from environment.block_blast_env import build_action_mask, decode_action
from environment.observations import build_observation, profile_for_channels
from game.game import Action, BlockBlastGame


class ImitationAgent(Agent):
    """Plays Block Blast with a saved behavioral-cloning policy."""

    name = "imitation"

    def __init__(
        self,
        model_path: Union[str, Path],
        deterministic: bool = True,
        device: str = "auto",
        seed: Optional[int] = None,
    ):
        import torch  # deferred: keep `import agents` torch-free

        from training.imitation_model import load_policy

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.policy, self.meta = load_policy(self.model_path, device=device)
        self._torch = torch
        self.device = next(self.policy.parameters()).device
        self.observation_profile = profile_for_channels(
            self.policy.observation_channels
        )
        self.deterministic = deterministic
        self._rng = np.random.default_rng(seed)

    def action_probabilities(self, game: BlockBlastGame):
        """Masked action probabilities for the current state.

        Returns ``(probs, mask)``: a (192,) float array summing to 1 over
        legal actions and the boolean legality mask.
        """
        torch = self._torch
        obs = build_observation(game, self.observation_profile)
        mask = build_action_mask(game)
        obs_t = torch.as_tensor(obs[None], dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(mask[None], device=self.device)
        with torch.no_grad():
            logits = self.policy.masked_logits(obs_t, mask_t)[0]
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy(), mask

    def _predict(self, game: BlockBlastGame) -> int:
        probs, _ = self.action_probabilities(game)
        if self.deterministic:
            return int(probs.argmax())  # invalid actions have probability 0
        return int(self._rng.choice(len(probs), p=probs))

    def act(self, game: BlockBlastGame) -> Action:
        return decode_action(self._predict(game))

    def decide(self, game: BlockBlastGame) -> DecisionInfo:
        probs, mask = self.action_probabilities(game)
        if self.deterministic:
            action_id = int(probs.argmax())
        else:
            action_id = int(self._rng.choice(len(probs), p=probs))
        top = probs.argsort()[::-1][:5]
        candidates = [(float(probs[a]), decode_action(int(a))) for a in top]
        return DecisionInfo(
            action=decode_action(action_id),
            value=None,
            valid_actions=int(mask.sum()),
            extra={
                "top_candidates": candidates,
                "action_probability": float(probs[action_id]),
            },
        )
