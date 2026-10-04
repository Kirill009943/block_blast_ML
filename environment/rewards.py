"""Reward configuration and named reward profiles.

All reward weights live here — change them in exactly one place.

Components (every one is logged individually by the training callbacks, so
you can tell whether the agent learned useful behavior or is exploiting a
single term):

* ``place_per_cell``  + per placed cell. Placing is the only way to make
  progress, so this rewards survival without being farmable.
* ``line``            + per cleared line.
* ``combo``           + ``combo * lines**2``: superlinear multi-line bonus.
* ``game_over``       - on termination.
* ``holes``           - per newly created hole (empty cell trapped under an
  occupied one).
* ``fragmentation``   - per disconnected empty region on the new board.
* ``occupancy``       - per occupied cell on the new board (rewards keeping
  the board open).
* ``future_moves``    + per valid action remaining after the move (rewards
  keeping options open; absolute level — use with small weights only).
* ``future_moves_delta`` + per NET change in legal options caused by the
  move: ``weight * (valid_after - valid_before)``, clipped to
  ``+/- future_moves_delta_clip``. Moves that destroy many future options
  are penalized; moves that open the board are rewarded. Bounded and
  farm-resistant: the episode sum of deltas telescopes to
  ``final - initial``, so it cannot be farmed by oscillation. Set to 0 on
  the move that triggers a piece-set refresh (the new random set would
  drown the placement signal).
* ``fragmentation_delta`` - per NET new disconnected empty region:
  ``-weight * (regions_after - regions_before)``, clipped to
  ``+/- fragmentation_delta_clip``.
* ``survival_per_move`` + per move survived.
* ``reward_scale``    the final per-step reward is multiplied by this
  factor. Pure PPO value-target scaling: the logged per-component
  diagnostics stay in raw game units, only the scalar the optimizer
  sees changes. Use e.g. 0.05 when the critic cannot fit returns of
  magnitude ~200-300 (symptom: value_loss in the thousands, explained
  variance near 0). Default 1.0 = no scaling.
* ``reward_clip``     per-step reward is clipped to ``+/- reward_clip``
  when > 0 (bounds the value targets PPO's critic must fit — the
  unbounded combo spikes are a major driver of return variance).
  Applied AFTER ``reward_scale``.

Profiles are selected with ``--reward-profile``; ``baseline`` reproduces
the original training reward exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RewardConfig:
    """All reward weights in one place. See module docstring for details."""

    place_per_cell: float = 1.0
    line: float = 10.0
    combo: float = 5.0  # multiplied by lines_cleared ** 2
    game_over: float = 50.0  # subtracted on termination
    holes: float = 0.1  # subtracted per newly created hole; 0 disables
    fragmentation: float = 0.0  # subtracted per empty region on new board
    occupancy: float = 0.0  # subtracted per occupied cell on new board
    future_moves: float = 0.0  # added per valid action after the move
    survival_per_move: float = 0.0  # added per non-terminal move
    future_moves_delta: float = 0.0  # weight per net legal-option change
    future_moves_delta_clip: float = 3.0  # bound on the weighted component
    fragmentation_delta: float = 0.0  # weight per net new empty region
    fragmentation_delta_clip: float = 3.0  # bound on the weighted component
    reward_scale: float = 1.0  # final reward multiplier (PPO value scaling)
    reward_clip: float = 0.0  # per-step clip after scaling; 0 disables


REWARD_PROFILES: Dict[str, RewardConfig] = {
    # original reward — do not change, it is the permanent baseline
    "baseline": RewardConfig(),
    # heavier game-over penalty + small per-move survival bonus
    "survival": RewardConfig(
        game_over=100.0,
        survival_per_move=0.2,
        holes=0.1,
    ),
    # line clears dominate; placing cells barely matters
    "lines": RewardConfig(
        place_per_cell=0.5,
        line=15.0,
        combo=10.0,
        game_over=50.0,
        holes=0.1,
    ),
    # board-quality shaping: holes, fragmentation, occupancy, future options
    "strategic": RewardConfig(
        place_per_cell=1.0,
        line=10.0,
        combo=5.0,
        game_over=50.0,
        holes=0.2,
        fragmentation=0.05,
        occupancy=0.02,
        future_moves=0.02,
    ),
    # long-term survival profile: line clears stay dominant, but the agent
    # is also rewarded for keeping future options open (delta-based),
    # penalized for digging holes and fragmenting the board (delta-based),
    # and gets a small per-move survival bonus. Per-step rewards are
    # clipped so the critic's value targets stay learnable.
    "balanced": RewardConfig(
        place_per_cell=0.5,
        line=10.0,
        combo=4.0,
        game_over=50.0,
        holes=0.25,
        fragmentation_delta=0.2,
        future_moves_delta=0.05,
        future_moves_delta_clip=3.0,
        survival_per_move=0.1,
        reward_clip=25.0,
    ),
    "balanced_v2": RewardConfig(
      place_per_cell=0.5,
      line=10.0,
      combo=4.0,
      game_over=80.0,
      holes=0.25,
      fragmentation_delta=0.2,
      future_moves_delta=0.05,
      survival_per_move=0.5,
      reward_clip=25.0,
  ),
}

DEFAULT_REWARD_PROFILE = "baseline"


def get_reward_config(profile: str) -> RewardConfig:
    """Return the :class:`RewardConfig` for a named profile."""
    try:
        return REWARD_PROFILES[profile]
    except KeyError:
        known = ", ".join(sorted(REWARD_PROFILES))
        raise ValueError(f"Unknown reward profile {profile!r}. Known: {known}") from None
