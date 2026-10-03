"""AI Brain mode: watch a trained RL agent play while seeing its decisions.

Left: the normal Block Blast board. Right: the agent's "brain" — masked
action-probability heatmaps (one 8x8 grid per piece slot, illegal
positions in dark gray, selected action outlined), the critic's value
estimate, the top-5 candidate placements with probabilities, a small
architecture diagram, and a live preview of the first conv layer's
feature maps.

This mode is only for observing a trained model — never used in training.

    python main.py --mode brain --agent rl:models/main_final.zip --ai-delay 0.3
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np
import pygame

from agents.rl_agent import RLAgent
from environment.block_blast_env import decode_action, encode_action
from game import BOARD_SIZE, BlockBlastGame
from ui.pygame_app import PALETTE, BG, TEXT, HIGHLIGHT

CELL = 52
GAP = 4
BOARD_PX = BOARD_SIZE * (CELL + GAP) + GAP
MARGIN = 20
PANEL_X = MARGIN + BOARD_PX + MARGIN
WIDTH = PANEL_X + 430
HEIGHT = max(BOARD_PX + 2 * MARGIN, 760)

HEAT = 96  # pixel size of each policy heatmap
ARCH_BOX = (110, 26)


class BrainApp:
    """Pygame app visualizing the RL agent's decision process live."""

    def __init__(self, agent: RLAgent, seed: int = 0, ai_delay: float = 0.3,
                 headless: bool = False):
        if headless:
            import os

            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.display.set_caption("Block Blast — AI Brain")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 16)
        self.small_font = pygame.font.SysFont("consolas", 13)
        self.big_font = pygame.font.SysFont("consolas", 26, bold=True)

        self.agent = agent
        self.game = BlockBlastGame(seed=seed)
        self.seed = seed
        self.ai_delay = ai_delay
        self.game_number = 1
        self.total_score = 0
        self.best_score = 0

        # decision state for the currently visualized move
        self.probs: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.value = 0.0
        self.selected_action: Optional[int] = None
        self.top5: List[Tuple[int, float]] = []
        self.feature_maps: Optional[np.ndarray] = None

        self._move_ready_at = time.monotonic() + self.ai_delay
        self._execute_at: Optional[float] = None
        self._restart_at: Optional[float] = None
        self.game_over = False

    # ------------------------------------------------------------------ #
    # decision pipeline
    # ------------------------------------------------------------------ #

    def _decide(self) -> None:
        self.probs, self.mask, self.value = self.agent.action_probabilities(self.game)
        order = np.argsort(self.probs)[::-1]
        self.top5 = [(int(a), float(self.probs[a])) for a in order[:5] if self.probs[a] > 0]
        self.selected_action = int(order[0]) if len(order) else None
        self.feature_maps = self._conv1_maps()

    def _conv1_maps(self, n_maps: int = 8) -> Optional[np.ndarray]:
        """First conv layer feature maps for the activation preview."""
        import torch

        from environment.observations import build_observation

        obs = build_observation(self.game, self.agent.observation_profile)
        obs_t = torch.as_tensor(obs[None], dtype=torch.float32, device=self.agent.model.device)
        conv1 = self.agent.model.policy.features_extractor.cnn[0]
        with torch.no_grad():
            acts = conv1(obs_t)[0].cpu().numpy()
        return acts[:n_maps]

    def _restart(self) -> None:
        self.total_score += self.game.score
        self.best_score = max(self.best_score, self.game.score)
        self.game_number += 1
        self.game = BlockBlastGame(seed=self.seed + self.game_number * 1000)
        self.game_over = False
        self.probs = None
        self.mask = None
        self.top5 = []
        self.selected_action = None
        self.feature_maps = None
        self._move_ready_at = time.monotonic() + self.ai_delay
        self._restart_at = None

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        self._restart()
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        self.ai_delay = max(0.05, self.ai_delay / 2)
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.ai_delay = min(3.0, self.ai_delay * 2)

            now = time.monotonic()
            if self._restart_at is not None and now >= self._restart_at:
                self._restart()
            elif self._execute_at is not None and now >= self._execute_at:
                assert self.selected_action is not None
                result = self.game.place_piece(*decode_action(self.selected_action))
                self._execute_at = None
                self.probs = None
                if result.game_over:
                    self.game_over = True
                    self._restart_at = now + 2.0
                else:
                    self._move_ready_at = now + self.ai_delay
            elif not self.game_over and now >= self._move_ready_at and self.probs is None:
                if self.game.is_game_over():
                    self.game_over = True
                    self._restart_at = now + 2.0
                else:
                    self._decide()
                    self._execute_at = now + self.ai_delay

            self._draw()
            pygame.display.flip()
            self.clock.tick(30)
        pygame.quit()

    # ------------------------------------------------------------------ #
    # drawing
    # ------------------------------------------------------------------ #

    def _cell_rect(self, row: int, col: int) -> pygame.Rect:
        x = MARGIN + col * (CELL + GAP) + GAP
        y = MARGIN + row * (CELL + GAP) + GAP
        return pygame.Rect(x, y, CELL, CELL)

    def _draw(self) -> None:
        self.screen.fill(BG)
        self._draw_board()
        self._draw_brain_panel()

    def _draw_board(self) -> None:
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                pygame.draw.rect(self.screen, PALETTE[int(self.game.board[row, col])],
                                 self._cell_rect(row, col), border_radius=6)
        # highlight the selected placement
        if self.selected_action is not None and self._execute_at is not None:
            piece_idx, row, col = decode_action(self.selected_action)
            piece = self.game.pieces[piece_idx]
            if piece is not None:
                for r, c in piece.cells:
                    pygame.draw.rect(self.screen, HIGHLIGHT,
                                     self._cell_rect(row + r, col + c), width=3,
                                     border_radius=6)

    def _draw_heatmap(self, slot: int, origin: Tuple[int, int]) -> None:
        ox, oy = origin
        cell = HEAT // BOARD_SIZE
        frame = pygame.Rect(ox - 2, oy - 2, HEAT + 4, HEAT + 4)
        pygame.draw.rect(self.screen, (70, 74, 82), frame, width=1)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                action = slot * 64 + r * BOARD_SIZE + c
                rect = pygame.Rect(ox + c * cell, oy + r * cell, cell - 1, cell - 1)
                if self.mask is None or self.probs is None or not self.mask[action]:
                    color = (45, 48, 56)  # illegal / no data
                else:
                    p = self.probs[action]
                    color = (int(40 + 215 * p / max(self.top5[0][1], 1e-9)), 90,
                             int(200 - 150 * p))
                pygame.draw.rect(self.screen, color, rect)
        # outline the selected action
        if self.selected_action is not None and self._execute_at is not None:
            s_piece, s_row, s_col = decode_action(self.selected_action)
            if s_piece == slot:
                rect = pygame.Rect(ox + s_col * cell - 1, oy + s_row * cell - 1,
                                   cell + 1, cell + 1)
                pygame.draw.rect(self.screen, HIGHLIGHT, rect, width=2)

    def _draw_architecture(self, origin: Tuple[int, int]) -> None:
        ox, oy = origin
        layers = ["Input (4,8,8)", "Conv 64", "Conv 64", "Conv 128",
                  "FC 256", "Policy 192 / Value 1"]
        for i, name in enumerate(layers):
            rect = pygame.Rect(ox, oy + i * (ARCH_BOX[1] + 6), ARCH_BOX[0] + 60, ARCH_BOX[1])
            pygame.draw.rect(self.screen, (40, 60, 100), rect, border_radius=4)
            pygame.draw.rect(self.screen, (120, 150, 200), rect, width=1, border_radius=4)
            self.screen.blit(self.small_font.render(name, True, TEXT),
                             (rect.x + 6, rect.y + 6))
            if i < len(layers) - 1:
                x = rect.centerx
                pygame.draw.line(self.screen, (120, 150, 200), (x, rect.bottom),
                                 (x, rect.bottom + 6), 2)

    def _draw_brain_panel(self) -> None:
        x = PANEL_X
        self.screen.blit(self.big_font.render("AI BRAIN", True, TEXT), (x, 16))
        lines = [
            f"Game #{self.game_number}   Score {self.game.score}   Best {self.best_score}",
            f"Moves {self.game.moves}   Lines {self.game.total_lines_cleared}",
            f"Legal actions: {int(self.mask.sum()) if self.mask is not None else len(self.game.get_valid_actions())}",
            f"Value estimate: {self.value:.2f}",
            f"Move delay: {self.ai_delay:.2f}s (+/- to change, R restart)",
        ]
        for i, text in enumerate(lines):
            self.screen.blit(self.font.render(text, True, TEXT), (x, 54 + i * 20))

        # selected action
        y = 54 + len(lines) * 20 + 6
        if self.selected_action is not None and self._execute_at is not None:
            p, r, c = decode_action(self.selected_action)
            prob = float(self.probs[self.selected_action])
            self.screen.blit(self.font.render(
                f"SELECTED: piece {p} at ({r}, {c})  p={prob:.3f}", True,
                (120, 220, 120)), (x, y))
        y += 26

        # policy heatmaps
        self.screen.blit(self.small_font.render("action probability per piece slot:",
                                                True, TEXT), (x, y))
        y += 20
        for slot in range(3):
            self._draw_heatmap(slot, (x + slot * (HEAT + 30), y))
            self.screen.blit(self.small_font.render(f"slot {slot}", True, TEXT),
                             (x + slot * (HEAT + 30), y + HEAT + 4))
        y += HEAT + 26

        # top-5 candidates
        self.screen.blit(self.small_font.render("top candidates:", True, TEXT), (x, y))
        for i, (action, prob) in enumerate(self.top5):
            p, r, c = decode_action(action)
            self.screen.blit(self.small_font.render(
                f"  {i + 1}. piece {p} ({r},{c})  p={prob:.3f}", True, TEXT),
                (x, y + 18 * (i + 1)))
        y += 18 * 6 + 8

        # activation preview
        if self.feature_maps is not None:
            self.screen.blit(self.small_font.render("conv1 activations:", True, TEXT), (x, y))
            thumb = 24
            for i, fmap in enumerate(self.feature_maps):
                norm = (fmap - fmap.min()) / max(fmap.ptp(), 1e-9)
                surf = pygame.Surface((BOARD_SIZE, BOARD_SIZE))
                for r in range(BOARD_SIZE):
                    for c in range(BOARD_SIZE):
                        v = int(255 * norm[r, c])
                        surf.set_at((c, r), (v, v, min(255, v + 40)))
                surf = pygame.transform.scale(surf, (thumb, thumb))
                self.screen.blit(surf, (x + i * (thumb + 4), y + 18))
            y += 18 + thumb + 10

        # architecture mini-diagram
        self.screen.blit(self.small_font.render("network:", True, TEXT), (x, y))
        self._draw_architecture((x, y + 18))


def run_brain(agent: RLAgent, seed: int = 0, ai_delay: float = 0.3,
              headless: bool = False) -> None:
    BrainApp(agent, seed=seed, ai_delay=ai_delay, headless=headless).run()
