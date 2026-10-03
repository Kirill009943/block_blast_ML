"""Pygame UI for Block Blast.

Human mode:
    * click a piece in the tray to select it
    * hover the board for a ghost preview (green = legal, red = illegal)
    * click a cell to place

AI watch mode:
    * the chosen agent plays visibly: the selected piece and target cells
      are highlighted before placement, cleared lines flash
    * ``+`` / ``-`` adjust the move delay, D toggles the debug panel

Keys: H = toggle Human/AI, R = restart, D = debug info, +/- = AI delay,
Esc = quit.

The UI only talks to the game engine and agents through their public API;
no game logic lives here.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import pygame

from agents.base import Agent, DecisionInfo
from environment.block_blast_env import RewardConfig
from game import BOARD_SIZE, EMPTY, BlockBlastGame

AI_MOVE_DELAY = 0.01  # seconds between AI moves (adjust with +/-)
HIGHLIGHT_TIME = 0.01  # how long the chosen placement is highlighted
FLASH_TIME = 0.01  # line-clear flash duration

CELL = 56
GAP = 4
BOARD_PX = BOARD_SIZE * (CELL + GAP) + GAP
MARGIN = 20
PANEL_X = MARGIN + BOARD_PX + MARGIN
WIDTH = PANEL_X + 320
HEIGHT = BOARD_PX + 2 * MARGIN + 50

PALETTE = {
    0: (40, 44, 52),
    1: (235, 87, 87),
    2: (242, 153, 74),
    3: (242, 201, 76),
    4: (111, 207, 151),
    5: (86, 204, 242),
    6: (47, 128, 237),
    7: (155, 81, 224),
    8: (224, 86, 168),
}
BG = (24, 26, 32)
TEXT = (220, 224, 230)
GHOST_OK = (80, 200, 120, 120)
GHOST_BAD = (200, 80, 80, 120)
HIGHLIGHT = (255, 255, 255)


class PygameApp:
    """Interactive Block Blast application (human play + AI watch)."""

    def __init__(self, agent: Optional[Agent] = None, seed: Optional[int] = None,
                 headless: bool = False):
        if headless:
            import os

            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.display.set_caption("Block Blast")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.big_font = pygame.font.SysFont("consolas", 28, bold=True)

        self.game = BlockBlastGame(seed=seed)
        self.agent = agent
        self.ai_mode = agent is not None
        self.debug = False
        self.ai_delay = AI_MOVE_DELAY

        self.selected_slot: Optional[int] = None
        self.game_over = False

        # AI state machine
        self._next_ai_move_at = time.monotonic() + self.ai_delay
        self._pending: Optional[DecisionInfo] = None
        self._pending_until = 0.0
        self._flash_until = 0.0
        self._last_decision: Optional[DecisionInfo] = None
        self._last_points = 0

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
                    running = self._handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)

            if self.ai_mode and not self.game_over:
                self._ai_tick()

            self._draw()
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

    def _handle_key(self, key: int) -> bool:
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_r:
            self._restart()
        elif key == pygame.K_h:
            if self.agent is not None:
                self.ai_mode = not self.ai_mode
                self._next_ai_move_at = time.monotonic() + self.ai_delay
        elif key == pygame.K_d:
            self.debug = not self.debug
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.ai_delay = max(0.000001, self.ai_delay / 2)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.ai_delay = min(5.0, self.ai_delay * 2)
        return True

    def _restart(self) -> None:
        self.game.reset()
        self.game_over = False
        self.selected_slot = None
        self._pending = None
        self._last_decision = None
        self._next_ai_move_at = time.monotonic() + self.ai_delay

    # ------------------------------------------------------------------ #
    # human input
    # ------------------------------------------------------------------ #

    def _board_cell_at(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        x, y = pos
        bx, by = x - MARGIN, y - MARGIN
        if bx < 0 or by < 0:
            return None
        col, row = bx // (CELL + GAP), by // (CELL + GAP)
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            return int(row), int(col)
        return None

    def _tray_slot_at(self, pos: Tuple[int, int]) -> Optional[int]:
        x, y = pos
        for slot in range(3):
            rect = self._tray_rect(slot)
            if rect.collidepoint(x, y) and self.game.pieces[slot] is not None:
                return slot
        return None

    def _tray_rect(self, slot: int) -> pygame.Rect:
        return pygame.Rect(PANEL_X, 200 + slot * 130, 280, 110)

    def _handle_click(self, pos: Tuple[int, int]) -> None:
        if self.ai_mode or self.game_over:
            return
        slot = self._tray_slot_at(pos)
        if slot is not None:
            self.selected_slot = slot if self.selected_slot != slot else None
            return
        cell = self._board_cell_at(pos)
        if cell is not None and self.selected_slot is not None:
            row, col = cell
            result = self.game.place_piece(self.selected_slot, row, col)
            if result.success:
                self._after_move(result)
                self.selected_slot = None

    # ------------------------------------------------------------------ #
    # AI control
    # ------------------------------------------------------------------ #

    def _ai_tick(self) -> None:
        now = time.monotonic()
        if now < self._flash_until:
            return  # wait for the clear animation
        if self._pending is not None:
            if now >= self._pending_until:
                decision, self._pending = self._pending, None
                assert decision.action is not None
                result = self.game.place_piece(*decision.action)
                if result.success:
                    self._after_move(result)
            return
        if now < self._next_ai_move_at:
            return
        if self.game.is_game_over():
            self.game_over = True
            return
        assert self.agent is not None
        decision = self.agent.decide(self.game)
        self._last_decision = decision
        self._pending = decision
        self._pending_until = now + HIGHLIGHT_TIME

    def _after_move(self, result) -> None:
        self._last_points = result.points_gained
        if result.lines_cleared:
            # board is already cleared by the engine; flash the board frame
            self._flash_until = time.monotonic() + FLASH_TIME
        if result.game_over:
            self.game_over = True
        self._next_ai_move_at = time.monotonic() + self.ai_delay

    # ------------------------------------------------------------------ #
    # drawing
    # ------------------------------------------------------------------ #

    def _draw(self) -> None:
        self.screen.fill(BG)
        self._draw_board()
        self._draw_ghost()
        self._draw_panels()
        if self.debug:
            self._draw_debug()

    def _cell_rect(self, row: int, col: int) -> pygame.Rect:
        x = MARGIN + col * (CELL + GAP) + GAP
        y = MARGIN + row * (CELL + GAP) + GAP
        return pygame.Rect(x, y, CELL, CELL)

    def _draw_board(self) -> None:
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                rect = self._cell_rect(row, col)
                color = PALETTE[int(self.game.board[row, col])]
                pygame.draw.rect(self.screen, color, rect, border_radius=6)
        # flash the board frame briefly after a line clear
        if time.monotonic() < self._flash_until:
            frame = pygame.Rect(MARGIN, MARGIN, BOARD_PX, BOARD_PX)
            pygame.draw.rect(self.screen, HIGHLIGHT, frame, width=4, border_radius=8)
        # highlight the AI's chosen target cells
        if self._pending is not None and self._pending.action is not None:
            piece_idx, row, col = self._pending.action
            piece = self.game.pieces[piece_idx]
            if piece is not None:
                for r, c in piece.cells:
                    rect = self._cell_rect(row + r, col + c)
                    pygame.draw.rect(self.screen, HIGHLIGHT, rect, width=3, border_radius=6)

    def _draw_ghost(self) -> None:
        if self.ai_mode or self.selected_slot is None or self.game_over:
            return
        cell = self._board_cell_at(pygame.mouse.get_pos())
        if cell is None:
            return
        row, col = cell
        piece = self.game.pieces[self.selected_slot]
        if piece is None:
            return
        ok = self.game.can_place(piece, row, col)
        overlay = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
        overlay.fill(GHOST_OK if ok else GHOST_BAD)
        for r, c in piece.cells:
            rr, cc = row + r, col + c
            if 0 <= rr < BOARD_SIZE and 0 <= cc < BOARD_SIZE:
                self.screen.blit(overlay, self._cell_rect(rr, cc))

    def _draw_piece(self, piece, origin: Tuple[int, int], scale: int) -> None:
        ox, oy = origin
        for r, c in piece.cells:
            rect = pygame.Rect(ox + c * (scale + 2), oy + r * (scale + 2), scale, scale)
            pygame.draw.rect(self.screen, PALETTE[piece.color], rect, border_radius=4)

    def _draw_panels(self) -> None:
        mode = "AI WATCH" if self.ai_mode else "HUMAN"
        title = self.big_font.render("BLOCK BLAST", True, TEXT)
        self.screen.blit(title, (PANEL_X, 24))
        self.screen.blit(self.font.render(f"Score: {self.game.score}", True, TEXT),
                         (PANEL_X, 70))
        self.screen.blit(self.font.render(f"Moves: {self.game.moves}   Lines: {self.game.total_lines_cleared}",
                                          True, TEXT), (PANEL_X, 96))
        self.screen.blit(self.font.render(f"Mode: {mode}  (+/- delay: {self.ai_delay:.2f}s)",
                                          True, TEXT), (PANEL_X, 122))
        self.screen.blit(self.font.render("H: mode  R: restart  D: debug", True, TEXT),
                         (PANEL_X, 144))

        for slot in range(3):
            rect = self._tray_rect(slot)
            selected = slot == self.selected_slot and not self.ai_mode
            pending = (self._pending is not None and self._pending.action is not None
                       and self._pending.action[0] == slot)
            border = HIGHLIGHT if (selected or pending) else (70, 74, 82)
            pygame.draw.rect(self.screen, (32, 34, 42), rect, border_radius=8)
            pygame.draw.rect(self.screen, border, rect, width=2, border_radius=8)
            piece = self.game.pieces[slot]
            if piece is not None:
                scale = 22
                ox = rect.x + (rect.width - piece.width * (scale + 2)) // 2
                oy = rect.y + (rect.height - piece.height * (scale + 2)) // 2
                self._draw_piece(piece, (ox, oy), scale)
            else:
                used = self.font.render("used", True, (90, 94, 102))
                self.screen.blit(used, (rect.x + 110, rect.y + 42))

        if self.game_over:
            over = self.big_font.render("GAME OVER", True, (235, 87, 87))
            self.screen.blit(over, (PANEL_X, HEIGHT - 60))
            self.screen.blit(self.font.render("press R to restart", True, TEXT),
                             (PANEL_X, HEIGHT - 30))

    def _draw_debug(self) -> None:
        info = self._last_decision
        lines = ["AI decision", "-----------"]
        if info is not None and info.action is not None:
            piece_idx, row, col = info.action
            lines += [
                f"Piece:    {piece_idx}",
                f"Position: ({row}, {col})",
                f"Value:    {info.value:.2f}" if info.value is not None else "Value:    n/a",
                f"Valid:    {info.valid_actions} actions",
                f"Last pts: {self._last_points}",
            ]
        else:
            lines.append("(no move yet)")
        for i, text in enumerate(lines):
            y = HEIGHT - 22 * (len(lines) - i) - 26
            self.screen.blit(self.font.render(text, True, TEXT), (MARGIN, y))


def load_default_agent() -> Agent:
    """Best available agent: best trained model > smoke model > heuristic."""
    from agents.heuristic_agent import HeuristicAgent

    candidates = [
        Path("models/best/best_model.zip"),
        Path("models/demo_final.zip"),
        Path("models/smoke_final.zip"),
    ]
    for path in candidates:
        if path.exists():
            try:
                from agents.rl_agent import RLAgent

                print(f"Loaded RL agent: {path}")
                return RLAgent(path)
            except Exception as exc:  # missing torch etc. — fall back
                print(f"Could not load {path}: {exc}")
    print("No trained model found — using the heuristic agent.")
    return HeuristicAgent()


def run_app(agent: Optional[Agent] = None, human: bool = False,
            seed: Optional[int] = None) -> None:
    """Start the UI. ``human=True`` starts in human mode even with an agent."""
    if agent is None and not human:
        agent = load_default_agent()
    app = PygameApp(agent=agent, seed=seed)
    if human:
        app.ai_mode = False
    app.run()
