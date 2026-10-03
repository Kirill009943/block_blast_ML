"""Pygame modes for collecting human demonstrations.

``RecordingApp`` (human training mode): the normal human game, but every
placement is recorded into the demonstration dataset BEFORE it mutates the
state (via ``DemoRecorder.begin_step``/``end_step``). Invalid placements
are never recorded. ``R`` starts a new game without restarting the app;
quitting flushes the dataset.

``FeedbackApp`` (human feedback mode): an agent suggests a move
(highlighted in cyan). ``A`` accepts it (recorded as
``human_accepted_ai``); placing a different move records ``human`` with
the AI suggestion stored as metadata. This progressively collects
high-quality human corrections.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pygame

from agents.base import Agent
from environment.block_blast_env import encode_action
from training.demos import DemoRecorder
from ui.pygame_app import HEIGHT, MARGIN, PygameApp

SUGGESTION_COLOR = (86, 204, 242)  # cyan
DEFAULT_DATASET_DIR = "results/demos"


class RecordingApp(PygameApp):
    """Human mode that records every move into a DemoRecorder."""

    def __init__(
        self,
        dataset_dir: str = DEFAULT_DATASET_DIR,
        seed: Optional[int] = None,
        observation_profile: str = "basic",
        source: str = "human",
        headless: bool = False,
    ):
        super().__init__(agent=None, seed=seed, headless=headless)
        self.recorder = DemoRecorder(
            dataset_dir, source=source, observation_profile=observation_profile
        )
        self.game_count = 1
        pygame.display.set_caption("Block Blast — Training Mode")

    # ------------------------------------------------------------------ #
    # recording-aware placement
    # ------------------------------------------------------------------ #

    def _place_and_record(
        self, action: Tuple[int, int, int], source: Optional[str] = None,
        suggested_action: int = -1,
    ) -> None:
        """Record ``action`` (pre-state), execute it, store the outcome."""
        slot, row, col = action
        piece = self.game.pieces[slot]
        if piece is None or not self.game.can_place(piece, row, col):
            return  # invalid placements are never recorded
        self.recorder.begin_step(self.game, action,
                                 suggested_action=suggested_action, source=source)
        result = self.game.place_piece(slot, row, col)
        self.recorder.end_step(result)
        self._after_move(result)
        self.selected_slot = None

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
            self._place_and_record((self.selected_slot, row, col))

    # ------------------------------------------------------------------ #
    # game lifecycle
    # ------------------------------------------------------------------ #

    def _restart(self) -> None:
        if self.recorder.pending_samples:
            self.recorder.finish_game()
            self.game_count += 1
        super()._restart()

    def _after_move(self, result) -> None:
        super()._after_move(result)
        if result.game_over:
            self.recorder.finish_game()
            self.game_count += 1

    def run(self) -> None:
        try:
            super().run()
        finally:
            self.recorder.close()
            print(f"Dataset saved to {self.recorder.directory} "
                  f"({self.recorder.total_samples} new samples, "
                  f"{self.recorder.total_games} games)")

    # ------------------------------------------------------------------ #
    # HUD
    # ------------------------------------------------------------------ #

    def _draw_panels(self) -> None:
        super()._draw_panels()
        lines = [
            "TRAINING MODE",
            f"Game: {self.game_count}  Steps: {self.game.moves}",
            f"Dataset samples: {self.recorder.total_samples + self.recorder.pending_samples}",
        ]
        for i, text in enumerate(lines):
            self.screen.blit(
                self.font.render(text, True, SUGGESTION_COLOR),
                (MARGIN, HEIGHT - 24 * (len(lines) - i) - 8),
            )


class FeedbackApp(RecordingApp):
    """RecordingApp where an agent suggests each move (A = accept)."""

    def __init__(
        self,
        agent: Agent,
        dataset_dir: str = DEFAULT_DATASET_DIR,
        seed: Optional[int] = None,
        observation_profile: str = "basic",
        headless: bool = False,
    ):
        super().__init__(dataset_dir, seed=seed,
                         observation_profile=observation_profile,
                         source="human", headless=headless)
        self.agent = agent
        self._suggestion = None  # (piece_idx, row, col) or None
        self._refresh_suggestion()
        pygame.display.set_caption("Block Blast — Feedback Mode")

    def _refresh_suggestion(self) -> None:
        if self.game.is_game_over():
            self._suggestion = None
            return
        decision = self.agent.decide(self.game)
        self._suggestion = decision.action

    def _suggested_action_id(self) -> int:
        return encode_action(*self._suggestion) if self._suggestion else -1

    def _handle_key(self, key: int) -> bool:
        if key == pygame.K_a and self._suggestion is not None and not self.game_over:
            self._place_and_record(
                self._suggestion,
                source="human_accepted_ai",
                suggested_action=self._suggested_action_id(),
            )
            return True
        return super()._handle_key(key)

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
            # the human's own move overrides the suggestion (kept as metadata)
            self._place_and_record(
                (self.selected_slot, row, col),
                source="human",
                suggested_action=self._suggested_action_id(),
            )

    def _after_move(self, result) -> None:
        super()._after_move(result)
        self._refresh_suggestion()

    def _restart(self) -> None:
        super()._restart()
        self._refresh_suggestion()

    def _draw_panels(self) -> None:
        super()._draw_panels()
        # highlight the suggested placement on the board
        if self._suggestion is not None and not self.game_over:
            piece_idx, row, col = self._suggestion
            piece = self.game.pieces[piece_idx]
            if piece is not None:
                for r, c in piece.cells:
                    rect = self._cell_rect(row + r, col + c)
                    pygame.draw.rect(self.screen, SUGGESTION_COLOR, rect,
                                     width=3, border_radius=6)
            text = (f"AI suggests: piece {piece_idx} -> ({row}, {col})   "
                    f"[A]ccept / click own move")
        else:
            text = "AI suggestion: n/a"
        self.screen.blit(self.font.render(text, True, SUGGESTION_COLOR),
                         (MARGIN, HEIGHT - 24 * 4 - 32))


def run_record_app(
    dataset_dir: str = DEFAULT_DATASET_DIR,
    seed: Optional[int] = None,
    observation_profile: str = "basic",
) -> None:
    """Human training mode: play normally while every move is recorded."""
    RecordingApp(dataset_dir=dataset_dir, seed=seed,
                 observation_profile=observation_profile).run()


def run_feedback_app(
    agent: Agent,
    dataset_dir: str = DEFAULT_DATASET_DIR,
    seed: Optional[int] = None,
    observation_profile: str = "basic",
) -> None:
    """Human feedback mode: accept or override the agent's suggestions."""
    FeedbackApp(agent, dataset_dir=dataset_dir, seed=seed,
                observation_profile=observation_profile).run()
