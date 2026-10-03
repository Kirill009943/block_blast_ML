"""Tests for the pure game engine (game/game.py)."""

import numpy as np
import pytest

from game import BOARD_SIZE, EMPTY, BlockBlastGame, GameConfig
from game.pieces import PIECE_TYPES
from helpers import make_piece, set_pieces


# ---------------------------------------------------------------------- #
# placement
# ---------------------------------------------------------------------- #

def three_singles():
    return [make_piece([(0, 0)], color=i + 1) for i in range(3)]


def test_place_piece_success():
    game = BlockBlastGame(seed=0)
    pieces = three_singles()
    pieces[0] = make_piece([(0, 0), (1, 0)])
    set_pieces(game, pieces)

    result = game.place_piece(0, 3, 4)

    assert result.success
    assert result.cells_placed == 2
    assert game.board[3, 4] == 1
    assert game.board[4, 4] == 1
    assert game.pieces[0] is None
    assert not result.pieces_refreshed


def test_place_piece_overlap_rejected():
    game = BlockBlastGame(seed=0)
    game.board[0, 0] = 5
    domino = make_piece([(0, 0), (0, 1)])
    set_pieces(game, [domino, None, None])

    result = game.place_piece(0, 0, 0)

    assert not result.success
    # board unchanged apart from the pre-existing cell
    assert game.board.sum() == 5
    assert game.pieces[0] is not None


@pytest.mark.parametrize("row,col", [(-1, 0), (0, -1), (BOARD_SIZE, 0), (0, BOARD_SIZE),
                                     (BOARD_SIZE - 1, 0)])
def test_place_piece_out_of_bounds(row, col):
    game = BlockBlastGame(seed=0)
    domino = make_piece([(0, 0), (1, 0)])  # vertical: needs row+1 to exist
    set_pieces(game, [domino] + three_singles()[:2])

    assert not game.place_piece(0, row, col).success


def test_place_used_piece_rejected():
    game = BlockBlastGame(seed=0)
    set_pieces(game, [None, None, None])
    assert not game.place_piece(0, 0, 0).success


def test_place_invalid_index_rejected():
    game = BlockBlastGame(seed=0)
    assert not game.place_piece(7, 0, 0).success
    assert not game.place_piece(-1, 0, 0).success


def test_can_place_boundaries():
    game = BlockBlastGame(seed=0)
    domino_v = make_piece([(0, 0), (1, 0)])
    assert game.can_place(domino_v, BOARD_SIZE - 2, BOARD_SIZE - 1)
    assert not game.can_place(domino_v, BOARD_SIZE - 1, 0)


# ---------------------------------------------------------------------- #
# line clearing
# ---------------------------------------------------------------------- #

def fill_row_except(game, row, free_col):
    for c in range(BOARD_SIZE):
        if c != free_col:
            game.board[row, c] = 1


def test_row_clearing():
    game = BlockBlastGame(seed=0)
    fill_row_except(game, 2, free_col=0)
    single = make_piece([(0, 0)])
    set_pieces(game, [single, None, None])

    result = game.place_piece(0, 2, 0)

    assert result.success
    assert result.rows_cleared == 1
    assert result.cols_cleared == 0
    assert np.all(game.board[2] == EMPTY)


def test_column_clearing():
    game = BlockBlastGame(seed=0)
    for r in range(BOARD_SIZE - 1):
        game.board[r, 5] = 1
    single = make_piece([(0, 0)])
    set_pieces(game, [single, None, None])

    result = game.place_piece(0, BOARD_SIZE - 1, 5)

    assert result.cols_cleared == 1
    assert result.rows_cleared == 0
    assert np.all(game.board[:, 5] == EMPTY)


def test_simultaneous_row_and_column_clearing():
    game = BlockBlastGame(seed=0)
    # Row 0 complete except (0,0); column 0 complete except (0,0).
    # One piece at (0,0) completes both; both must clear (original console
    # code cleared rows first and could miss the column).
    fill_row_except(game, 0, free_col=0)
    for r in range(1, BOARD_SIZE):
        game.board[r, 0] = 2
    single = make_piece([(0, 0)])
    set_pieces(game, [single, None, None])

    result = game.place_piece(0, 0, 0)

    assert result.rows_cleared == 1
    assert result.cols_cleared == 1
    assert result.lines_cleared == 2
    assert np.all(game.board == EMPTY)


def test_multi_row_combo_bonus():
    config = GameConfig(points_per_cell=0, points_per_line=10, combo_bonus=10)
    game = BlockBlastGame(seed=0, config=config)
    for row in (3, 4):
        fill_row_except(game, row, free_col=7)
    domino_v = make_piece([(0, 0), (1, 0)])
    set_pieces(game, [domino_v, None, None])

    result = game.place_piece(0, 3, 7)

    assert result.lines_cleared == 2
    # 2 lines * 10 + combo bonus 10 * (2*1//2) = 30
    assert result.points_gained == 30
    assert game.score == 30


# ---------------------------------------------------------------------- #
# scoring
# ---------------------------------------------------------------------- #

def test_score_counts_cells_and_lines():
    config = GameConfig(points_per_cell=1, points_per_line=10, combo_bonus=10)
    game = BlockBlastGame(seed=0, config=config)
    fill_row_except(game, 1, free_col=0)
    single = make_piece([(0, 0)])
    set_pieces(game, [single, None, None])

    result = game.place_piece(0, 1, 0)

    assert result.points_gained == 1 + 10  # 1 cell + 1 line
    assert game.get_score() == 11


def test_original_scoring_mode():
    config = GameConfig(points_per_cell=0, points_per_line=10, combo_bonus=0)
    game = BlockBlastGame(seed=0, config=config)
    fill_row_except(game, 0, free_col=0)
    single = make_piece([(0, 0)])
    set_pieces(game, [single, None, None])

    result = game.place_piece(0, 0, 0)

    assert result.points_gained == 10


# ---------------------------------------------------------------------- #
# piece generation
# ---------------------------------------------------------------------- #

def test_new_pieces_generated_after_all_three_used():
    game = BlockBlastGame(seed=1)
    singles = [make_piece([(0, 0)], color=i + 1) for i in range(3)]
    set_pieces(game, singles)

    for i in range(3):
        result = game.place_piece(i, 0, i * 2)
        assert result.success
        if i < 2:
            assert not result.pieces_refreshed
        else:
            assert result.pieces_refreshed

    assert all(p is not None for p in game.pieces)
    assert len(game.pieces) == 3


def test_seeded_piece_sequence_reproducible():
    game_a = BlockBlastGame(seed=42)
    game_b = BlockBlastGame(seed=42)
    for _ in range(5):  # force several regenerations
        seq_a = [(p.type_id, p.color) for p in game_a.pieces]
        seq_b = [(p.type_id, p.color) for p in game_b.pieces]
        assert seq_a == seq_b
        for g in (game_a, game_b):
            g.pieces = g._draw_pieces()


def test_different_seeds_diverge():
    game_a = BlockBlastGame(seed=1)
    game_b = BlockBlastGame(seed=2)
    seq_a = [(p.type_id, p.color) for p in game_a.pieces]
    seq_b = [(p.type_id, p.color) for p in game_b.pieces]
    assert seq_a != seq_b


# ---------------------------------------------------------------------- #
# game over
# ---------------------------------------------------------------------- #

def test_game_over_detection_no_space():
    game = BlockBlastGame(seed=0)
    # Fill everything except one cell; only a 1-cell piece could fit.
    game.board[:] = 1
    game.board[4, 4] = EMPTY
    big = make_piece([(0, 0), (0, 1), (1, 0), (1, 1)])  # 2x2 square
    set_pieces(game, [big, None, None])

    assert game.is_game_over()


def test_not_game_over_when_move_exists():
    game = BlockBlastGame(seed=0)
    assert not game.is_game_over()


def test_place_piece_reports_game_over():
    game = BlockBlastGame(seed=0)
    game.board[:] = 1
    game.board[0, 0] = EMPTY
    game.board[0, 1] = EMPTY
    # remaining piece is 2x2 which cannot fit after this move's piece
    domino_h = make_piece([(0, 0), (0, 1)])
    square = make_piece([(0, 0), (0, 1), (1, 0), (1, 1)])
    set_pieces(game, [domino_h, square, None])
    # columns 0 and 1 are full -> placing the domino clears them
    result = game.place_piece(0, 0, 0)
    assert result.success
    # after clearing columns 0 and 1 the rest of the board is still full,
    # the 2x2 square still fits in the cleared area -> not game over
    assert not result.game_over
    result2 = game.place_piece(1, 0, 0)
    assert result2.success
    # refresh happens; new pieces unknown, just verify flag consistency
    assert result2.game_over == game.is_game_over()


# ---------------------------------------------------------------------- #
# valid actions
# ---------------------------------------------------------------------- #

def test_get_valid_actions_all_legal():
    game = BlockBlastGame(seed=3)
    actions = game.get_valid_actions()
    assert actions, "fresh board must have valid actions"
    for piece_idx, row, col in actions:
        piece = game.pieces[piece_idx]
        assert piece is not None
        assert game.can_place(piece, row, col)


def test_get_valid_actions_matches_brute_force():
    game = BlockBlastGame(seed=5)
    rng = np.random.default_rng(0)
    game.board = (rng.integers(0, 2, size=(BOARD_SIZE, BOARD_SIZE))).astype(np.int8) * 3

    actions = set(game.get_valid_actions())
    brute = set()
    for idx, piece in enumerate(game.pieces):
        if piece is None:
            continue
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if game.can_place(piece, row, col):
                    brute.add((idx, row, col))
    assert actions == brute


def test_get_valid_actions_skips_used_pieces():
    game = BlockBlastGame(seed=0)
    set_pieces(game, [None, game.pieces[1], None])
    assert all(idx == 1 for idx, _, _ in game.get_valid_actions())


# ---------------------------------------------------------------------- #
# state snapshot
# ---------------------------------------------------------------------- #

def test_get_state_returns_copies():
    game = BlockBlastGame(seed=0)
    state = game.get_state()
    state["board"][0, 0] = 9
    assert game.board[0, 0] == EMPTY
    assert state["score"] == game.score
    assert state["used"] == [False, False, False]
