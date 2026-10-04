"""End-to-end semantics tests for observations, actions, masks and demos."""

import numpy as np

from agents.solver_agent import SolverAgent
from environment.block_blast_env import (
    NUM_ACTIONS,
    build_action_mask,
    decode_action,
    encode_action,
)
from environment.observations import build_observation
from game import BOARD_SIZE, BlockBlastGame
from helpers import make_piece, set_pieces
from training.demos import DemoDataset, DemoRecorder


def brute_can_place(board, piece, row, col):
    for r, c in piece.cells:
        rr, cc = row + r, col + c
        if rr < 0 or rr >= BOARD_SIZE or cc < 0 or cc >= BOARD_SIZE:
            return False
        if board[rr, cc] != 0:
            return False
    return True


def brute_action_mask(game):
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for slot, piece in enumerate(game.pieces):
        if piece is None:
            continue
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if brute_can_place(game.board, piece, row, col):
                    mask[encode_action(slot, row, col)] = True
    return mask


def test_piece_permutation_preserves_slot_action_semantics():
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    board[7, 7] = 4
    piece_a = make_piece([(0, 0)], color=1, name="single")
    piece_b = make_piece([(0, 0), (0, 1)], color=2, name="domino_h")
    piece_c = make_piece([(0, 0), (1, 0), (1, 1)], color=3, name="corner")

    game_a = BlockBlastGame(seed=0)
    game_a.board[:] = board
    set_pieces(game_a, [piece_a, piece_b, piece_c])

    game_b = BlockBlastGame(seed=0)
    game_b.board[:] = board
    set_pieces(game_b, [piece_c, piece_a, piece_b])

    obs_a = build_observation(game_a, "basic")
    obs_b = build_observation(game_b, "basic")
    np.testing.assert_array_equal(obs_a[0], obs_b[0])
    np.testing.assert_array_equal(obs_b[1], piece_c.mask())
    np.testing.assert_array_equal(obs_b[2], piece_a.mask())
    np.testing.assert_array_equal(obs_b[3], piece_b.mask())

    # Same action id slot now means the piece currently in that slot.
    assert decode_action(encode_action(0, 2, 3)) == (0, 2, 3)
    assert game_b.pieces[0] is piece_c
    result = game_b.place_piece(0, 2, 3)
    assert result.success
    for r, c in piece_c.cells:
        assert game_b.board[2 + r, 3 + c] == piece_c.color


def test_enhanced_piece_legal_channels_follow_piece_slots():
    piece_a = make_piece([(0, 0)], color=1, name="single")
    piece_b = make_piece([(0, 0), (0, 1)], color=2, name="domino_h")
    piece_c = make_piece([(0, 0), (1, 0), (1, 1)], color=3, name="corner")

    game = BlockBlastGame(seed=0)
    game.board[:] = 0
    set_pieces(game, [piece_c, piece_a, piece_b])
    obs = build_observation(game, "enhanced_piece_legal")

    assert obs.shape == (10, 8, 8)
    # On an empty board, each slot-specific legal map covers exactly the
    # cells that can be covered by that slot's piece, not by "any" piece.
    for slot, piece in enumerate(game.pieces):
        expected = np.zeros((8, 8), dtype=np.float32)
        for action_id, legal in enumerate(build_action_mask(game)):
            p, row, col = decode_action(action_id)
            if not legal or p != slot:
                continue
            for r, c in piece.cells:
                expected[row + r, col + c] += 1.0
        expected /= expected.max()
        np.testing.assert_allclose(obs[7 + slot], expected)


def test_action_mask_matches_independent_bruteforce_across_states():
    rng = np.random.default_rng(12)
    for seed in range(10):
        game = BlockBlastGame(seed=seed)
        for _ in range(8):
            if game.is_game_over():
                break
            np.testing.assert_array_equal(build_action_mask(game), brute_action_mask(game))
            valid = np.flatnonzero(build_action_mask(game))
            game.place_piece(*decode_action(int(rng.choice(valid))))
        np.testing.assert_array_equal(build_action_mask(game), brute_action_mask(game))


def test_consumed_piece_slot_has_no_observation_or_legal_actions():
    game = BlockBlastGame(seed=0)
    set_pieces(game, [
        make_piece([(0, 0)], color=1),
        make_piece([(0, 0), (0, 1)], color=2),
        make_piece([(0, 0), (1, 0)], color=3),
    ])
    assert game.place_piece(1, 0, 0).success

    obs = build_observation(game, "enhanced_piece_legal")
    mask = build_action_mask(game).reshape(3, 8, 8)
    assert obs[2].sum() == 0
    assert obs[8].sum() == 0
    assert not mask[1].any()
    assert mask[0].any() and mask[2].any()


def test_solver_demo_label_is_pre_action_and_current_slot(tmp_path):
    game = BlockBlastGame(seed=3)
    solver = SolverAgent(beam_width=8)
    action = solver.act(game)
    before_obs = build_observation(game)
    before_mask = build_action_mask(game)

    with DemoRecorder(tmp_path, source="solver") as rec:
        rec.begin_step(game, action)
        result = game.place_piece(*action)
        rec.end_step(result)
        rec.finish_game()

    data = DemoDataset(tmp_path).load(sources=["solver"])
    action_id = encode_action(*action)
    assert int(data["actions"][0]) == action_id
    np.testing.assert_array_equal(data["observations"][0], before_obs.astype(bool))
    np.testing.assert_array_equal(data["action_masks"][0], before_mask)
    assert data["action_masks"][0, action_id]
    # After the move, the selected slot is consumed, so a shifted post-action
    # label would not still describe the same current slot.
    selected_slot = action[0]
    assert game.pieces[selected_slot] is None or result.pieces_refreshed


def test_enhanced_demo_observations_preserve_continuous_channels(tmp_path):
    game = BlockBlastGame(seed=5)
    # Create a nontrivial board so enhanced channels contain fractional values.
    for _ in range(4):
        action = SolverAgent(beam_width=8).act(game)
        game.place_piece(*action)
    action = SolverAgent(beam_width=8).act(game)
    expected = build_observation(game, "enhanced")
    assert np.any((expected[4:] > 0) & (expected[4:] < 1))

    with DemoRecorder(tmp_path, source="solver", observation_profile="enhanced") as rec:
        rec.begin_step(game, action)
        rec.end_step(game.place_piece(*action))
        rec.finish_game()

    data = DemoDataset(tmp_path).load(sources=["solver"])
    assert data["observations"].dtype == np.float32
    np.testing.assert_allclose(data["observations"][0], expected)
