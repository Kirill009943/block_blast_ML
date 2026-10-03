"""Tests for the baseline and RL agents."""

import numpy as np
import pytest

from agents import HeuristicAgent, RandomAgent
from game import BlockBlastGame
from helpers import make_piece, set_pieces


def play_game(agent, seed=0, max_moves=10_000):
    game = BlockBlastGame(seed=seed)
    while not game.is_game_over() and game.moves < max_moves:
        action = agent.act(game)
        result = game.place_piece(*action)
        assert result.success, f"{agent.name} chose an invalid action"
    return game


def test_random_agent_picks_valid_actions():
    agent = RandomAgent(seed=0)
    game = BlockBlastGame(seed=0)
    for _ in range(50):
        if game.is_game_over():
            break
        action = agent.act(game)
        assert action in game.get_valid_actions()
        game.place_piece(*action)


def test_random_agent_completes_a_game():
    game = play_game(RandomAgent(seed=1), seed=1)
    assert game.is_game_over()
    assert game.score > 0


def test_heuristic_agent_completes_a_game():
    game = play_game(HeuristicAgent(), seed=2)
    assert game.is_game_over()
    assert game.score > 0


def test_heuristic_prefers_line_clear():
    game = BlockBlastGame(seed=0)
    # row 3 almost complete; a vertical domino can complete it at (3, 0)
    for c in range(1, 8):
        game.board[3, c] = 1
    domino_v = make_piece([(0, 0), (1, 0)])
    far_single = make_piece([(0, 0)])
    set_pieces(game, [domino_v, far_single, make_piece([(0, 0), (0, 1)])])

    agent = HeuristicAgent()
    action = agent.act(game)

    # both the domino and the single cell complete row 3 at column 0 —
    # either is correct, what matters is that the line gets cleared
    assert action[1:] == (3, 0)
    result = game.place_piece(*action)
    assert result.lines_cleared == 1


def test_heuristic_beats_random_on_average():
    """Sanity check that the heuristic is a meaningful benchmark."""
    def avg_score(agent_factory, seeds):
        scores = []
        for seed in seeds:
            game = BlockBlastGame(seed=seed)
            agent = agent_factory(seed)
            while not game.is_game_over():
                game.place_piece(*agent.act(game))
            scores.append(game.score)
        return float(np.mean(scores))

    seeds = range(10)
    random_avg = avg_score(lambda s: RandomAgent(seed=s), seeds)
    heuristic_avg = avg_score(lambda s: HeuristicAgent(), seeds)
    assert heuristic_avg > random_avg


def test_heuristic_decide_returns_debug_info():
    game = BlockBlastGame(seed=0)
    info = HeuristicAgent().decide(game)
    assert info.action in game.get_valid_actions()
    assert info.valid_actions == len(game.get_valid_actions())
    assert info.value is not None
    assert info.extra["top_candidates"]


def test_rl_agent_with_untrained_model(tmp_path):
    """Smoke test: a fresh MaskablePPO must still act only via the mask."""
    sb3_contrib = pytest.importorskip("sb3_contrib")
    torch = pytest.importorskip("torch")

    from environment.block_blast_env import BlockBlastEnv
    from agents.rl_agent import RLAgent

    env = BlockBlastEnv()
    model = sb3_contrib.MaskablePPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1)
    path = tmp_path / "untrained.zip"
    model.save(str(path))

    agent = RLAgent(path)
    game = BlockBlastGame(seed=0)
    action = agent.act(game)
    assert action in game.get_valid_actions()
    value = agent.value_estimate(game)
    assert isinstance(value, float)


# ---------------------------------------------------------------------- #
# SolverAgent
# ---------------------------------------------------------------------- #

from agents import SolverAgent


def test_solver_picks_valid_actions():
    game = BlockBlastGame(seed=0)
    agent = SolverAgent(beam_width=8)
    for _ in range(30):
        if game.is_game_over():
            break
        action = agent.act(game)
        assert action in game.get_valid_actions()
        game.place_piece(*action)


def test_solver_survives_long_games():
    """With the board-aware generator the solver should never reach game over."""
    for seed in (0, 1):
        game = play_game(SolverAgent(beam_width=8), seed=seed, max_moves=300)
        assert not game.is_game_over(), f"solver died after {game.moves} moves (seed {seed})"
        assert game.moves == 300


def test_solver_is_deterministic():
    def run(seed):
        game = play_game(SolverAgent(beam_width=8), seed=seed, max_moves=100)
        return game.score, game.moves, game.total_lines_cleared

    assert run(7) == run(7)


def test_solver_finds_line_clear_ordering():
    """Row 0 is one domino short of completion: the solver must place a
    domino at (0,0) — the only move that clears a line."""
    game = BlockBlastGame(seed=0)
    game.board[:] = 0
    for c in range(2, 8):
        game.board[0, c] = 1
    game._valid_cache = None
    domino = make_piece([(0, 0), (0, 1)])
    set_pieces(game, [domino, domino, None])

    agent = SolverAgent(beam_width=8)
    action = agent.act(game)
    assert action[1:] == (0, 0)  # only placement that clears the row
    result = game.place_piece(*action)
    assert result.lines_cleared == 1
    assert not game.is_game_over()


def test_solver_decide_reports_search_info():
    game = BlockBlastGame(seed=0)
    info = SolverAgent(beam_width=8).decide(game)
    assert info.action in game.get_valid_actions()
    assert info.extra["source"] in ("beam", "exhaustive", "greedy")
    assert info.extra["nodes_expanded"] > 0
