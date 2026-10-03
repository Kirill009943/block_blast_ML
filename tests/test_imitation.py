"""Tests for the imitation-learning system: dataset, recording, model,
agent and augmentation."""

import numpy as np
import pytest

from agents import HeuristicAgent, RandomAgent
from environment.block_blast_env import (
    NUM_ACTIONS,
    build_action_mask,
    encode_action,
)
from environment.observations import build_observation
from game import BlockBlastGame
from game.pieces import Piece
from training.demos import (
    DemoDataset,
    DemoRecorder,
    _pack,
    _unpack,
)


def record_games(directory, source="heuristic", games=3, seed=100, flush_games=25):
    """Record ``games`` full games (random agent: fast) into ``directory``."""
    with DemoRecorder(directory, source=source, flush_games=flush_games) as rec:
        for i in range(games):
            game = BlockBlastGame(seed=seed + i)
            agent = RandomAgent(seed=seed + i)
            while not game.is_game_over():
                action = agent.act(game)
                rec.begin_step(game, action)
                rec.end_step(game.place_piece(*action))
            rec.finish_game()
    return directory


# ---------------------------------------------------------------------- #
# dataset
# ---------------------------------------------------------------------- #

def test_bit_packing_roundtrip():
    bits = np.random.default_rng(0).integers(0, 2, size=(7, 192)).astype(bool)
    assert np.array_equal(_unpack(_pack(bits), 192), bits)


def test_record_save_load_roundtrip(tmp_path):
    record_games(tmp_path, games=2)
    data = DemoDataset(tmp_path).load()
    n = len(data["actions"])
    assert n > 0
    assert data["observations"].shape == (n, 4, 8, 8)
    assert data["observations"].dtype == bool
    assert data["action_masks"].shape == (n, NUM_ACTIONS)
    # every recorded action is legal in its stored mask
    assert data["action_masks"][np.arange(n), data["actions"]].all()
    # step/score bookkeeping is coherent
    assert (data["score_before"] >= 0).all()
    assert data["game_over"].sum() == 2  # one per game


def test_multiple_games_and_sources(tmp_path):
    record_games(tmp_path / "shared", games=2, source="heuristic")
    with DemoRecorder(tmp_path / "shared", source="human") as rec:
        game = BlockBlastGame(seed=1)
        agent = RandomAgent(seed=1)
        while not game.is_game_over():
            action = agent.act(game)
            rec.begin_step(game, action)
            rec.end_step(game.place_piece(*action))
        rec.finish_game()

    data = DemoDataset(tmp_path / "shared").load()
    assert len(np.unique(data["game_ids"])) == 3
    assert set(np.unique(data["steps"])) >= {0}

    only_human = DemoDataset(tmp_path / "shared").load(sources=["human"])
    assert (only_human["sources"] == only_human["sources"][0]).all()
    assert len(only_human["actions"]) < len(data["actions"])

    stats = DemoDataset(tmp_path / "shared").stats()
    assert stats["total_games"] == 3
    assert stats["sources"]["human"]["games"] == 1


def test_deterministic_loading(tmp_path):
    record_games(tmp_path, games=2)
    first = DemoDataset(tmp_path).load()
    second = DemoDataset(tmp_path).load()
    for key in ("observations", "actions", "action_masks", "game_ids"):
        assert np.array_equal(first[key], second[key])


def test_train_val_split_by_game(tmp_path):
    record_games(tmp_path, games=10)
    dataset = DemoDataset(tmp_path)
    data = dataset.load()
    train_idx, val_idx = dataset.game_split(
        len(data["actions"]), data["game_ids"], val_fraction=0.2, seed=0)
    assert len(train_idx) and len(val_idx)
    assert not set(train_idx) & set(val_idx)
    # no game appears in both subsets
    assert not set(data["game_ids"][train_idx]) & set(data["game_ids"][val_idx])
    # deterministic
    again = dataset.game_split(
        len(data["actions"]), data["game_ids"], val_fraction=0.2, seed=0)
    assert np.array_equal(train_idx, again[0]) and np.array_equal(val_idx, again[1])


def test_game_ids_continue_across_sessions(tmp_path):
    record_games(tmp_path, games=2)
    record_games(tmp_path, games=1)  # second recorder, same directory
    data = DemoDataset(tmp_path).load()
    assert sorted(np.unique(data["game_ids"])) == [0, 1, 2]


# ---------------------------------------------------------------------- #
# recording semantics
# ---------------------------------------------------------------------- #

def test_observation_recorded_before_state_change(tmp_path):
    game = BlockBlastGame(seed=0)
    agent = HeuristicAgent()
    boards_before, boards_after = [], []
    with DemoRecorder(tmp_path, source="heuristic") as rec:
        for _ in range(5):
            action = agent.act(game)
            boards_before.append(game.board.copy())
            rec.begin_step(game, action)
            result = game.place_piece(*action)
            rec.end_step(result)
            boards_after.append(game.board.copy())
        rec.finish_game()

    data = DemoDataset(tmp_path).load()
    for i in range(5):
        recorded_board = data["observations"][i][0]
        # stored channel 0 is the occupancy BEFORE the move
        assert np.array_equal(recorded_board, boards_before[i] != 0)
        # and it differs from the post-move board (something was placed)
        assert not np.array_equal(recorded_board, boards_after[i] != 0)


def test_invalid_action_is_rejected(tmp_path):
    game = BlockBlastGame(seed=0)
    game.board[:] = 1
    game.board[0, 0] = 0  # only a 1-cell piece could fit at (0, 0)
    game._valid_cache = None
    big = Piece(type_id=0, name="big",
                cells=((0, 0), (0, 1), (1, 0), (1, 1)), color=1)
    game.pieces = [big, None, None]
    with DemoRecorder(tmp_path, source="human") as rec:
        with pytest.raises(ValueError):
            rec.begin_step(game, (0, 0, 0))  # 2x2 square does not fit
    assert rec.pending_samples == 0


def test_game_over_recorded_correctly(tmp_path):
    record_games(tmp_path, games=3)
    data = DemoDataset(tmp_path).load()
    for game_id in np.unique(data["game_ids"]):
        sel = data["game_ids"] == game_id
        assert data["game_over"][sel].sum() == 1  # exactly one terminal step
        assert data["steps"][sel].max() == sel.sum() - 1  # steps are 0..n-1


# ---------------------------------------------------------------------- #
# autonomous collection
# ---------------------------------------------------------------------- #

def test_collect_multiple_games(tmp_path):
    from training.collect_demos import collect

    totals = collect("heuristic", 3, dataset_dir=str(tmp_path), base_seed=500,
                     progress=False)
    assert totals["games"] == 3
    data = DemoDataset(tmp_path).load()
    assert len(data["actions"]) == totals["samples"] == totals["moves"]


def test_recording_does_not_alter_gameplay(tmp_path):
    def play(record):
        game = BlockBlastGame(seed=5)
        agent = HeuristicAgent()
        rec = DemoRecorder(tmp_path / "rec", source="heuristic") if record else None
        while not game.is_game_over():
            action = agent.act(game)
            if rec:
                rec.begin_step(game, action)
            result = game.place_piece(*action)
            if rec:
                rec.end_step(result)
        if rec:
            rec.finish_game()
            rec.close()
        return game.score, game.moves, game.total_lines_cleared

    assert play(record=True) == play(record=False)


def test_collection_flushes_without_ram_growth(tmp_path):
    with DemoRecorder(tmp_path, source="heuristic", flush_games=1) as rec:
        for i in range(5):
            game = BlockBlastGame(seed=700 + i)
            agent = RandomAgent(seed=i)
            while not game.is_game_over():
                action = agent.act(game)
                rec.begin_step(game, action)
                rec.end_step(game.place_piece(*action))
            rec.finish_game()
            # flush_games=1: nothing buffered beyond the current game
            assert len(rec._shard_samples) == 0
    shards = list(tmp_path.glob("shard_*.npz"))
    assert len(shards) == 5


# ---------------------------------------------------------------------- #
# augmentation
# ---------------------------------------------------------------------- #

from training.augment import (
    TRANSFORMS,
    augment_dataset,
    transform_observation,
    transform_sample,
)


def _flipped_piece(piece, transform):
    cells = list(piece.cells)
    if transform in ("hflip", "rot180"):
        width = piece.width
        cells = [(r, width - 1 - c) for r, c in cells]
    if transform in ("vflip", "rot180"):
        height = piece.height
        cells = [(height - 1 - r, c) for r, c in cells]
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    cells = tuple(sorted((r - min_r, c - min_c) for r, c in cells))
    return Piece(type_id=piece.type_id, name=piece.name, cells=cells,
                 color=piece.color)


def _flipped_board(board, transform):
    out = board
    if transform in ("hflip", "rot180"):
        out = out[:, ::-1]
    if transform in ("vflip", "rot180"):
        out = out[::-1, :]
    return np.ascontiguousarray(out)


def _midgame_state(seed=3, moves=6):
    game = BlockBlastGame(seed=seed)
    agent = HeuristicAgent()
    for _ in range(moves):
        if game.is_game_over():
            break
        game.place_piece(*agent.act(game))
    return game


def test_double_transform_is_identity():
    game = _midgame_state()
    obs = build_observation(game)
    mask = build_action_mask(game)
    action = int(np.flatnonzero(mask)[0])
    for transform in TRANSFORMS[1:]:
        obs2, action2, mask2 = transform_sample(obs, action, mask, transform)
        obs3, action3, mask3 = transform_sample(obs2, action2, mask2, transform)
        assert np.array_equal(obs3, obs)
        assert action3 == action
        assert np.array_equal(mask3, mask)


@pytest.mark.parametrize("transform", TRANSFORMS[1:])
def test_transform_matches_engine(transform):
    """Transformed (obs, mask) must equal the engine's own encoding of the
    flipped game state — the strongest correctness guarantee available."""
    game = _midgame_state()
    obs = build_observation(game)
    mask = build_action_mask(game)
    valid = np.flatnonzero(mask)
    rng = np.random.default_rng(0)
    actions = rng.choice(valid, size=min(10, len(valid)), replace=False)

    flipped = BlockBlastGame(seed=0)
    flipped.board[:] = _flipped_board(game.board, transform)
    flipped.pieces = [
        _flipped_piece(p, transform) if p is not None else None
        for p in game.pieces
    ]
    flipped._valid_cache = None
    flipped._playable_types_cache = None

    expected_obs = build_observation(flipped)
    expected_mask = build_action_mask(flipped)

    transformed_obs = transform_observation(obs, transform)
    assert np.allclose(transformed_obs, expected_obs)

    for action in actions:
        _, new_action, new_mask = transform_sample(obs, int(action), mask, transform)
        assert np.array_equal(new_mask, expected_mask)
        assert new_mask[new_action], "augmented action must remain valid"


def test_augmentation_rejects_enhanced_observation():
    game = _midgame_state()
    obs = build_observation(game, "enhanced")
    with pytest.raises(ValueError):
        transform_observation(obs, "hflip")


def test_augment_dataset_stacks_copies(tmp_path):
    record_games(tmp_path, games=1)
    data = DemoDataset(tmp_path).load()
    n = len(data["actions"])
    augmented = augment_dataset(data)
    assert len(augmented["actions"]) == n * len(TRANSFORMS)
    assert len(augmented["observations"]) == n * len(TRANSFORMS)
    assert len(augmented["game_ids"]) == n * len(TRANSFORMS)
    # transformed samples keep their masks valid
    assert augmented["action_masks"][
        np.arange(n * len(TRANSFORMS)), augmented["actions"]
    ].all()


# ---------------------------------------------------------------------- #
# imitation model + agent (need torch)
# ---------------------------------------------------------------------- #

torch = pytest.importorskip("torch")

from agents.imitation_agent import ImitationAgent
from training.imitation_model import (
    ImitationPolicy,
    load_policy,
    save_checkpoint,
)


def _tiny_checkpoint(tmp_path):
    """Train 1 epoch on a tiny recorded dataset; return the .pt path."""
    from training.train_imitation import parse_args, train

    record_games(tmp_path / "demos", games=4, seed=900)
    args = parse_args([
        "--data", str(tmp_path / "demos"), "--source", "all",
        "--epochs", "1", "--batch-size", "32", "--val-fraction", "0.25",
        "--out", str(tmp_path / "bc.pt"),
    ])
    return train(args)


def test_model_forward_and_masking():
    game = _midgame_state()
    policy = ImitationPolicy(observation_channels=4)
    obs = torch.as_tensor(build_observation(game)[None], dtype=torch.float32)
    mask = torch.as_tensor(build_action_mask(game)[None])
    logits = policy(obs)
    assert logits.shape == (1, NUM_ACTIONS)
    masked = policy.masked_logits(obs, mask)
    probs = torch.softmax(masked, dim=-1)
    # invalid actions carry no probability mass
    assert float(probs[0][~mask[0]].sum()) == 0.0
    assert int(probs[0].argmax()) in set(np.flatnonzero(mask[0]).tolist())


def test_checkpoint_roundtrip(tmp_path):
    policy = ImitationPolicy(observation_channels=4)
    path = save_checkpoint(policy, tmp_path / "m.pt", config={"epochs": 0})
    loaded, meta = load_policy(path, device="cpu")
    assert meta["observation_channels"] == 4
    for key, value in policy.state_dict().items():
        assert torch.equal(loaded.state_dict()[key].cpu(), value.cpu())


def test_tiny_training_run_produces_agent(tmp_path):
    path = _tiny_checkpoint(tmp_path)
    assert path.exists()
    agent = ImitationAgent(path)
    game = _midgame_state()
    action = agent.act(game)
    assert action in game.get_valid_actions()


def test_agent_deterministic_and_stochastic(tmp_path):
    path = _tiny_checkpoint(tmp_path)
    game = _midgame_state()
    det = ImitationAgent(path, deterministic=True)
    assert det.act(game) == det.act(game)

    sto = ImitationAgent(path, deterministic=False, seed=0)
    for _ in range(20):
        assert sto.act(game) in game.get_valid_actions()

    info = det.decide(game)
    assert info.extra["top_candidates"]
    assert 0.0 < info.extra["action_probability"] <= 1.0


def test_ppo_state_dict_compatibility(tmp_path):
    """BC weights must load into a MaskablePPO policy with strict=False."""
    sb3_contrib = pytest.importorskip("sb3_contrib")
    from environment.block_blast_env import BlockBlastEnv
    from training.cnn_extractor import BlockBlastCNN

    policy = ImitationPolicy(observation_channels=4)
    env = BlockBlastEnv()
    model = sb3_contrib.MaskablePPO(
        "MlpPolicy", env,
        policy_kwargs=dict(
            features_extractor_class=BlockBlastCNN,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
        ),
        n_steps=8, batch_size=8, n_epochs=1,
    )
    before = model.policy.state_dict()
    model.policy.load_state_dict(policy.state_dict(), strict=False)
    after = model.policy.state_dict()
    # policy side transferred…
    assert torch.equal(after["action_net.weight"].cpu(), policy.action_net.weight)
    # …value head untouched
    assert torch.equal(after["value_net.weight"].cpu(), before["value_net.weight"].cpu())
