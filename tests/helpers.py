"""Shared test helpers."""

from game.pieces import Piece


def make_piece(cells, color=1, type_id=0, name="test"):
    return Piece(type_id=type_id, name=name, cells=tuple(cells), color=color)


def set_pieces(game, pieces):
    """Force the piece slots of ``game`` to the given pieces."""
    game.pieces = list(pieces)
    game._valid_cache = None
    game._playable_types_cache = None
