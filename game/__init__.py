"""Block Blast game engine package (pure logic, no I/O)."""

from game.game import BOARD_SIZE, EMPTY, BlockBlastGame, GameConfig, MoveResult
from game.pieces import PIECE_TYPES, Piece

__all__ = [
    "BOARD_SIZE",
    "EMPTY",
    "BlockBlastGame",
    "GameConfig",
    "MoveResult",
    "PIECE_TYPES",
    "Piece",
]
