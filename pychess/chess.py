"""
chess stuff using python-chess
by Z.
"""

from headers.utils import *


class Piece(Enum):
    WHITE_KING = 1
    WHITE_QUEEN = 2
    WHITE_BISHOP = 3
    WHITE_KNIGHT = 4
    WHITE_ROOK = 5
    WHITE_PAWN = 6

    BLACK_KING = -1
    BLACK_QUEEN = -2
    BLACK_BISHOP = -3
    BLACK_KNIGHT = -4
    BLACK_ROOK = -5
    BLACK_PAWN = -6

    SPACE = 0

    def get_fen(self) -> str:
        return {
            Piece.WHITE_KING: 'K',
            Piece.WHITE_QUEEN: 'Q',
            Piece.WHITE_BISHOP: 'B',
            Piece.WHITE_KNIGHT: 'N',
            Piece.WHITE_ROOK: 'R',
            Piece.WHITE_PAWN: 'P',
            Piece.BLACK_KING: 'k',
            Piece.BLACK_QUEEN: 'q',
            Piece.BLACK_BISHOP: 'b',
            Piece.BLACK_KNIGHT: 'n',
            Piece.BLACK_ROOK: 'r',
            Piece.BLACK_PAWN: 'p',
            Piece.SPACE: '.'
        }[self]

    @staticmethod
    def from_fen(letter: str) -> Any:
        return {
            'K': Piece.WHITE_KING,
            'Q': Piece.WHITE_QUEEN,
            'B': Piece.WHITE_BISHOP,
            'N': Piece.WHITE_KNIGHT,
            'R': Piece.WHITE_ROOK,
            'P': Piece.WHITE_PAWN,
            'k': Piece.BLACK_KING,
            'q': Piece.BLACK_QUEEN,
            'b': Piece.BLACK_BISHOP,
            'n': Piece.BLACK_KNIGHT,
            'r': Piece.BLACK_ROOK,
            'p': Piece.BLACK_PAWN,
            '.': Piece.SPACE
        }[letter]


class Board:
    def __init__(self,
                 player: int,
                 fen: None | str = None):
        self.board = chess.Board(fen)
        self.player = player

    def legal_move(self) -> List:
        for move in self.board.legal_moves:
            pass
        raise NotImplementedError

    def is_legal(self, move: str | chess.Move) -> bool:
        return self.board.is_legal(move)


def main():
    pass


if __name__ == "__main__":
    main()
