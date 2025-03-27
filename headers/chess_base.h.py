import copy
import math
import numpy as np
from enum import Enum
from typing import Any, List, Tuple, Dict

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

    def FEN_notation_letter(self) -> str:
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
            Piece.SPACE: ' '
        }[self]

    @staticmethod
    def from_FEN_notation_letter(letter: str) -> Any:
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
            ' ': Piece.SPACE
        }[letter]


class Board:
    def __init__(self,
                 FEN: str="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
                 r: int = 1,
                 p: 1 | -1 = 1) -> None:
        self.board = [[Piece.SPACE for _ in range(8)] for _ in range(8)]
        FEN = FEN.split('/')
        for i in range(8):
            j = 0
            for c in FEN[i]:
                if c.isnumeric():
                    j += int(c)
                else:
                    self.board[i][j] = Piece.from_FEN_notation_letter(c)
                    j += 1
        self.round = r
        self.player = p
        self.history = []

    def __str__(self):
        return '\n'.join([' '.join([str(j.value) for j in i]) for i in self.board])

    @property
    def FEN(self) -> str:
        fen_rows = []
        for row in self.board:
            empty_count = 0
            fen_row = ''
            for square in row:
                if square == Piece.SPACE:
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += square.FEN_notation_letter()
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_rows.append(fen_row)
        return '/'.join(fen_rows)

    def can_move(self,
                 fr: Tuple[int, int],
                 to: Tuple[int, int]) -> bool:
        x1, y1 = fr
        x2, y2 = to
        piece = self.board[x1][y1]
        des = self.board[x2][y2]
        if piece.value*self.player <= 0 or des.value*self.player > 0:
            return False
        raise NotImplementedError

    def _move(self,
              fr: Tuple[int, int],
              to: Tuple[int, int]) -> None:
        x1, y1 = fr
        x2, y2 = to
        self.history.append(copy.deepcopy(self.board))
        self.board[x2][y2] = self.board[x1][y1]
        self.board[x1][y1] = Piece.SPACE

    def move(self, mov: str) -> None:
        raise NotImplementedError
        self._move(...)
        self._move(...)
        self.round += 1
        self.player *= -1

    def available_moves(self) -> List[Tuple[int, int]]:
        raise NotImplementedError


if __name__ == '__main__':
    b = Board()
    print(b.FEN)
    a = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
    print(a.FEN)
