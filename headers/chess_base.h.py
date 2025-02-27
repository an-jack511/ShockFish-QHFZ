from enum import Enum
from typing import Any, List, Tuple, Dict


class piece(Enum):
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
            piece.WHITE_KING: 'K',
            piece.WHITE_QUEEN: 'Q',
            piece.WHITE_BISHOP: 'B',
            piece.WHITE_KNIGHT: 'N',
            piece.WHITE_ROOK: 'R',
            piece.WHITE_PAWN: 'P',
            piece.BLACK_KING: 'k',
            piece.BLACK_QUEEN: 'q',
            piece.BLACK_BISHOP: 'b',
            piece.BLACK_KNIGHT: 'n',
            piece.BLACK_ROOK: 'r',
            piece.BLACK_PAWN: 'p',
            piece.SPACE: ' '
        }[self]

    @staticmethod
    def from_FEN_notation_letter(letter: str) -> Any:
        return {
            'K': piece.WHITE_KING,
            'Q': piece.WHITE_QUEEN,
            'B': piece.WHITE_BISHOP,
            'N': piece.WHITE_KNIGHT,
            'R': piece.WHITE_ROOK,
            'P': piece.WHITE_PAWN,
            'k': piece.BLACK_KING,
            'q': piece.BLACK_QUEEN,
            'b': piece.BLACK_BISHOP,
            'n': piece.BLACK_KNIGHT,
            'r': piece.BLACK_ROOK,
            'p': piece.BLACK_PAWN,
            ' ': piece.SPACE
        }[letter]


class board:
    def __init__(self, FEN: str = "") -> None:
        if FEN != "":
            self.board = [[piece.SPACE for _ in range(8)] for _ in range(8)]
            FEN = FEN.split('/')
            for i in range(8):
                j = 0
                for c in FEN[i]:
                    if c.isnumeric():
                        j += int(c)
                    else:
                        self.board[i][j] = piece.from_FEN_notation_letter(c)
                        j += 1
        self.turn = 1
        self.player = 1
        self.board = [[piece.SPACE for _ in range(8)] for _ in range(8)]
        for i in range(8):
            self.board[1][i] = piece.WHITE_PAWN
            self.board[6][i] = piece.BLACK_PAWN
        self.board[0][0] = piece.WHITE_ROOK
        self.board[0][7] = piece.WHITE_ROOK
        self.board[7][0] = piece.BLACK_ROOK
        self.board[7][7] = piece.BLACK_ROOK
        self.board[0][1] = piece.WHITE_KNIGHT
        self.board[0][6] = piece.WHITE_KNIGHT
        self.board[7][1] = piece.BLACK_KNIGHT
        self.board[7][6] = piece.BLACK_KNIGHT
        self.board[0][2] = piece.WHITE_BISHOP
        self.board[0][5] = piece.WHITE_BISHOP
        self.board[7][2] = piece.BLACK_BISHOP
        self.board[7][5] = piece.BLACK_BISHOP
        self.board[0][3] = piece.WHITE_QUEEN
        self.board[0][4] = piece.WHITE_KING
        self.board[7][3] = piece.BLACK_QUEEN
        self.board[7][4] = piece.BLACK_KING

    # def __init__
    def __str__(self):
        return '\n'.join([' '.join([str(j.value) for j in i]) for i in self.board])

    @property
    def FEN(self) -> str:
        fen_rows = []
        for row in self.board:
            empty_count = 0
            fen_row = ''
            for square in row:
                if square == piece.SPACE:
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


if __name__ == '__main__':
    b = board()
    print(b.FEN)
    a = board("rnbqkbnr/pppppppp/8/PPPPPPPP/8/8/PPPPPPPP/RNBQKBNR")
    print(a.FEN)