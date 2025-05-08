"""
chess stuff using python-chess
by Z.
"""
import copy

import chess

from headers.utils import *


class Board:
    def __init__(self,
                 player: 1 | -1 = None,
                 fen: None | str = chess.STARTING_FEN):
        self.board = chess.Board(fen)
        self.player = player

    def legal_moves(self) -> List:
        return list(self.board.legal_moves)

    def move(self, move: chess.Move):
        self.board.push(move)

    def repli_move(self, move: chess.Move):
        new = copy.deepcopy(self)
        new.board.push(move)
        return new

    @staticmethod
    def endgame_status(board: chess.Board) -> int | None:
        res = board.outcome()
        if res is not None:
            res = {
                True: 1,    # white
                False: -1,  # black
                None: 0     # draw
            }[res.winner]
        return res

    def __str__(self):
        return str(self.board)


def main():
    pass


if __name__ == "__main__":
    main()
