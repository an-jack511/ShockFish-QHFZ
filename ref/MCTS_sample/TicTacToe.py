from headers.utils import *


class TicTacToe:
    def __init__(self):
        self.cnt_row = 3
        self.cnt_col = 3
        self.las_move = None
        self.board = [[0 for __ in range(self.cnt_col)] for _ in range(self.cnt_row)]
    
    def clone(self):
        new_state = TicTacToe()
        new_state.board = copy.deepcopy(self.board)
        new_state.las_move = self.las_move
        return new_state

    def print_board(self):
        for i in range(self.cnt_row):
            for j in range(self.cnt_col):
                print("O" if self.board[i][j] == 1 else "X" if self.board[i][j] == -1 else "*", end=' ')
            print()
    
    def legal_moves(self):
        moves = []
        for i in range(self.cnt_row):
            for j in range(self.cnt_col):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves

    def repli_move(self, player, move):
        new_state = self.clone()
        new_state.board[move[0]][move[1]] = player
        new_state.las_move = move
        return new_state
    
    def move(self, player, move):
        self.board[move[0]][move[1]] = player
        self.las_move = move
    
    def judge_tie(self, state):
        for i in range(self.cnt_row):
            for j in range(self.cnt_col):
                if state[i][j] == 0:
                    return False
        return True
    
    def judge_win(self, player, state):
        for i in range(self.cnt_row):
            if state[i][0] == state[i][1] == state[i][2] == player:
                return True
        for i in range(self.cnt_col):
            if state[0][i] == state[1][i] == state[2][i] == player:
                return True
        if state[0][0] == state[1][1] == state[2][2] == player:
            return True
        if state[0][2] == state[1][1] == state[2][0] == player:
            return True
        return False
    
    def endgame_status(self, player, state):
        if self.judge_win(player, state):
            return 10
        if self.judge_win(-player, state):
            return -10
        if self.judge_tie(state):
            return 0
        return None
