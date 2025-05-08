import numpy as np
import random
from TicTacToe import TicTacToe

class Node(TicTacToe):
    def __init__(self, c_puct = 1, P = 1, player = None):
        super().__init__()
        self.parent = None
        self.player = player
        self.children = []
        self.moves = {}
        self.N = 0 # total number of visits
        self.W = 0 # total value of node
        self.Q = 0 # average value of node
        self.P = P # prior probability (analyze by neural network)
        self.c_puct = c_puct
            
    def expand(self, state):
        for mov in state.Get_moves():
            if mov in self.moves.values():
                continue
            self.children.append(Node(player = -self.player))
            self.moves[self.children[-1]] = mov
            self.children[-1].parent = self
            self.children[-1].player = -self.player
            # print("Move: ", mov) # debug

    def select(self, state):
        if self.Endgame_status(self.player, state.board) != -2:
            return self;
        cur_child, cur_res = None, -np.inf
        for mov in state.Get_moves():
            if mov not in self.moves.values():
                self.expand(state)
                return self.children[-1]
        # print("jj")
        for child in self.children:
            # print("Child!!!!") # debug
            if child.N == 0:
                return child
            ucb = child.Q + self.c_puct * child.P * np.sqrt(self.N) / (1 + child.N)
            if ucb > cur_res:
                cur_res = ucb
                cur_child = child
        return cur_child.select(state.RepliMove(self.player, self.moves[cur_child]))

    def backup(self, result):
        self.N += 1
        self.W += result
        self.Q = self.W / self.N

    def default_policy(self, state, player):
        cur = state.clone()
        if self.Endgame_status(player, cur.board) != -2:
            return -self.Endgame_status(player, cur.board)
        moves = cur.Get_moves()
        while moves:
            mov_idx = random.randint(0,len(moves) - 1)
            mov = moves[mov_idx]  # Fix potential multidimensional issue
            cur.Move(player, mov)
            moves.remove(mov)
            player = -player
            if self.Endgame_status(player, cur.board) != -2:
                return -self.Endgame_status(player, cur.board)
        return 0

    def backpropagate(self, result):
        self.backup(result)
        if self.parent:
            self.parent.backpropagate(-result)

class MonteCarlo(Node):
    def __init__(self, root, time = 2000):
        self.root = root
        self.time = time
    
    def search(self, state, player):
        for i in range(self.time):
            if(self.root.Endgame_status(player, state.board) != -2):
                break
            cur = self.root.select(state.clone())
            reward = cur.default_policy(state.clone(), player)
            cur.backpropagate(reward)
            self.root.backup(reward)

        cur_mov, cur_res = None, -np.inf
        for child in self.root.children:
            if child.N > cur_res:
                cur_res = child.N
                cur_mov = self.root.moves[child]
        # print("Move: ", cur_mov)
        return cur_mov
