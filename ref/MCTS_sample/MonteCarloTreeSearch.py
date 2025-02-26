import numpy as np
from TicTacToe import TicTacToe

class Node(TicTacToe):
    def __init__(self, c_puct = 1, P = 1):
        super().__init__()
        self.children = []
        self.moves = {}
        self.N = 0 # total number of visits
        self.W = 0 # total value of node
        self.Q = 0 # average value of node
        self.P = P # prior probability (analyze by neural network)
        self.c_puct = c_puct

            
    def expand(self, state, player):
        for mov in self.Get_moves(state):
            self.children.append(Node())
            self.moves[self.children[-1]] = mov
            # print("Move: ", mov) # debug

    def select(self):
        cur_child, cur_res = 1, -np.inf
        for child in self.children:
            puct = child.Q + self.c_puct * child.P * np.sqrt(self.N + 1) / (1 + child.N)
            if(puct > cur_res):
                cur_res = puct
                cur_child = child
        return cur_child

    def backup(self, result):
        self.N += 1
        self.W += result
        self.Q = self.W / self.N

    def simulate(self, state, player):
        cur = state.clone()
        if self.Endgame_status(player, cur.board) != -2:
            return -self.Endgame_status(player, cur.board)
        if not self.children:
            self.expand(cur, player) # here may add the prediction of the neural network of P,V
        nxtnode = self.select()
        reward = nxtnode.simulate(cur.RepliMove(-player, self.moves[nxtnode]), -player)
        self.backup(reward)
        return -reward

class MonteCarlo:
    def __init__(self, root, time = 2000):
        self.root = root
        self.time = time
    
    def search(self, state, player):
        for i in range(self.time):
            self.root.simulate(state, player)

        cur_mov, cur_res = None, -np.inf
        for child in self.root.children:
            if child.N > cur_res:
                cur_res = child.N
                cur_mov = self.root.moves[child]
        # print("Move: ", cur_mov)
        return cur_mov
