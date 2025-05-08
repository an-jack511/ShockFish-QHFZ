from headers.utils import *
from TicTacToe import TicTacToe


class Node(TicTacToe):
    def __init__(self, c_puct=1, p=1, player=None):
        super().__init__()
        self.parent = None
        self.player = player
        self.children = []
        self.moves = {}
        self.N = 0  # total number of visits
        self.W = 0  # total value of node
        self.Q = 0  # average value of node
        self.P = p  # prior probability (analyze by neural network)
        self.c_puct = c_puct
            
    def expand(self, state):
        for mov in state.legal_moves():
            if mov in self.moves.values():
                continue
            self.children.append(Node(player=-self.player))
            self.moves[self.children[-1]] = mov
            self.children[-1].parent = self
            self.children[-1].player = -self.player
            # print("Move: ", mov) # debug

    def select(self, state):
        if self.endgame_status(self.player, state.board) is not None:
            return self
        cur_child, cur_res = None, -np.inf
        for mov in state.legal_moves():
            if mov not in self.moves.values():
                self.expand(state)
                return self.children[-1]
        for child in self.children:
            if child.N == 0:
                return child
            ucb = child.Q + self.c_puct * child.P * np.sqrt(self.N) / (1 + child.N)
            if ucb > cur_res:
                cur_res = ucb
                cur_child = child
        return cur_child.select(state.repli_move(self.player, self.moves[cur_child]))

    def backup(self, result):
        self.N += 1
        self.W += result
        self.Q = self.W / self.N

    def default_policy(self, state, player):
        cur = copy.deepcopy(state)
        res = self.endgame_status(player, cur.board)
        if res is not None:
            return -res
        moves = cur.legal_moves()
        while moves:
            mov_idx = random.randint(0, len(moves) - 1)
            mov = moves[mov_idx]  # Fix potential multidimensional issue
            cur.move(player, mov)
            moves.remove(mov)
            player = -player
            res = self.endgame_status(player, cur.board)
            if res is not None:
                return -res
        return 0

    def backprop(self, result):
        self.backup(result)
        if self.parent:
            self.parent.backprop(-result)


class MonteCarlo(Node):
    def __init__(self, root, n_iter=2000):
        self.root = root
        self.n_iter = n_iter
    
    def search(self, state, player):
        for i in range(self.n_iter):
            if self.root.endgame_status(player, state.board) is not None:
                break
            cur = self.root.select(copy.deepcopy(state))
            reward = cur.default_policy(copy.deepcopy(state), player)
            cur.backprop(reward)
            self.root.backup(reward)

        cur_mov, cur_res = None, -np.inf
        for child in self.root.children:
            if child.N > cur_res:
                cur_res = child.N
                cur_mov = self.root.moves[child]
        # print("Move: ", cur_mov)
        return cur_mov
