from headers.utils import *


class Node:
    def __init__(self,
                 index: int | None = None,
                 parent: int | None = None,
                 move: Any = None,
                 player: int | None = None,
                 c_puct: int = 1,
                 p: int = 1):
        super().__init__()
        self.id = index
        self.parent = parent
        self.children = []
        self.player = player
        self.move = move
        self.N = 0  # total number of visits
        self.W = 0  # total value of node
        self.Q = 0  # average value of node
        self.P = p  # prior probability (analyze by neural network) [!]
        self.c_puct = c_puct  # [!]

    def update(self, result):
        self.N += 1
        self.W += result
        self.Q = self.W / self.N

    def select(self, state):
        if self.Endgame_status(self.player, state.board) != -2:
            return self
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


class MonteCarlo:
    def __init__(self,
                 n_iter: int = 2000):
        self.node = []
        self.n_iter = n_iter

    def __len__(self) -> int:
        return len(self.node)

    def expand(self,
               idx: int,  # node index
               board: chess.Board):
        node = self.node[idx]
        for mov in board.legal_moves:
            if mov in [self.node[i].move for i in node.children]:
                continue
            i = len(self)
            nxt = Node(i, idx, mov, -node.player)
            node.children.append(i)
            self.node.append(nxt)

            print("Move: ", mov)  # debug

    def search(self, state, player):
        for i in range(self.n_iter):
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
