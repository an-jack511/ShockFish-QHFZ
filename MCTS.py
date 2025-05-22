import chess

from headers.utils import *
from pychess.chess import Board
from model.model import DNN


net = DNN()


class Node(Board):
    def __init__(self,
                 c_puct=1,
                 p=1,
                 player: 1 | -1 = None):
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

    @staticmethod
    def state_to_tensor(state)-> Tensor:
        fen = state.board.fen()
        print(fen)
        raise NotImplementedError

    def expand(self, state):
        for mov in state.legal_moves():
            if mov in self.moves.values():
                continue
            self.children.append(Node(player=-self.player, p=net(self.state_to_tensor(state))[0].item()))
            self.moves[self.children[-1]] = mov
            self.children[-1].parent = self
            self.children[-1].player = -self.player

    def select(self, state):
        if self.endgame_status(state.board) is not None:
            print("!")
            print(state)
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
        return cur_child.select(state.repli_move(self.moves[cur_child]))

    def backup(self, result):
        self.N += 1
        self.W += result
        self.Q = self.W / self.N

    def default_policy(self, state, player):
        res = self.endgame_status(state.board)
        cur = copy.deepcopy(state)
        if res is not None:
            return -res*player
        moves = state.legal_moves()
        while moves:
            mov_idx = random.randint(0, len(moves) - 1)
            mov = moves[mov_idx]  # Fix potential multidimensional issue
            cur.move(mov)
            moves.remove(mov)
            player = -player
            res = self.endgame_status(cur.board)
            print(cur, "\n")
            if res is not None:
                return -res*player
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
            print(i)
            if self.root.endgame_status(state.board) is not None:
                break
            cur = self.root.select(copy.deepcopy(state))
            reward = cur.default_policy(copy.deepcopy(state), player)
            cur.backprop(reward)
            self.root.backup(reward)

        cur_mov, cur_res = None, -np.inf
        for child in self.root.children:
            print(f"Move: {self.root.moves[child]} N={child.N} Q={child.Q}")
            if child.N > cur_res:
                cur_res = child.N
                cur_mov = self.root.moves[child]
        print(f"Selected: {cur_mov}, N={cur_res}")
        return cur_mov


class Game(Board):

    def __init__(self):
        super().__init__()

    def human(self, p):
        print(self)
        print("Your turn")
        while True:
            try:
                mov = input("Enter move (uci): ")
                mov = chess.Move.from_uci(mov)
                self.move(mov)
                print(self)
                break
            except Exception as err:
                print(err)

    def computer(self, p, it):
        print("Computer's turn")
        computer_mov = MonteCarlo(Node(player=p), n_iter=it).search(copy.deepcopy(self), p)
        self.move(computer_mov)

    def play(self, p, it):
        if p == 1:
            self.human(p)
        else:
            self.computer(p, it)
        # self.print_board()
        res = self.endgame_status(self.board)
        if res is None:
            return True
        if res:
            print("Player", p, "wins")
            return False
        print("Draw")
        return False

    def round(self, p, it):
        while self.play(p, it):
            p = -p


def main():
    game = Game()
    player = -1  # int(input("Choose to be (1/-1) move: "))
    n_iter = 1000  # int(input("Choose the time for the computer to think: "))
    game.round(player, n_iter)


if __name__ == "__main__":
    main()
