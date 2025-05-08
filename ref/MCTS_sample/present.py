from TicTacToe import TicTacToe
from MonteCarloTreeSearch import Node, MonteCarlo


class Game(TicTacToe):

    def __init__(self):
        super().__init__()
                      
    def human(self, p):
        print("Your turn")
        row, col = input("Enter row and column: ").split()
        row, col = int(row), int(col)  # 0-based
        if row < 3 and col < 3 and self.board[row][col] == 0:
            self.move(p, (row, col))
            self.print_board()
        else:
            print("Invalid move")
            return self.human(p)

    def computer(self, p, it):
        print("Computer's turn")
        computer_mov = MonteCarlo(Node(player=p), n_iter=it).search(self.clone(), p)
        self.move(p, computer_mov)
        self.print_board()

    def play(self, p, it):
        if p == 1:
            self.human(p)
        else:
            self.computer(p, it)
        # self.print_board()
        if self.judge_win(p, self.board):
            print("Player", p, "wins")
            return True
        if self.judge_tie(self.board):
            print("Draw")
            return True
        return False

    def round(self, p, it):
        while not self.play(p, it):
            p = -p
            

if __name__ == '__main__':
    game = Game()
    player = int(input("Choose to be (1/-1) move: "))
    n_iter = int(input("Choose the time for the computer to think: "))
    game.round(player, n_iter)
    # game.print_board()
