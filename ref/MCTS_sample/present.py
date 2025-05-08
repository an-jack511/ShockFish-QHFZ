from TicTacToe import TicTacToe
from MonteCarloTreeSearch import Node, MonteCarlo


class Game(TicTacToe):

    def __init__(self):
        super().__init__()
                      
    def Human(self, player):
        print("Your turn")
        row, col = input("Enter row and column: ").split()
        row, col = int(row), int(col)  # 0-based
        if row < 3 and col < 3 and self.board[row][col] == 0:
            self.Move(player, (row, col))
            self.print_board()
        else:
            print("Invalid move")
            return self.Human(player)

    def Computer(self, player, rt):
        print("Computer's turn")
        computer_mov = MonteCarlo(Node(player = player), n_iter= rt).search(self.clone(), player)
        self.Move(player, computer_mov)
        self.print_board()

    def Play(self, player, rt):
        if player == 1:
            self.Human(player)
        else:
            self.Computer(player, rt)
        # self.print_board()
        if self.Judge_win(player, self.board):
            print("Player", player, "wins")
            return True
        if self.Judge_tie(self.board):
            print("Draw")
            return True
        return False

    def Round(self, player, rt):
        while not self.Play(player, rt):
            player = -player
            

if __name__ == '__main__':
    game = Game()
    player = int(input("Choose to be (1/-1) move: "))
    rt = int(input("Choose the time for the computer to think: "))
    game.Round(player, rt)
    # game.print_board()
