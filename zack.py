import numpy as np


class Player:
    
    def __init__(self, rows, cols, connect_number, 
                 timeout_setup, timeout_move, max_invalid_moves, 
                 cylinder):
        self.rows = rows
        self.cols = cols
        self.connect_number = connect_number
        self.timeout_setup = timeout_setup
        self.timeout_move = timeout_move
        self.max_invalid_moves = max_invalid_moves
        self.cylinder = cylinder

    def setup(self,piece_color):
        """
        This method will be called once at the beginning of the game so the player
        can conduct any setup before the move timer begins. The setup method is
        also timed.
        """
        self.piece_color = 1 if piece_color == '+' else -1
        self.opp_color = -1*self.piece_color
        self.depth=4

    def heuristic(self, board: np.ndarray):
        directions = [(0,1),(1,0),(1,1),(1,-1)]
        val = 0
        for i in range(self.rows):
            for j in range(self.cols):
                for dir in directions:
                    if i+self.connect_number*dir[0]>self.rows:
                        continue
                    locations = [(i+k*dir[0], (j+k*dir[1]) % self.cols) for k in range(self.connect_number)]
                    pieces = [board[loc] for loc in locations]
                    count_own = pieces.count(self.piece_color)
                    count_opponent = pieces.count(self.opp_color)

                    if count_own==self.connect_number:
                        val += 100000
                    elif count_opponent==self.connect_number:
                        val -= 100000
                    elif count_own>0 and count_opponent>0:
                        continue
                    elif count_own>0:
                        missing_locs = [locations[k] for k in range(self.connect_number) if pieces[k]==0]
                        holes_below = sum([sum(board[loc[0]+1:, loc[1]]==0) for loc in missing_locs])
                        val += pow(10, count_own) - pow(10, count_own-1)* pow(holes_below, 1/2)
                    elif count_opponent>0:
                        missing_locs = [locations[k] for k in range(self.connect_number) if pieces[k]==0]
                        holes_below = sum([sum(board[loc[0]+1:, loc[1]]==0) for loc in missing_locs])
                        val -= pow(10, count_opponent) - pow(10, count_opponent-1)* pow(holes_below, 1/2)
        return val
    def check_winner(self, board: np.ndarray, lastTurn: int):
        directions = [(0,1),(1,0),(1,1),(1,-1)]
        for i in range(self.rows):
            for j in range(self.cols):
                for dir in directions:
                    if i+self.connect_number*dir[0]>self.rows:
                        continue
                    locations = [(i+k*dir[0], (j+k*dir[1]) % self.cols) for k in range(self.connect_number)]
                    pieces = [board[loc] for loc in locations]
                    if pieces.count(lastTurn)==self.connect_number:
                        return True
        return False
    def minimax(self, board: np.ndarray, depth: int, maximizing_player: bool, alpha: float, beta: float):
        if self.check_winner(board, self.opp_color if maximizing_player else self.piece_color):
            return (float('-inf') if maximizing_player else float('inf')), None
        if depth==0:
            return self.heuristic(board), None
        valid_moves = [col for col in range(self.cols) if board[0, col] == 0]
        if not valid_moves:
            return self.heuristic(board), None
        if maximizing_player:
            max_eval = float('-inf')
            best_move = valid_moves[0]
            for move in valid_moves:
                for row in range(self.rows-1, -1, -1):
                    if board[row, move] == 0:
                        board[row, move] = self.piece_color
                        break
                eval, _ = self.minimax(board, depth-1, False, alpha, beta)
                board[row, move] = 0
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = float('inf')
            best_move = None
            for move in valid_moves:
                for row in range(self.rows-1, -1, -1):
                    if board[row, move] == 0:
                        board[row, move] = self.opp_color
                        break
                eval, _ = self.minimax(board, depth-1, True, alpha, beta)
                board[row, move] = 0
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def play(self, board: np.ndarray):
        """
        Given a 2D array representing the game board, return an integer value (0,1,2,...,number of columns-1) corresponding to
        the column of the board where you want to drop your disc.
        The coordinates of the board increase along the right and down directions. 

        Parameters
        ----------
        board : np.ndarray
            A 2D array where 0s represent empty slots, +1s represent your pieces,
            and -1s represent the opposing player's pieces.

                `index   0   1   2   . column` \\
                `--------------------------` \\
                `0   |   0.  0.  0.  top` \\
                `1   |   -1  0.  0.  .` \\
                `2   |   +1  -1  -1  .` \\
                `.   |   -1  +1  +1  .` \\
                `row |   left        bottom/right`

        Returns
        -------
        integer corresponding to the column of the board where you want to drop your disc.
        """
        if all(all(board[i, j] == 0 for i in range(self.rows)) for j in range(self.cols)):
            return np.random.randint(0, self.cols-1)
        return self.minimax(board, self.depth, True, float('-inf'), float('inf'))[1]

__all__ = ['Player']