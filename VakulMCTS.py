import time
import math
import numpy as np

from player import Player


class MCTSNode:
    __slots__ = ['parent', 'move', 'N', 'W', 'children', 'untried_moves',
                 'player', 'proven']

    def __init__(self, parent=None, move=None, player=None):
        self.parent = parent
        self.move = move
        self.player = player
        self.N = 0
        self.W = 0.0
        self.children = {}
        self.untried_moves = []
        self.proven = None


class VakulMCTS(Player):
    def __init__(self, rows, cols, connect_number,
                 timeout_setup, timeout_move, max_invalid_moves, cylinder):
        self.rows = rows
        self.cols = cols
        self.connect_number = connect_number
        self.timeout_setup = timeout_setup
        self.timeout_move = timeout_move
        self.max_invalid_moves = max_invalid_moves
        self.cylinder = cylinder

    def setup(self, piece_color):
        self.piece_color = piece_color
        self.my_piece = +1
        self.opponent_piece = -1

    def play(self, game_state: np.ndarray) -> int:
        deadline = time.time() + self.timeout_move - 0.1
        state = game_state.copy()

        legal = self.get_legal_moves(state)
        if not legal:
            return 0
        if len(legal) == 1:
            return legal[0]

        # Check for immediate win
        for col in legal:
            row = self.drop_piece(state, col, self.my_piece)
            if self.check_win(state, row, col, self.my_piece):
                self.undo_piece(state, row, col)
                return col
            self.undo_piece(state, row, col)

        # Check for must-block opponent win
        for col in legal:
            row = self.drop_piece(state, col, self.opponent_piece)
            if self.check_win(state, row, col, self.opponent_piece):
                self.undo_piece(state, row, col)
                return col
            self.undo_piece(state, row, col)

        # MCTS search
        root = MCTSNode(parent=None, move=None, player=self.my_piece)
        root.untried_moves = legal[:]

        _time = time.time
        _iterate = self.mcts_iteration
        while True:
            for _ in range(64):
                _iterate(root, state)
            if _time() >= deadline:
                break

        if not root.children:
            return legal[0]
        return max(root.children.values(), key=lambda c: c.N).move

    # ============ MAIN MCTS ALGORITHM ============

    def mcts_iteration(self, root, state):
        move_stack = []

        # Selection
        node = root
        while not node.untried_moves and node.children:
            if node.proven is not None:
                break
            node = self.select_child(node)
            if node is None:
                break
            row = self.drop_piece(state, node.move, node.parent.player)
            move_stack.append((row, node.move))

        if node is None:
            for r, c in reversed(move_stack):
                self.undo_piece(state, r, c)
            return

        if node.proven is not None:
            self.backpropagate(node, node.proven)
            for r, c in reversed(move_stack):
                self.undo_piece(state, r, c)
            return

        # Expansion
        if not node.untried_moves and not node.children:
            node.untried_moves = self.get_legal_moves(state)

        if node.untried_moves:
            move = node.untried_moves.pop()
            row = self.drop_piece(state, move, node.player)
            move_stack.append((row, move))

            child = MCTSNode(parent=node, move=move, player=-node.player)
            node.children[move] = child
            node = child

            # MCTS-Solver: check if this move wins
            if self.check_win(state, row, move, -node.player):
                result = 1.0 if -node.player == self.my_piece else -1.0
                node.proven = result
                self.backpropagate(node, result)
                self.propagate_proven(node)
                for r, c in reversed(move_stack):
                    self.undo_piece(state, r, c)
                return

        # Simulation: 1-ply tactical check + heuristic evaluation
        result = self.leaf_evaluation(state, node.player)

        # Backpropagation
        self.backpropagate(node, result)

        for r, c in reversed(move_stack):
            self.undo_piece(state, r, c)

    def select_child(self, node):
        best = None
        best_score = -float('inf')
        C = 0.5
        _sqrt = math.sqrt

        fpu = (node.W / node.N - 0.25) if node.N > 0 else 0.0
        log_N = math.log(node.N) if node.N > 0 else 0

        for child in node.children.values():
            if child.proven is not None:
                child_good = (
                    (child.proven == 1.0 and child.parent.player == self.my_piece) or
                    (child.proven == -1.0 and child.parent.player == self.opponent_piece)
                )
                if child_good:
                    return child
                continue

            if child.N == 0:
                s = fpu + C * _sqrt(log_N + 1)
            else:
                s = child.W / child.N + C * _sqrt(log_N / child.N)
            if s > best_score:
                best_score = s
                best = child

        return best

    def backpropagate(self, node, result):
        current = node
        while current is not None:
            current.N += 1
            if current.player == self.my_piece:
                current.W -= result
            else:
                current.W += result
            current = current.parent

    def propagate_proven(self, node):
        child = node
        parent = child.parent
        while parent is not None:
            child_good_for_parent = (
                (child.proven == 1.0 and parent.player == self.my_piece) or
                (child.proven == -1.0 and parent.player == self.opponent_piece)
            )
            child_bad_for_parent = (
                (child.proven == -1.0 and parent.player == self.my_piece) or
                (child.proven == 1.0 and parent.player == self.opponent_piece)
            )

            if child_good_for_parent:
                parent.proven = 1.0 if parent.player == self.my_piece else -1.0
            elif child_bad_for_parent:
                all_bad = True
                for c in parent.children.values():
                    if c.proven is None:
                        all_bad = False
                        break
                    c_bad = (
                        (c.proven == -1.0 and parent.player == self.my_piece) or
                        (c.proven == 1.0 and parent.player == self.opponent_piece)
                    )
                    if not c_bad:
                        all_bad = False
                        break
                if all_bad and not parent.untried_moves:
                    parent.proven = -1.0 if parent.player == self.my_piece else 1.0
                else:
                    break
            else:
                break

            child = parent
            parent = child.parent

    # ============ EVALUATION FUNCTION ============

    def leaf_evaluation(self, state, current_player):
        """1-ply tactical check + heuristic evaluation from Part 1."""
        legal = self.get_legal_moves(state)
        if not legal:
            return 0.0

        piece = self.my_piece if current_player == self.my_piece else self.opponent_piece

        # Check if current player can win immediately
        for col in legal:
            row = self.drop_piece(state, col, piece)
            won = self.check_win(state, row, col, piece)
            self.undo_piece(state, row, col)
            if won:
                return 1.0 if piece == self.my_piece else -1.0

        # Check if opponent can win immediately (double threat = forced loss)
        opp = -piece
        threat_count = 0
        for col in legal:
            row = self.drop_piece(state, col, opp)
            if self.check_win(state, row, col, opp):
                threat_count += 1
            self.undo_piece(state, row, col)

        if threat_count >= 2:
            return -1.0 if piece == self.my_piece else 1.0

        # Heuristic evaluation
        score = self.board_evaluation(state)
        return math.tanh(score / 2000.0)

    def board_evaluation(self, game_state):
        weights = {0: 0, 1: 1, 2: 12, 3: 150, 4: 2500, 5: 100000}
        score = 0
        my_threats = 0
        opp_threats = 0

        for r in range(self.rows):
            for c in range(self.cols):
                # Horizontal
                if self.cylinder or c + self.connect_number <= self.cols:
                    my_count, opp_count = 0, 0
                    for i in range(self.connect_number):
                        cell = game_state[r][(c + i) % self.cols]
                        if cell == self.my_piece:
                            my_count += 1
                        elif cell == self.opponent_piece:
                            opp_count += 1
                    if my_count > 0 and opp_count == 0:
                        s = weights[my_count]
                        score += s
                        if s >= 2500:
                            my_threats += 1
                    elif opp_count > 0 and my_count == 0:
                        s = weights[opp_count]
                        score -= s
                        if s >= 2500:
                            opp_threats += 1

                # Vertical
                if r + self.connect_number <= self.rows:
                    my_count, opp_count = 0, 0
                    for i in range(self.connect_number):
                        cell = game_state[r + i][c]
                        if cell == self.my_piece:
                            my_count += 1
                        elif cell == self.opponent_piece:
                            opp_count += 1
                    if my_count > 0 and opp_count == 0:
                        s = weights[my_count]
                        score += s
                        if s >= 2500:
                            my_threats += 1
                    elif opp_count > 0 and my_count == 0:
                        s = weights[opp_count]
                        score -= s
                        if s >= 2500:
                            opp_threats += 1

                # Positive diagonal
                if r + self.connect_number <= self.rows and (self.cylinder or c + self.connect_number <= self.cols):
                    my_count, opp_count = 0, 0
                    for i in range(self.connect_number):
                        cell = game_state[r + i][(c + i) % self.cols]
                        if cell == self.my_piece:
                            my_count += 1
                        elif cell == self.opponent_piece:
                            opp_count += 1
                    if my_count > 0 and opp_count == 0:
                        s = weights[my_count]
                        score += s
                        if s >= 2500:
                            my_threats += 1
                    elif opp_count > 0 and my_count == 0:
                        s = weights[opp_count]
                        score -= s
                        if s >= 2500:
                            opp_threats += 1

                # Negative diagonal
                if r - self.connect_number + 1 >= 0 and (self.cylinder or c + self.connect_number <= self.cols):
                    my_count, opp_count = 0, 0
                    for i in range(self.connect_number):
                        cell = game_state[r - i][(c + i) % self.cols]
                        if cell == self.my_piece:
                            my_count += 1
                        elif cell == self.opponent_piece:
                            opp_count += 1
                    if my_count > 0 and opp_count == 0:
                        s = weights[my_count]
                        score += s
                        if s >= 2500:
                            my_threats += 1
                    elif opp_count > 0 and my_count == 0:
                        s = weights[opp_count]
                        score -= s
                        if s >= 2500:
                            opp_threats += 1

        # Double-threat bonus
        if my_threats >= 2:
            score += my_threats * 500
        if opp_threats >= 2:
            score -= opp_threats * 600

        return score

    # ============ BOARD STATE UTILITIES ============

    def get_legal_moves(self, state):
        cols = []
        for col in range(self.cols):
            if state[0][col] == 0:
                cols.append(col)
        return cols

    def get_landing_row(self, state, col):
        for row in range(self.rows - 1, -1, -1):
            if state[row][col] == 0:
                return row
        return -1

    def drop_piece(self, state, col, player):
        row = self.get_landing_row(state, col)
        state[row][col] = player
        return row

    def undo_piece(self, state, row, col):
        state[row][col] = 0

    def check_win(self, state, row, col, player):
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            r, c = row + dr, col + dc
            for _ in range(4):
                if r < 0 or r >= self.rows:
                    break
                cc = c % self.cols if self.cylinder else c
                if not self.cylinder and (c < 0 or c >= self.cols):
                    break
                if state[r][cc] == player:
                    count += 1
                    if count >= 5:
                        return True
                else:
                    break
                r += dr
                c += dc
            r, c = row - dr, col - dc
            for _ in range(4):
                if r < 0 or r >= self.rows:
                    break
                cc = c % self.cols if self.cylinder else c
                if not self.cylinder and (c < 0 or c >= self.cols):
                    break
                if state[r][cc] == player:
                    count += 1
                    if count >= 5:
                        return True
                else:
                    break
                r -= dr
                c -= dc
        return False


Player = VakulMCTS
