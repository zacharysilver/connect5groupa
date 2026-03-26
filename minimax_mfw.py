import numpy as np

from player import Player


class Minimax(Player):

    def play(self, board: np.ndarray):
        import time

        # --- Heuristic parameters ---
        WEIGHT_BASE          = 4
        WRAP_BONUS           = 1.5
        MULTIPLE_THREAT_BONUS = 10000
        CROSS_WRAP_THREAT_BONUS = 8000
        CHAIN_BASE           = 1.3
        DEFAULT_MAX_DEPTH    = 5
        TIME_BUFFER          = 0.1

        time_limit = min(self.timeout_move - TIME_BUFFER, 5.0)
        start = time.time()
        n = self.connect_number

        # --- Board helpers ---

        def get_valid_moves(b):
            center = self.cols // 2
            cols = [c for c in range(self.cols) if b[0, c] == 0]
            return sorted(cols, key=lambda c: abs(c - center))

        def drop_piece(b, col, piece):
            b = b.copy()
            for row in range(self.rows - 1, -1, -1):
                if b[row, col] == 0:
                    b[row, col] = piece
                    return b
            return b

        def check_win(b, piece):
            # Horizontal
            for r in range(self.rows):
                for c in range(self.cols - n + 1):
                    if np.sum(b[r, c:c+n] == piece) == n:
                        return True
            # Horizontal wrap (cylinder)
            if self.cylinder:
                for r in range(self.rows):
                    for c in range(self.cols - n + 1, self.cols):
                        if all(b[r, (c + i) % self.cols] == piece for i in range(n)):
                            return True
            # Vertical
            for r in range(self.rows - n + 1):
                for c in range(self.cols):
                    if np.sum(b[r:r+n, c] == piece) == n:
                        return True
            # Diagonal down-right
            for r in range(self.rows - n + 1):
                for c in range(self.cols - n + 1):
                    if all(b[r+i, (c+i) % self.cols] == piece for i in range(n)):
                        return True
            # Diagonal up-right
            for r in range(n - 1, self.rows):
                for c in range(self.cols - n + 1):
                    if all(b[r-i, (c+i) % self.cols] == piece for i in range(n)):
                        return True
            return False

        def is_playable(b, r, c):
            """Empty cell that gravity allows playing into next."""
            return b[r, c] == 0 and (r == self.rows - 1 or b[r + 1, c] != 0)

        # --- Heuristic evaluation ---

        def evaluate(b):
            if check_win(b, 1):
                return float('inf')
            if check_win(b, -1):
                return float('-inf')
            if not get_valid_moves(b):
                return 0.0

            def score_for(piece):
                total = 0.0
                threats = []  # (r, c) of each immediate winning cell

                def process_window(positions, is_wrap=False):
                    nonlocal total
                    piece_count = sum(1 for r, c in positions if b[r, c] == piece)
                    opp_count   = sum(1 for r, c in positions if b[r, c] == -piece)
                    if opp_count > 0:
                        return  # blocked window
                    empty_positions = [(r, c) for r, c in positions if b[r, c] == 0]
                    playable   = sum(1 for r, c in empty_positions if is_playable(b, r, c))
                    unplayable = len(empty_positions) - playable

                    ws = (WEIGHT_BASE ** piece_count) * (0.5 ** unplayable)
                    if is_wrap:
                        ws *= WRAP_BONUS
                    total += ws

                    # Immediate winning threat: one empty playable cell away from win
                    if piece_count == n - 1 and playable == 1 and opp_count == 0:
                        win_cell = next((rc for rc in empty_positions if is_playable(b, *rc)), None)
                        if win_cell:
                            threats.append(win_cell)

                # Horizontal windows
                for r in range(self.rows):
                    for c in range(self.cols - n + 1):
                        process_window([(r, c + i) for i in range(n)])
                    if self.cylinder:
                        for c in range(self.cols - n + 1, self.cols):
                            process_window([(r, (c + i) % self.cols) for i in range(n)], is_wrap=True)

                # Vertical windows
                for r in range(self.rows - n + 1):
                    for c in range(self.cols):
                        process_window([(r + i, c) for i in range(n)])

                # Diagonal down-right
                for r in range(self.rows - n + 1):
                    for c in range(self.cols - n + 1):
                        process_window([(r + i, c + i) for i in range(n)])

                # Diagonal up-right
                for r in range(n - 1, self.rows):
                    for c in range(self.cols - n + 1):
                        process_window([(r - i, c + i) for i in range(n)])

                return total, threats

            off_score, off_threats = score_for(1)
            def_score, _           = score_for(-1)

            # Chain bonus: reward long consecutive horizontal runs
            def chain_bonus(piece):
                bonus = 0.0
                for r in range(self.rows):
                    chain = 0
                    for c in range(self.cols):
                        if b[r, c] == piece:
                            chain += 1
                            bonus += CHAIN_BASE ** chain
                        else:
                            chain = 0
                return bonus

            off_chain = chain_bonus(1)
            def_chain  = chain_bonus(-1)

            # Multiple simultaneous threats (deduplicate by cell)
            unique_threats = list(set(off_threats))
            threat_bonus = 0.0
            if len(unique_threats) > 1:
                threat_bonus += MULTIPLE_THREAT_BONUS * (len(unique_threats) - 1)
                if self.cylinder and len(unique_threats) >= 2:
                    cols_t = [c for _, c in unique_threats]
                    if max(cols_t) - min(cols_t) > self.cols // 2:
                        threat_bonus += CROSS_WRAP_THREAT_BONUS

            return (off_score - def_score * 2.0
                    + off_chain - def_chain * 0.5
                    + threat_bonus)

        # --- Minimax with alpha-beta pruning ---

        def minimax(b, depth, alpha, beta, maximizing):
            if time.time() - start >= time_limit:
                return None, None       # timeout sentinel — unwind immediately
            if depth == 0:
                return None, evaluate(b)

            moves = get_valid_moves(b)
            if not moves:
                return None, 0.0

            piece = 1 if maximizing else -1
            best_col = moves[0]
            best_val = float('-inf') if maximizing else float('inf')

            for col in moves:
                if time.time() - start >= time_limit:
                    break               # stop looping, return best found so far

                new_b = drop_piece(b, col, piece)

                # Immediate win: no need to search deeper
                if check_win(new_b, piece):
                    return col, float('inf') if maximizing else float('-inf')

                _, val = minimax(new_b, depth - 1, alpha, beta, not maximizing)

                if val is None:         # child timed out — propagate sentinel upward
                    break

                if maximizing:
                    if val > best_val:
                        best_val, best_col = val, col
                    alpha = max(alpha, val)
                else:
                    if val < best_val:
                        best_val, best_col = val, col
                    beta = min(beta, val)

                if beta <= alpha:
                    break

            return best_col, best_val

        # --- Iterative deepening ---

        valid = get_valid_moves(board)
        if not valid:
            return 0

        best_col = valid[0]
        for depth in range(1, DEFAULT_MAX_DEPTH + 1):
            if time.time() - start >= time_limit:
                break
            col, val = minimax(board, depth, float('-inf'), float('inf'), True)
            if val is None:
                break  # timed out mid-search; keep result from previous depth
            if col is not None:
                best_col = col
            if val in (float('inf'), float('-inf')):
                break  # found forced win/loss, no point searching deeper

        return best_col


# Alias so the game engine can load this module as 'minimax' and find 'Player'
Player = Minimax