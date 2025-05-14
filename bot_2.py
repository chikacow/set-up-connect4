import copy
from typing import List, Dict, Set, Tuple
import numpy as np
import time

class ConnectFour:
    def __init__(self, board: List[List[int]], valid_moves: List[int]):
        self.rows = 6
        self.cols = 7
        self.board = copy.deepcopy(board)
        self.heights = [0] * self.cols
        self.transposition_table: Dict[int, Tuple[int, int, int]] = {}
        self.valid_moves = valid_moves
        self.forbidden_cells: Set[Tuple[int, int]] = set()

        # Initialize forbidden cells and heights
        for c in range(self.cols):
            # Find the lowest empty cell, respecting forbidden cells
            self.heights[c] = 0
            for r in range(self.rows):
                if self.board[r][c] == -1:
                    self.forbidden_cells.add((r, c))
            while self.heights[c] < self.rows and (self.board[self.heights[c]][c] != 0 or (self.heights[c], c) in self.forbidden_cells):
                self.heights[c] += 1

    def make_move(self, col: int, player: int) -> bool:
        if col < 0 or col >= self.cols or col not in self.valid_moves:
            return False
        row = self.heights[col]
        if row >= self.rows or self.board[row][col] != 0 or (row, col) in self.forbidden_cells:
            return False
        self.board[row][col] = player
        self.heights[col] = row + 1
        # Skip forbidden or occupied cells
        while self.heights[col] < self.rows and (self.board[self.heights[col]][col] != 0 or (self.heights[col], col) in self.forbidden_cells):
            self.heights[col] += 1
        return True

    def undo_move(self, col: int) -> None:
        if col < 0 or col >= self.cols:
            return
        row = self.heights[col] - 1
        # Find the last placed piece, skipping forbidden or empty cells
        while row >= 0 and (self.board[row][col] == 0 or (row, col) in self.forbidden_cells):
            row -= 1
        if row >= 0 and self.board[row][col] in [1, 2]:
            self.board[row][col] = 0
            self.heights[col] = row
            # Ensure heights points to the next valid empty cell
            while self.heights[col] < self.rows and (self.board[self.heights[col]][col] != 0 or (self.heights[col], col) in self.forbidden_cells):
                self.heights[col] += 1

    def is_winner(self, player: int) -> bool:
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if all(self.board[r][c+i] == player for i in range(4)):
                    return True
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if all(self.board[r+i][c] == player for i in range(4)):
                    return True
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if all(self.board[r+i][c+i] == player for i in range(4)):
                    return True
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if all(self.board[r-i][c+i] == player for i in range(4)):
                    return True
        return False

    def is_full(self) -> bool:
        return all(self.heights[col] >= self.rows for col in range(self.cols))

    def get_valid_moves(self) -> List[int]:
        valid = []
        for col in range(self.cols):
            row = self.heights[col]
            if row < self.rows and self.board[row][col] == 0 and (row, col) not in self.forbidden_cells:
                valid.append(col)
        return valid

    def get_board_hash(self) -> int:
        return hash(str(self.board))

    def is_terminal(self) -> bool:
        return self.is_winner(1) or self.is_winner(2) or self.is_full()


class ConnectFourAI:
    def __init__(self):
        self.max_depth = 12  # Increase depth for better foresight
        self.CENTER_WEIGHT = 100
        self.THREE_WEIGHT = 100000  # Much higher weight for three-in-a-row
        self.TWO_WEIGHT = 50
        self.OPEN_THREE_WEIGHT = 500000  # Higher weight for open three
        self.WIN_SCORE = 100000000
        self.LOSE_SCORE = -100000000
        self.BLOCK_WIN_SCORE = 90000000  # Very high score for blocking
        self.DRAW_SCORE = 0
        self.time_limit = 1

    def order_moves(self, game: ConnectFour, moves: List[int], player: int) -> List[int]:
        move_scores = []
        opponent = 3 - player

        for col in moves:
            score = 0
            score += self.CENTER_WEIGHT * (3 - abs(3 - col))
            game.make_move(col, player)
            if game.is_winner(player):
                game.undo_move(col)
                return [col]
            game.undo_move(col)
            game.make_move(col, opponent)
            opponent_wins = game.is_winner(opponent)
            game.undo_move(col)
            if not opponent_wins:
                for next_col in game.get_valid_moves():
                    game.make_move(next_col, opponent)
                    if game.is_winner(opponent):
                        score += self.BLOCK_WIN_SCORE
                    game.undo_move(next_col)
            else:
                score += self.BLOCK_WIN_SCORE
            score += self.evaluate_move_potential(game, col, player)
            move_scores.append((score, col))

        move_scores.sort(reverse=True, key=lambda x: x[0])
        return [move for _, move in move_scores]

    def evaluate_move_potential(self, game: ConnectFour, col: int, player: int) -> int:
        if game.heights[col] >= game.rows:
            return -10000
        row = game.heights[col]
        score = 0
        opponent = 3 - player

        left = max(col - 3, 0)
        right = min(col + 3, game.cols - 1)
        window = [game.board[row][c] if game.heights[c] > row else -1
                  for c in range(left, right + 1)]
        score += self._evaluate_window_potential(window, player, opponent)

        if row >= 3:
            window = [game.board[r][col] for r in range(row - 3, row + 1)]
            score += self._evaluate_window_potential(window, player, opponent)

        for i in range(4):
            r = row - i
            c = col - i
            if r >= 0 and c >= 0 and r + 3 < game.rows and c + 3 < game.cols:
                window = [game.board[r+i][c+i] for i in range(4)]
                score += self._evaluate_window_potential(window, player, opponent)

        for i in range(4):
            r = row + i
            c = col - i
            if r < game.rows and c >= 0 and r - 3 >= 0 and c + 3 < game.cols:
                window = [game.board[r-i][c+i] for i in range(4)]
                score += self._evaluate_window_potential(window, player, opponent)

        return score

    def _evaluate_window_potential(self, window: List[int], player: int, opponent: int) -> int:
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(0)

        if opponent_count == 0:
            if player_count == 3 and empty_count == 1:
                return self.OPEN_THREE_WEIGHT
            if player_count == 2 and empty_count == 2:
                return self.TWO_WEIGHT * 2
        elif player_count == 0:
            if opponent_count == 3 and empty_count == 1:
                return -100000
            if opponent_count == 2 and empty_count == 2:
                return -self.TWO_WEIGHT * 2
        return 0

    def evaluate_position(self, game: ConnectFour, player: int, depth: int) -> int:
        opponent = 3 - player

        if game.is_winner(player):
            return self.WIN_SCORE + depth
        if game.is_winner(opponent):
            return self.LOSE_SCORE - depth
        if game.is_full():
            return self.DRAW_SCORE

        score = 0
        board = game.board

        for r in range(game.rows):
            for c in range(game.cols - 3):
                window = [board[r][c+i] for i in range(4)]
                score += self._evaluate_window(window, player, opponent)

        for c in range(game.cols):
            for r in range(game.rows - 3):
                window = [board[r+i][c] for i in range(4)]
                score += self._evaluate_window(window, player, opponent)

        for r in range(game.rows - 3):
            for c in range(game.cols - 3):
                window = [board[r+i][c+i] for i in range(4)]
                score += self._evaluate_window(window, player, opponent)

        for r in range(3, game.rows):
            for c in range(game.cols - 3):
                window = [board[r-i][c+i] for i in range(4)]
                score += self._evaluate_window(window, player, opponent)

        center_cols = [2, 3, 4]
        for c in center_cols:
            if game.heights[c] > 0 and board[game.heights[c]-1][c] == player:
                score += self.CENTER_WEIGHT

        return score

    def _evaluate_window(self, window: List[int], player: int, opponent: int) -> int:
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(0)

        if opponent_count == 0:
            if player_count == 4:
                return self.WIN_SCORE
            if player_count == 3 and empty_count == 1:
                return self.THREE_WEIGHT * 2
            if player_count == 2 and empty_count == 2:
                return self.TWO_WEIGHT
            if player_count == 1 and empty_count == 3:
                return self.TWO_WEIGHT // 2
        elif player_count == 0:
            if opponent_count == 4:
                return self.LOSE_SCORE
            if opponent_count == 3 and empty_count == 1:
                return -100000
            if opponent_count == 2 and empty_count == 2:
                return -self.TWO_WEIGHT
        return 0

    def negamax(self, game: ConnectFour, depth: int, alpha: int, beta: int,
                player: int, start_time: float, max_time: float) -> Tuple[int, int]:
        if time.time() - start_time > max_time:
            return 0, -1

        hash_key = game.get_board_hash()
        if hash_key in game.transposition_table:
            tt_value, tt_depth, tt_flag = game.transposition_table[hash_key]
            if tt_depth >= depth:
                if tt_flag == 0:
                    return tt_value, -1
                elif tt_flag == 1:
                    alpha = max(alpha, tt_value)
                elif tt_flag == 2:
                    beta = min(beta, tt_value)
                if alpha >= beta:
                    return tt_value, -1

        if depth == 0 or game.is_terminal():
            return self.evaluate_position(game, player, depth), -1

        valid_moves = self.order_moves(game, game.get_valid_moves(), player)
        if not valid_moves:
            return self.evaluate_position(game, player, depth), -1

        best_move = valid_moves[0]
        best_value = float('-inf')

        for move in valid_moves:
            game.make_move(move, player)
            value = -self.negamax(game, depth - 1, -beta, -alpha,
                                  3 - player, start_time, max_time)[0]
            game.undo_move(move)

            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break

        if best_value <= alpha:
            flag = 2
        elif best_value >= beta:
            flag = 1
        else:
            flag = 0
        game.transposition_table[hash_key] = (best_value, depth, flag)

        return best_value, best_move

    def find_best_move(self, game: ConnectFour, player: int) -> int:
        start_time = time.time()
        valid_moves = game.get_valid_moves()
        best_move = valid_moves[0] if valid_moves else -1
        best_score = float('-inf')
        max_depth = self.max_depth

        pieces = sum(sum(1 for cell in row if cell in [1, 2]) for row in game.board)

        if pieces < 8:
            max_depth = min(12, self.max_depth + 2)
        elif pieces < 24:
            max_depth = min(14, self.max_depth + 4)
        else:
            max_depth = min(16, self.max_depth + 6)

        for depth in range(1, max_depth + 1):
            try:
                score, move = self.negamax(
                    game, depth, float('-inf'), float('inf'),
                    player, start_time, self.time_limit
                )
                if move != -1 and score > best_score:
                    best_move = move
                    best_score = score
            except TimeoutError:
                break

        return best_score, best_move


def print_board(board):
    for row in reversed(board):
        print("[" + " ".join(
            {0: ".", 1: "O", 2: "X", -1: "#"}.get(cell, "?") for cell in row
        ) + "]")