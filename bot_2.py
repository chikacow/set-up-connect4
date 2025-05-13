import copy
from typing import List, Dict, Tuple
import numpy as np
import time

class ConnectFour:
    def __init__(self, board: List[List[int]], valid_moves: List[int]):
        self.rows = 6
        self.cols = 7
        # Use the provided board directly (already reversed in main)
        self.board = copy.deepcopy(board)
        self.heights = [0] * self.cols
        self.transposition_table: Dict[int, Tuple[int, int, int]] = {}
        self.valid_moves = valid_moves
        self.forbidden_cells = set()

        # Find forbidden cells (-1) and update heights
        for r in range(self.rows):
            for c in range(self.cols):
                if self.board[r][c] == -1:
                    self.forbidden_cells.add((r, c))
                if self.board[r][c] in [1, 2, -1]:
                    self.heights[c] = max(self.heights[c], r + 1)

    def make_move(self, col: int, player: int) -> bool:
        if col not in self.valid_moves:
            return False
        while self.heights[col] < self.rows:
            row = self.heights[col]
            if self.board[row][col] == 0:
                self.board[row][col] = player
                self.heights[col] += 1
                self.valid_moves = self.get_valid_moves()
                return True
            elif self.board[row][col] == -1:
                self.heights[col] += 1
            else:
                return False
        return False

    def undo_move(self, col: int) -> None:
        for row in range(self.heights[col] - 1, -1, -1):
            if self.board[row][col] in [1, 2]:
                self.board[row][col] = 0
                self.heights[col] = row
                self.valid_moves = self.get_valid_moves()
                break

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
            while row < self.rows and self.board[row][col] == -1:
                row += 1
            if row < self.rows and self.board[row][col] == 0:
                valid.append(col)
        return valid

    def get_board_hash(self) -> int:
        return hash(str(self.board))

    def is_terminal(self) -> bool:
        return self.is_winner(1) or self.is_winner(2) or self.is_full()


class ConnectFourAI:
    def __init__(self):
        self.max_depth = 8
        self.CENTER_WEIGHT = 8
        self.THREE_WEIGHT = 100
        self.TWO_WEIGHT = 5
        self.OPEN_THREE_WEIGHT = 150
        self.WIN_SCORE = 100000
        self.LOSE_SCORE = -100000
        self.DRAW_SCORE = 0
        self.time_limit = 0.5

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
            if game.is_winner(opponent):
                score += self.WIN_SCORE // 2
            game.undo_move(col)
            score += self.evaluate_move_potential(game, col, player)
            move_scores.append((score, col))

        move_scores.sort(reverse=True, key=lambda x: x[0])
        return [move for _, move in move_scores]

    def evaluate_move_potential(self, game: ConnectFour, col: int, player: int) -> int:
        if game.heights[col] >= game.rows:
            return -1000
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
                return -self.THREE_WEIGHT * 2

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
                return -self.THREE_WEIGHT * 3
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

        valid_moves = self.order_moves(game, game.valid_moves, player)
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
                if best_value >= beta:
                    break
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
        best_move = game.valid_moves[0] if game.valid_moves else 3
        best_score = 0
        max_depth = self.max_depth

        pieces = sum(sum(1 for cell in row if cell in [1, 2]) for row in game.board)

        if pieces < 8:
            max_depth = min(8, self.max_depth + 2)
        elif pieces < 24:
            max_depth = min(10, self.max_depth + 4)
        else:
            max_depth = min(12, self.max_depth + 6)

        for depth in range(1, max_depth + 1):
            try:
                score, move = self.negamax(
                    game, depth, float('-inf'), float('inf'),
                    player, start_time, self.time_limit
                )
                if move != -1:
                    best_move = move
                    best_score = score
            except TimeoutError:
                break

        return best_score, best_move


def print_board(board):
    # Print board top-down for human readability
    for row in reversed(board):
        print("[" + " ".join(
            {0: ".", 1: "X", 2: "O", -1: "#"}.get(cell, "?") for cell in row
        ) + "]")

def main(input_data: Dict):
    board = input_data["board"]
    current_player = input_data["current_player"]
    valid_moves = input_data["valid_moves"]
    is_new_game = input_data["is_new_game"]
    # Reverse the board to match algorithm's expectation (bottom row first)
    reversed_board = [row[:] for row in reversed(board)]
    game = ConnectFour(reversed_board, valid_moves)
    ai = ConnectFourAI()

    print("Connect 4 AI with Input Board")
    print("Current board state (# indicates forbidden cells):")
    print_board(game.board)
    print(f"\nCurrent player: {current_player}")
    print(f"Valid moves: {game.valid_moves}")
    print(f"Is new game: {is_new_game}")

    best_score, best_move = ai.find_best_move(game, current_player)
    print(f"\nAI (Player {current_player}) chooses column: {best_move} with score: {best_score}")

    if game.make_move(best_move, current_player):
        print("\nNew board state:")
        print_board(game.board)

        if game.is_winner(current_player):
            print(f"\nPlayer {current_player} wins!")
        elif game.is_full():
            print("\nThe game is a draw!")
    else:
        print("Invalid move selected by AI")

    return best_move
def play():
    current_player = 1  # Player 1 is human, Player 2 is AI
    board_init = [
            [0, 0, -1, 0, 0, 0, 0],  # Top row in input
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 0]   # Bottom row in input
    ]
    valid_moves = [0, 1, 2, 3, 4, 5, 6]
    game = ConnectFour(board_init, valid_moves)
    ai = ConnectFourAI()

    print("Connect 4: Người chơi vs Máy (AI)")
    print("Hai ô ngẫu nhiên bị cấm sẽ được đánh dấu bằng '#'")
    print_board(game.board)

    while not game.is_full():
        print(f"\nLượt của người chơi {current_player}")
        print_board(game.board)
        print(f"Các cột hợp lệ: {game.get_valid_moves()}")

        if current_player == 1:
            # Lượt của người chơi (nhập từ bàn phím)
            try:
                move = int(input("Chọn cột (0-6): "))
                if move not in game.get_valid_moves():
                    print("Nước đi không hợp lệ. Thử lại.")
                    continue
            except ValueError:
                print("Vui lòng nhập số nguyên hợp lệ.")
                continue
        else:
            # Lượt của AI
            print("AI đang suy nghĩ...")
            _, move = ai.find_best_move(game, current_player)
            print(f"AI chọn cột: {move}")

        if not game.make_move(move, current_player):
            print("Nước đi không hợp lệ (có thể là ô cấm). Thử lại.")
            continue

        if game.is_winner(current_player):
            print_board(game.board)
            print(f"\nNgười chơi {current_player} thắng!")
            break

        current_player = 2 if current_player == 1 else 1  # Đổi lượt

    if game.is_full() and not game.is_winner(1) and not game.is_winner(2):
        print_board(game.board)
        print("\nVán đấu hòa!")
if __name__ == "__main__":
    sample_input = {
        "board": [
            [0, 0, -1, 0, 0, 0, 0],  # Top row in input
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0]   # Bottom row in input
        ],
        "current_player": 1,
        "valid_moves": [0, 1, 3, 4, 5, 6],
        "is_new_game": True
    }
    main(sample_input)
    # play()