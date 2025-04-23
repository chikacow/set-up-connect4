import connect4_ai
import time
from typing import List, Tuple

def extract_position_simple(board: List[List[int]]) -> str:
    """
    Extract position notation from the board state (columns 1-7)
    Returns: String like "4453" representing move sequence
    """
    column_counts = [0] * 7
    move_sequence = []
    
    # Count stones in each column (bottom to top)
    for col in range(7):
        for row in range(6):
            if board[row][col] != 0:
                column_counts[col] += 1
    
    # Reconstruct move sequence by matching player turns
    total_moves = sum(column_counts)
    temp_counts = column_counts.copy()
    
    for move_num in range(total_moves):
        current_player = 1 if move_num % 2 == 0 else 2
        
        for col in range(7):
            if temp_counts[col] > 0:
                row = 6 - temp_counts[col]
                if board[row][col] == current_player:
                    move_sequence.append(str(col + 1))  # Convert to 1-based
                    temp_counts[col] -= 1
                    break
    
    return ''.join(move_sequence)

def print_board(board: List[List[int]]):
    """Print the board with row and column indicators"""
    print("\n  1 2 3 4 5 6 7")
    print("  --------------")
    for row in board:
        print("| " + " ".join(str(cell) if cell != 0 else "." for cell in row) + " |")
    print("  --------------")

def make_move(board: List[List[int]], column: int, player: int) -> bool:
    """Make a move on the board (0-based columns)"""
    if column < 0 or column > 6:
        return False
    if board[0][column] != 0:
        return False
        
    for row in range(5, -1, -1):
        if board[row][column] == 0:
            board[row][column] = player
            return True
    return False

def get_valid_moves(board: List[List[int]]) -> List[int]:
    """Return list of valid column indices (0-based)"""
    return [col for col in range(7) if board[0][col] == 0]

def main():
    print("Connect 4 AI Tester")
    print("Columns are numbered 1-7 (left to right)")
    print("---------------------------------------")
    
    board = [[0 for _ in range(7)] for _ in range(6)]
    current_player = 2
    position_history = ""
    
    # new_game = input("Start a new game? (y/n): ").strip().lower()
    # if new_game != 'y':
    #     print("Exiting game.")
    #     return
    
    game_over = False
    
    while not game_over:
        print_board(board)
        valid_moves = [col + 1 for col in get_valid_moves(board)]  # Convert to 1-based
        
        if not valid_moves:
            print("\nGame ended in a draw!")
            break
            
        print(f"\nPlayer {current_player}'s turn")
        
        if current_player == 1:
            # Human player
            while True:
                try:
                    move = int(input("Enter column (1-7): "))
                    if move in valid_moves:
                        if make_move(board, move - 1, current_player):  # Convert to 0-based
                            position_history += str(move)
                            break
                    print("Invalid move. ", end="")
                    if move < 1 or move > 7:
                        print("Column must be between 1-7")
                    else:
                        print("Column is full")
                except ValueError:
                    print("Please enter a number")
        else:
            # AI player
            print(f"Current position notation: {position_history}")
            start_time = time.time()
            score, move = connect4_ai.solve_position(position_history)
            elapsed = time.time() - start_time
            
            print(f"AI (Player {current_player}) chooses column: {move} (score: {score})")
            print(f"Decision time: {elapsed:.3f} seconds")
            print(f"Valid moves: {valid_moves}")
            # Validate and make AI move
            if move + 1 in valid_moves:
                make_move(board, move , current_player)  # Convert to 0-based
                position_history += str(move + 1)
            else:
                print("Warning: AI chose invalid move, selecting first valid move")
                move = valid_moves[0]
                make_move(board, move , current_player)  # Convert to 0-based
                position_history += str(move + 1)
        
        # Check for winner
        if check_winner(board, current_player):
            print_board(board)
            print(f"\nPlayer {current_player} wins!")
            game_over = True
        
        current_player = 2 if current_player == 1 else 1

def check_winner(board: List[List[int]], player: int) -> bool:
    """Check if the current player has won"""
    # Check horizontal
    for row in range(6):
        for col in range(4):
            if all(board[row][col+i] == player for i in range(4)):
                return True
    
    # Check vertical
    for row in range(3):
        for col in range(7):
            if all(board[row+i][col] == player for i in range(4)):
                return True
    
    # Check diagonals
    for row in range(3):
        for col in range(4):
            if all(board[row+i][col+i] == player for i in range(4)):
                return True
            if all(board[row+i][col+3-i] == player for i in range(4)):
                return True
    return False

if __name__ == "__main__":
    main()