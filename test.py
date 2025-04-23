import connect4_ai

def extract_position_simple(board):
    """
    Extract position notation from the board state
    """
    column_counts = [0] * 7
    move_sequence = []
    
    # Count stones in each column
    for col in range(7):
        for row in range(6):
            if board[row][col] != 0:
                column_counts[col] += 1
    
    # Reconstruct moves
    total_moves = sum(column_counts)
    temp_counts = [0] * 7
    
    for move in range(total_moves):
        player = 1 if move % 2 == 0 else 2
        
        for col in range(7):
            if temp_counts[col] < column_counts[col]:
                row = 5 - temp_counts[col]
                if board[row][col] == player:
                    move_sequence.append(col)
                    temp_counts[col] += 1
                    break
    
    return ''.join(map(str, move_sequence))

def print_board(board):
    """Print the board with row and column indicators"""
    print("\n  0 1 2 3 4 5 6")
    print("  -------------")
    for row in board:
        print("| " + " ".join(str(cell) if cell != 0 else "." for cell in row) + " |")
    print("  -------------")

def make_move(board, column, player):
    """Make a move on the board"""
    for row in range(5, -1, -1):
        if board[row][column] == 0:
            board[row][column] = player
            return True
    return False

def main():
    print("Connect 4 Position Notation Test")
    print("Input moves (0-6) for 4 turns")
    print("-----------------------------")
    
    # Initialize empty board
    board = [[0 for _ in range(7)] for _ in range(6)]
    current_player = 1
    
    for turn in range(1, 5):
        print(f"\nTurn {turn}: Player {current_player}'s move")
        print_board(board)
        
        while True:
            if(current_player == 1):
                try:
                    move = int(input("Enter column (0-6): "))
                    if move < 0 or move > 6:
                        print("Please enter a number between 0 and 6")
                        continue
                    if board[0][move] != 0:
                        print("Column is full, try another")
                        continue
                    if make_move(board, move, current_player):
                        break
                except ValueError:
                    print("Please enter a valid number")
            else:
                position = extract_position_simple(board)
                print(f"\nPosition notation: {position}")
                print("AI's turn to make a move")
                # move = connect4_ai.solve_position(position)
                move = int(input("Enter column (0-6): "))
                print(f"AI chooses column: {move}")
            print_board(board)
        
        current_player = 2 if current_player == 1 else 1
    
    print("\nTest complete! Final position:", position)

if __name__ == "__main__":
    main()