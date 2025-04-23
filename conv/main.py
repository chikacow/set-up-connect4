# import connect4_ai
# import time  # Import the time module

# # Example usage
# start_time = time.time()  # Start the timer
# score, best_move = connect4_ai.solve_position('444')
# end_time = time.time()  # End the timer

# # Calculate the elapsed time
# elapsed_time = end_time - start_time

# # Print the results
# print(f"Best move: {best_move}, Score: {score}")
# print(f"Time taken: {elapsed_time:.4f} seconds")
import connect4_ai
import time

def display_board(board):
    """Display the Connect 4 board."""
    for row in board:
        print(' '.join(row))
    print()

def is_valid_move(board, column):
    """Check if a move is valid."""
    return board[0][column] == ' '

def make_move(board, column, player):
    """Make a move on the board."""
    for row in reversed(board):
        if row[column] == ' ':
            row[column] = player
            break

def check_winner(board, player):
    """Check if the given player has won."""
    # Check horizontal, vertical, and diagonal lines
    for row in range(6):
        for col in range(7):
            if (
                col + 3 < 7 and all(board[row][col + i] == player for i in range(4)) or
                row + 3 < 6 and all(board[row + i][col] == player for i in range(4)) or
                row + 3 < 6 and col + 3 < 7 and all(board[row + i][col + i] == player for i in range(4)) or
                row - 3 >= 0 and col + 3 < 7 and all(board[row - i][col + i] == player for i in range(4))
            ):
                return True
    return False

def is_draw(board):
    """Check if the game is a draw."""
    return all(cell != ' ' for row in board for cell in row)

def main():
    # Initialize the board
    board = [[' ' for _ in range(7)] for _ in range(6)]
    human_player = 'X'
    ai_player = 'O'

    print("Welcome to Connect 4!")
    display_board(board)

    while True:
        # Human move
        while True:
            try:
                column = int(input("Enter your move (1-7): ")) - 1
                if 0 <= column < 7 and is_valid_move(board, column):
                    break
                else:
                    print("Invalid move. Try again.")
            except ValueError:
                print("Please enter a valid number between 1 and 7.")

        make_move(board, column, human_player)
        display_board(board)

        if check_winner(board, human_player):
            print("You win!")
            break
        if is_draw(board):
            print("It's a draw!")
            break

        # AI move
        print("AI is thinking...")
        start_time = time.time()
        board_state = ''.join(''.join(row) for row in board)
        score, best_move = connect4_ai.solve_position(board_state)
        end_time = time.time()

        make_move(board, best_move, ai_player)
        display_board(board)

        print(f"AI chose column {best_move + 1} (Score: {score})")
        print(f"Time taken: {end_time - start_time:.4f} seconds")

        if check_winner(board, ai_player):
            print("AI wins!")
            break
        if is_draw(board):
            print("It's a draw!")
            break

if __name__ == "__main__":
    main()