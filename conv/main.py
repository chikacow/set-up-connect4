import connect4_ai
import time  # Import the time module

# Example usage
string_board = "434"
start_time = time.time()  # Start the timer
score, best_move = connect4_ai.solve_position(string_board)
end_time = time.time()  # End the timer

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the results
print(f"Position notation: {string_board}")
print(f"Best move: {best_move}, Score: {score}")
print(f"Time taken: {elapsed_time:.4f} seconds")