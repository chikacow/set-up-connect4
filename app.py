from fastapi import FastAPI, HTTPException
import random
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
# from bot import ConnectFourAI, ConnectFour, get_best_move
import copy
import time
# import connect4_ai
from bot_2 import ConnectFourAI, ConnectFour
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]
    is_new_game: bool

class AIResponse(BaseModel):
    move: int

position_history = ""
board = [[0 for _ in range(7)] for _ in range(6)]



@app.get("/api/test")
async def health_check():
    return {"status": "ok", "message": "Server is running"}

# def print_board(board: List[List[int]]):
#     """Print the board with row and column indicators"""
#     print("\n  1 2 3 4 5 6 7")
#     print("  --------------")
#     for row in board:
#         print("| " + " ".join(str(cell) if cell != 0 else "." for cell in row) + " |")
#     print("  --------------")

def print_board(board):
    # Print board top-down for human readability
    for row in reversed(board):
        print("[" + " ".join(
            {0: ".", 1: "X", 2: "O", -1: "#"}.get(cell, "?") for cell in row
        ) + "]")
def detect_opponent_move(current_board: List[List[int]], new_board: List[List[int]]) -> int:
    """
    Detect the column where the opponent made a move by comparing the current board
    with the new board. Returns the 0-based column index of the move.
    """
    for col in range(7):  # Iterate through columns
        for row in range(6):  # Iterate through rows
            if current_board[row][col] == 0 and new_board[row][col] != 0 and new_board[row][col] != 3:
                # A new piece has been added in this column
                return col
    return -1  # Return -1 if no move is detected (shouldn't happen in a valid game)
@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    global position_history
    global board 
    
    try:
        # Initialize position history for new game
        if game_state.is_new_game:
            position_history = ""
            board = [[0 for _ in range(7)] for _ in range(6)]
            # if(game_state.current_player == 2):
            #     init_board = [[0 for _ in range(7)] for _ in range(6)]

            #     opponent_move_col = detect_opponent_move(init_board, game_state.board)
            #     if opponent_move_col != -1:
            #         print(f"Opponent moved in column: {opponent_move_col + 1}")
            #         position_history += str(opponent_move_col + 1)
                
        #     print("Starting new game, reset position history")

        # if not game_state.valid_moves:
        #     print("No valid moves available - game ended")
        #     return AIResponse(move=game_state.valid_moves[0])
        # ## Initialize game and AI instances
        # game = ConnectFour()
        # ai = ConnectFourAI()

        # # Load the board state into the game
        # reversed_board = [row[:] for row in reversed(game_state.board)]
        
        # # Load the board state into the game
        # game.board = copy.deepcopy(reversed_board)
        
        # # Calculate the heights based on the board state
        # for col in range(game.cols):
        #     for row in range(game.rows):
        #         if game.board[row][col] != 0:
        #             game.heights[col] = row + 1
        
        # # Determine whose turn it is based on the number of pieces

        # current_player = game_state.current_player
        # ai_player = current_player
        # print("Current board state:")
        # for row in reversed(game.board):
        #     print(row)
        # print(f"\nCurrent player: {current_player}")
        # print(f"Valid moves: {game.get_valid_moves()}")
        
        # # Get AI's move
      
        # # Print current board state for debugging
        # print("\nCurrent board state:")
        # print_board(game_state.board)
        # print(f"Current player: {game_state.current_player}")
        # print(f"Valid moves (0-based): {game_state.valid_moves}")
        # print(f"Is new game: {game_state.is_new_game}")
        
        # # Detect opponent's move
        # if 'board' in globals():
        #     opponent_move_col = detect_opponent_move(board, game_state.board)
        #     if opponent_move_col != -1:
        #         print(f"Opponent moved in column: {opponent_move_col + 1}")
        #         position_history += str(opponent_move_col + 1)
        # print(f"Position history: {position_history}")
        # # Update the global board state
        # board = [row[:] for row in game_state.board]
        # # Get AI move
        # print("\nConsulting AI...")
        # start_time = time.time()
        # score, move_col = connect4_ai.solve_position(position_history)

        # elapsed = time.time() - start_time
        # print(f"AI suggested move (0-based): {move_col} (score: {score})")
        # print(f"Decision time: {elapsed:.3f} seconds")

        # if move_col not in game_state.valid_moves:
        #     print(f"Warning: AI suggested invalid move {move_col}, using first valid move")
        #     move_col = game_state.valid_moves[0]
        # # Make the move
        # if game.make_move(move_col, ai_player):
        #     if game.is_winner(ai_player):
        #         print(f"\nPlayer {ai_player} wins!")
        #         return AIResponse(move=move_col)
        #     elif game.is_full():
        #         print("\nThe game is a draw!")
        # else:
        #     print("Invalid move selected by AI")


        
        # # Update position history with 1-based column number
        # position_history += str(move_col + 1)
        # print(f"Updated position history: {position_history}")

        # # Check for winner (on a copied board to avoid modifying original)
        # temp_board = [row[:] for row in game_state.board]
        # if make_move(temp_board, move_col, game_state.current_player):
        #      # Update the global board state
        #     if check_winner(temp_board, game_state.current_player):
        #         print(f"Player {game_state.current_player} would win with this move!")
        #         return AIResponse(move=game_state.valid_moves[0])
        # print(f"Board after move:")
        # print_board(temp_board)
        # board = temp_board
        board = game_state.board
        current_player = game_state.current_player
        valid_moves = game_state.valid_moves
        is_new_game = game_state.is_new_game
        reversed_board = [row.copy() for row in reversed(board)]
        game = ConnectFour(reversed_board, valid_moves)
        ai = ConnectFourAI()

        print("Connect 4 AI with Input Board")
        print("Current board state (# indicates forbidden cells):")
        print_board(game.board)
        print(f"\nCurrent player: {current_player}")
        print(f"Valid moves: {game.valid_moves}")
        print(f"Is new game: {is_new_game}")
        start_time = time.time()
        best_score, best_move = ai.find_best_move(game, current_player)
        print(f"\nAI (Player {current_player}) chooses column: {best_move} with score: {best_score}")

        elapsed_time = time.time() - start_time
        print(f"AI đã suy nghĩ trong {elapsed_time:.3f} giây.")

        if game.make_move(best_move, current_player):
            print("\nNew board state:")
            print_board(game.board)

            if game.is_winner(current_player):
                print(f"\nPlayer {current_player} wins!")
            elif game.is_full():
                print("\nThe game is a draw!")
        else:
            print("Invalid move selected by AI")
        if best_move not in game_state.valid_moves:
            print(f"Warning: AI suggested invalid move {best_move}, using first valid move")
            best_move = game_state.valid_moves[0]
        return AIResponse(move=best_move)

    except Exception as e:
        print(f"Error in make_move: {str(e)}")
        if game_state.valid_moves:
            # Fallback to first valid move if error occurs
            fallback_move = game_state.valid_moves[0]
            position_history += str(fallback_move + 1)
            return AIResponse(move=fallback_move)
        raise HTTPException(status_code=400, detail=str(e))

# Helper functions (same as before)
def make_move(board: List[List[int]], column: int, player: int) -> bool:
    """Make a move on the board (0-based columns)"""
    for row in range(5, -1, -1):
        if board[row][column] == 0:
            board[row][column] = player
            return True
    return False

def check_winner(board: List[List[int]], player: int) -> bool:
    """Check if the current player has won"""
    # Horizontal, vertical, and diagonal checks
    for row in range(6):
        for col in range(4):
            if all(board[row][col+i] == player for i in range(4)):
                return True
    
    for row in range(3):
        for col in range(7):
            if all(board[row+i][col] == player for i in range(4)):
                return True
    
    for row in range(3):
        for col in range(4):
            if all(board[row+i][col+i] == player for i in range(4)):
                return True
            if all(board[row+i][col+3-i] == player for i in range(4)):
                return True
    return False

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)