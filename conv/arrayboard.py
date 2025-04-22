import sys
import os
from enum import Enum, auto
from typing import List, Tuple, Optional

# Check if running in a TTY, required for blessed
# If not, provide dummy implementations for terminal functions
_is_tty = sys.stdout.isatty()

if _is_tty:
    try:
        from blessed import Terminal
    except ImportError:
        print("Error: 'blessed' library not found. Please install it: pip install blessed")
        # Provide dummy implementations if blessed is not installed but we are in a TTY
        class DummyTerminal:
            def __getattr__(self, name):
                def dummy_method(*args, **kwargs):
                    # print(f"Dummy call to Terminal.{name}({args}, {kwargs})")
                    if name == 'get_location':
                        return (0, 0) # Default position
                    if name == 'height':
                        return 24 # Default height
                    if name == 'width':
                        return 80 # Default width
                    return self # Allow chaining
                return dummy_method
            def __call__(self, *args, **kwargs):
                return "" # Return empty string for styling calls like term.bold(...)
            def location(self, x=None, y=None):
                # print(f"Dummy call to Terminal.location(x={x}, y={y})")
                return self # Allow chaining
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        Terminal = DummyTerminal
        print("Warning: 'blessed' library not found. Terminal output will be basic.")
        _is_tty = False # Treat as non-tty if import fails

else:
    # Dummy Terminal class for non-TTY environments
    class DummyTerminal:
        def __getattr__(self, name):
            def dummy_method(*args, **kwargs):
                # print(f"Dummy call to Terminal.{name}({args}, {kwargs})")
                if name == 'get_location':
                    # Try getting terminal size, default if fails
                    try:
                        _cols, _rows = os.get_terminal_size()
                    except OSError:
                        _cols, _rows = 80, 24
                    # Return a plausible default position (e.g., bottom left)
                    return (0, _rows -1)
                if name == 'height':
                     try:
                        _cols, _rows = os.get_terminal_size()
                        return _rows
                     except OSError:
                        return 24 # Default height
                if name == 'width':
                    try:
                        _cols, _rows = os.get_terminal_size()
                        return _cols
                    except OSError:
                        return 80 # Default width
                return self # Allow chaining
            return dummy_method
        def __call__(self, *args, **kwargs):
            return "" # Return empty string for styling calls like term.bold(...)
        def location(self, x=None, y=None):
            # print(f"Dummy call to Terminal.location(x={x}, y={y})")
            return self # Allow chaining
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    Terminal = DummyTerminal
    # print("Warning: Not running in a TTY. Terminal output will be basic.")


# Constants
HEIGHT: int = 6
WIDTH: int = 7

# Enums
class Cell(Enum):
    PlayerOne = auto()
    PlayerTwo = auto()
    Empty = auto()

    def is_empty(self) -> bool:
        return self == Cell.Empty

class GameState(Enum):
    Playing = auto()
    PlayerOneWin = auto()
    PlayerTwoWin = auto()
    Draw = auto()

# Board Class
class ArrayBoard:
    def __init__(self):
        # cells are stored left-to-right, bottom-to-top
        self.cells: List[Cell] = [Cell.Empty] * (WIDTH * HEIGHT)
        self.heights: List[int] = [0] * WIDTH
        self.player_one: bool = True
        self.game: str = ""
        self.num_moves: int = 0
        self.state: GameState = GameState.Playing

    # @classmethod
    # @allow(unused) - Python doesn't have a direct equivalent for #[allow(unused)]
    #                 Methods are typically kept unless explicitly removed.
    @classmethod
    def new(cls) -> 'ArrayBoard':
        return cls()

    # @classmethod
    # @allow(unused)
    @classmethod
    def from_str(cls, moves: str) -> 'ArrayBoard':
        board = cls.new()

        for column_char in moves:
            try:
                column = int(column_char)
                # Call play_checked, ignore the returned state during setup
                board.play_checked(column)
            except ValueError:
                raise ValueError(f"could not parse '{column_char}' as a valid move")
            # Errors from play_checked (like full column) will raise exceptions

        return board

    def play_checked(self, column_one_indexed: int) -> GameState:
        if not (1 <= column_one_indexed <= WIDTH):
            raise ValueError(
                f"Invalid move, column {column_one_indexed} out of range. Columns must be between 1 and {WIDTH}"
            )

        column = column_one_indexed - 1
        if not self.playable(column):
            raise ValueError(f"Invalid move, column {column_one_indexed} full")

        # Check win/draw state *before* making the move, as per Rust logic
        if self.check_winning_move(column):
            self.state = GameState.PlayerOneWin if self.player_one else GameState.PlayerTwoWin
        elif self.check_draw_move(): # Check for draw *after* checking for win
             self.state = GameState.Draw
        else:
             self.state = GameState.Playing

        # Make the move regardless of the state check outcome (updates board state)
        self.play(column)
        self.game += str(column_one_indexed)

        # Return the state determined *before* the move was made, but after checks
        return self.state


    def check_draw_move(self) -> bool:
        # A draw occurs if only one cell is empty *before* the current move,
        # meaning the board will be full *after* this move, and it wasn't a winning move.
        empty_count = sum(1 for cell in self.cells if cell.is_empty())
        return empty_count == 1

    def display(self) -> None:
        term = Terminal()

        # Use term.stream instead of stdout directly for blessed compatibility
        stream = term.stream

        cols: str = "".join(map(str, range(1, WIDTH + 1)))
        # Print column headers - blessed handles styling
        # Note: blessed print doesn't automatically add newline unless specified
        print(term.bold(cols), file=stream)

        # Reserve space for the board - move cursor down HEIGHT lines
        # This replicates the effect of printing newlines in the Rust code
        # before getting the cursor position.
        initial_y, initial_x = term.get_location() or (0, 0) # Get current position
        target_y = initial_y + HEIGHT + 1 # Move below the board area + header
        print(term.move(target_y, 0), end='', file=stream) # Move cursor down
        stream.flush() # Ensure cursor move is processed

        # Get the starting position for drawing the board (bottom-left)
        # We position relative to where the cursor *would* be after printing newlines
        origin_y, origin_x = target_y -1, initial_x # One line up from the final cursor pos

        # Draw the board cells
        for idx, cell in enumerate(self.cells):
            col_idx = idx % WIDTH
            row_idx = idx // WIDTH
            # Calculate display position (x, y)
            # x corresponds to column, y corresponds to row from bottom
            pos_x = origin_x + col_idx
            pos_y = origin_y - row_idx # Subtract row_idx because we draw bottom-up

            # Map Cell enum to character and color
            if cell == Cell.PlayerOne:
                ch = "1"
                # Use blessed's color names or capabilities
                fg_color = term.red
            elif cell == Cell.PlayerTwo:
                ch = "2"
                fg_color = term.blue # Using blue instead of yellow as per active Rust code
            else: # Cell.Empty
                ch = "0"
                fg_color = term.black # Or term.bright_black for better visibility?

            # Move cursor and print styled character
            # blessed uses (y, x) for move
            with term.location(pos_x, pos_y):
                 # Use term styling functions
                 # Example: term.bold_red_on_black(...)
                 # Adjust based on available blessed features
                 # Using simple color for now
                 print(fg_color(term.bold(ch)), end='', file=stream)


        # Move cursor to the line below the board after drawing
        # The Rust code moves to (origin_x + WIDTH, origin_y) and prints a newline.
        # This effectively moves to the start of the next line after the board.
        print(term.move(origin_y + 1, 0), end='', file=stream)
        stream.flush() # Ensure all output is written


    def playable(self, column: int) -> bool:
        # Check bounds just in case, though play_checked should handle it
        if 0 <= column < WIDTH:
             return self.heights[column] < HEIGHT
        return False # Should not happen if called from play_checked

    def play(self, column: int) -> None:
        player = Cell.PlayerOne if self.player_one else Cell.PlayerTwo
        # Calculate the index in the 1D list
        index = column + WIDTH * self.heights[column]
        if 0 <= index < len(self.cells) and self.heights[column] < HEIGHT:
             self.cells[index] = player
             self.heights[column] += 1
             self.num_moves += 1
             self.player_one = not self.player_one
        else:
            # This case should ideally not be reached if playable() is checked first
            # Consider raising an error or logging if it occurs.
            print(f"Warning: Attempted to play in invalid location: col={column}, height={self.heights[column]}", file=sys.stderr)


    def check_winning_move(self, column: int) -> bool:
        # This check happens *before* the piece is placed,
        # so we check if placing a piece *would* result in a win.
        # The height used for checks is the *current* height in the column,
        # which is where the new piece *will* go.
        row = self.heights[column]

        # Ensure the move is valid (column not full) before checking
        if row >= HEIGHT:
            return False # Cannot win by placing in a full column

        player = Cell.PlayerOne if self.player_one else Cell.PlayerTwo

        # --- Check Vertical Alignment ---
        # Check if there are 3 pieces of the same player *below* the current spot
        if row >= 3:
            # Check cells at (column, row-1), (column, row-2), (column, row-3)
            if (self.cells[column + WIDTH * (row - 1)] == player and
                    self.cells[column + WIDTH * (row - 2)] == player and
                    self.cells[column + WIDTH * (row - 3)] == player):
                return True

        # --- Check Horizontal and Diagonal Alignments ---
        # The new piece is placed at (column, row)
        # We need to check runs of 4 including this new piece.

        # Iterate through directions: horizontal (0,1), diag up-right (1,1),
        # diag down-right (-1,1), diag up-left (1,-1) is covered by checking both sides
        # Rust code checks dy_dx = -1, 0, 1 which corresponds to:
        # dy_dx = -1: Diagonal down-right/up-left (\)
        # dy_dx =  0: Horizontal (-)
        # dy_dx =  1: Diagonal up-right/down-left (/)

        for dy_dx in [-1, 0, 1]: # Corresponds to dy/dx slope relative to horizontal check
            run = 0 # Counts consecutive pieces *excluding* the piece to be placed
            # Check in both directions (dx = -1 and dx = 1) along the line
            for dx in [-1, 1]:
                # Start checking from the neighbor cell in the current direction
                current_x = column + dx
                current_y = row + dx * dy_dx

                while True:
                    # Check bounds
                    if not (0 <= current_x < WIDTH and 0 <= current_y < HEIGHT):
                        break

                    # Check if the cell contains the player's piece
                    if self.cells[current_x + WIDTH * current_y] == player:
                        run += 1
                    else:
                        break # Streak broken

                    # Move to the next cell in the same direction
                    current_x += dx
                    current_y += dx * dy_dx

            # If the total run from both sides + the current piece is >= 4, it's a win
            if run >= 3:
                return True

        return False # No winning alignment found
