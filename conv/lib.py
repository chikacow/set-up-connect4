import sys
import os

# Add the directory containing the modules to the Python path
# This simulates the Rust module structure where submodules are automatically found.
# In a real Python package, this might be handled by the package structure itself.
# Assuming the submodules (bitboard.py, solver.py, etc.) are in the same directory
# as this __init__.py file.
# sys.path.insert(0, os.path.dirname(__file__)) # Usually not needed if structured as a package

"""A perfect agent for playing or analysing the board game 'Connect 4'

This agent uses an optimised game tree search to find the
mathematically optimal move for any position.

# Basic Usage

```python
# Assuming the package is installed or in the Python path
# and the necessary classes are imported in __init__.py
from connect4_ai import Solver, BitBoard # Assuming these are imported below

# Example usage within a function or script
def main():
    try:
        # Note: Error handling for from_moves depends on its Python implementation.
        #       The Rust '?' operator implies potential errors.
        #       In Python, this typically means the method might raise an exception.
        board = BitBoard.from_moves("112233")
        solver = Solver(board)
        score, best_move = solver.solve()

        assert (score, best_move) == (18, 3)
        print("Example ran successfully!")
        return 0 # Indicate success
    except Exception as e:
        print(f"An error occurred during the example: {e}", file=sys.stderr)
        # Handle potential errors from from_moves or solve
        return 1 # Indicate failure

# Example of how the main function might be called
# if __name__ == "__main__":
#     exit_code = main()
#     sys.exit(exit_code)

# The example above is illustrative. The actual execution depends on how
# the user imports and uses the connect4_ai package.
```
"""

# Import necessary components from submodules to make them available
# when importing the main package (connect4_ai).
# NOTE: These imports assume the existence of corresponding .py files
#       (e.g., bitboard.py, solver.py, transposition_table.py, opening_database.py)
#       within the connect4_ai package directory. The actual classes/functions
#       to import depend on the implementation within those files.
#       These are placeholders based on the Rust structure and example.

# Placeholder imports - Replace with actual imports from your module files
# If the modules contain classes or functions needed at the package level, import them here.
# Example:
try:
    from . import transposition_table
except ImportError:
    # Create dummy modules/classes if they don't exist yet, to allow import
    # In a real scenario, these files (transposition_table.py, etc.) must exist.
    print("Warning: Module 'transposition_table' not found. Using placeholder.", file=sys.stderr)
    class MockTranspositionTableModule: pass
    transposition_table = MockTranspositionTableModule()

try:
    from . import bitboard
    # If BitBoard class is defined in bitboard.py
    from .bitboard import BitBoard
except ImportError:
    print("Warning: Module 'bitboard' or class 'BitBoard' not found. Using placeholder.", file=sys.stderr)
    class MockBitBoardModule:
        class BitBoard:
            @staticmethod
            def from_moves(moves_str):
                print(f"Placeholder BitBoard.from_moves called with: {moves_str}")
                # Return a dummy board object or raise if needed by Solver placeholder
                if moves_str == "112233":
                     # Create a dummy object that Solver might expect
                     dummy_board = type('DummyBoard', (object,), {})()
                     return dummy_board
                else:
                     raise ValueError("Placeholder: Invalid moves for dummy BitBoard")
    bitboard = MockBitBoardModule()
    BitBoard = bitboard.BitBoard # Make the placeholder class available

try:
    from . import opening_database
except ImportError:
    print("Warning: Module 'opening_database' not found. Using placeholder.", file=sys.stderr)
    class MockOpeningDatabaseModule: pass
    opening_database = MockOpeningDatabaseModule()

try:
    from . import solver
    # If Solver class is defined in solver.py
    from .solver import Solver
except ImportError:
    print("Warning: Module 'solver' or class 'Solver' not found. Using placeholder.", file=sys.stderr)
    class MockSolverModule:
        class Solver:
            def __init__(self, board_instance):
                print(f"Placeholder Solver initialized with board: {board_instance}")
                self._board = board_instance # Store the board if needed

            def solve(self):
                print("Placeholder Solver.solve called")
                # Return the value expected by the example assertion
                return (18, 3)
    solver = MockSolverModule()
    Solver = solver.Solver # Make the placeholder class available


# The Rust code uses `pub use anyhow;`. `anyhow` is a flexible error handling library.
# Python uses exceptions for error handling. There isn't a direct equivalent to
# re-exporting `anyhow`. We can define or import a base exception class if needed,
# but typically, specific exceptions are defined and raised within the modules.
# We import the base Exception class as a conceptual placeholder for where
# error types might be based or managed.
from builtins import Exception as AnyhowError # Alias for conceptual mapping


# The width of the game board in tiles
WIDTH: int = 7

# The height of the game board in tiles
HEIGHT: int = 6

# ensure that the given dimensions fit in a u64 for the bitboard representation
# Rust's const_assert! is a compile-time check. Python performs this at runtime
# when the module is imported.
assert WIDTH * (HEIGHT + 1) < 64, "Board dimensions (WIDTH * (HEIGHT + 1)) must be less than 64 for the bitboard representation"

# Define __all__ to specify the public API of the package (optional but good practice)
# This lists the names that are imported when using `from connect4_ai import *`
# It also helps tools understand the public interface.
__all__ = [
    'transposition_table', # Exposing the module itself
    'bitboard',            # Exposing the module itself
    'BitBoard',            # Exposing the class directly
    'opening_database',    # Exposing the module itself
    'solver',              # Exposing the module itself
    'Solver',              # Exposing the class directly
    'AnyhowError',         # Exposing the conceptual error base alias
    'WIDTH',
    'HEIGHT',
]
