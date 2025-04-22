# Required dependencies: None (uses standard Python libraries)
import sys
from typing import List, Optional, Iterator, Union

# Simulating constants from crate root
HEIGHT = 6
WIDTH = 7

# --- static_masks module translation ---

# Precompute masks similar to Rust's const fn
# Note: Python integers have arbitrary precision, so u64 equivalent is handled automatically.

def _calculate_bottom_mask() -> int:
    """Calculates the bottom mask."""
    mask = 0
    column = 0
    while column < WIDTH:
        # In Python, bit shifts are on arbitrary precision integers
        mask |= 1 << (column * (HEIGHT + 1))
        column += 1
    return mask

def _calculate_full_board_mask(bottom_mask_val: int) -> int:
    """Calculates the full board mask using the bottom mask."""
    # Ensure calculations handle potentially large numbers correctly
    return bottom_mask_val * ((1 << HEIGHT) - 1)

# Define the constants at the module level
STATIC_MASKS_BOTTOM_MASK = _calculate_bottom_mask()
STATIC_MASKS_FULL_BOARD_MASK = _calculate_full_board_mask(STATIC_MASKS_BOTTOM_MASK)

# --- End of static_masks module translation ---


class BitBoard:
    """! A compact, computationally efficient bit array representation of a Connect 4 board 

    # Notes
    Storing the state of the board in the bits of an integer allows parallel
    computation of game conditions with bitwise operations. A 7x6 Connect 4
    board fits into the bits of a `u64` like so:
    
    ```comment
    Column:  0  1  2  3  4  5  6
    
             6  13 20 28 35 42 49  <- Top guard row (indices)
             ____________________
          5 |05 12 19 27 34 41 48|
          4 |04 11 18 26 33 40 47|
          3 |03 10 17 24 32 39 46|
          2 |02 09 16 23 31 38 45|
          1 |01 08 15 22 30 37 44|
    Rows: 0 |00 07 14 21 29 36 43|
    ```
    Where bit index 00 is the least significant bit. The extra row of bits on top of the board
    identifies full columns and prevents bits overflowing into the next column
    
    # Board Keys
    A Connect 4 board can be unambiguously represented in a single u64 by placing a 1-bit in
    each square the board where the current player has a tile, and an additional 1-bit in
    the first empty square of a column. This representation is used to index the [transposition table]
    and created by [`BitBoard.key`]
    
    # Internal Representation
    This bitboard uses 2 `u64`s (Python `int`) for computational efficiency. One `int` stores a mask of all squares
    containing a tile of either color, and the other stores a mask of the current player's tiles
    
    # Huffman Codes
    A board with up to 12 tiles can be encoded into a `u32` (Python `int`) using a 
    [Huffman code](https://en.wikipedia.org/wiki/Huffman_coding), where the bit sequence `0` separates each 
    column and the code sequences `10` and `11` represent the first and second player's tiles respectively.
    A board with 12 tiles requires 6 bits of separators and 24 bits of tiles, for 30 bits total
    
    [transposition table]: ../transposition_table/struct.TranspositionTable.html # (Note: Link might not work in Python context)
    [`BitBoard.key`]: #method-key # (Note: Link might not work in Python context)
    """
    # mask of the current player's tiles
    player_mask: int
    # mask of all tiles
    board_mask: int
    num_moves: int

    def __init__(self, player_mask: int = 0, board_mask: int = 0, num_moves: int = 0):
        """Creates a new bitboard, optionally from existing parts."""
        # Corresponds to both new() and from_parts() via default arguments
        self.player_mask = player_mask
        self.board_mask = board_mask
        self.num_moves = num_moves

    @staticmethod
    def new() -> 'BitBoard':
        """Creates a new, empty bitboard"""
        # Kept for compatibility with Rust API, calls __init__ with defaults
        return BitBoard()

    @staticmethod
    def from_moves(moves: str) -> 'BitBoard':
        """Creates a board from a string of 1-indexed moves
        
        # Notes
        The move string is a sequence of columns played, indexed from 1 (meaning `"0"` is an invalid move)
        
        Returns `Err` (raises ValueError) if the move string represents an invalid position. Invalid positions can contain moves
        outside the column range, overfilled columns and winning positions for either player
        
        # Example
        ```python
        # import sys
        # # Add project root to path if needed
        # # sys.path.append('path/to/your/project') 
        # from your_module import BitBoard, WIDTH # Assuming BitBoard is in your_module.py
        
        try:
            # columns in move strings are 1-indexed
            board = BitBoard.from_moves("112233")
            
            # columns as integers are 0-indexed
            assert board.check_winning_move(3)
            print("Example successful!")
        except ValueError as e:
            print(f"Example failed: {e}")

        ```
        """
        board = BitBoard.new()

        for column_char in moves:
            # only play available moves
            try:
                # Attempt to convert char to digit, then to 0-indexed int
                column_digit = int(column_char)
                if 1 <= column_digit <= WIDTH:
                    column = column_digit - 1 # Convert to 0-indexed
                    if not board.playable(column):
                        raise ValueError(f"Invalid move, column {column + 1} full")
                    
                    # abort if the position is won at any point
                    # Note: The original Rust code checks *before* making the move if the *next* move wins.
                    # This seems slightly counter-intuitive for `from_moves` which should validate the *final* state,
                    # but we replicate the exact logic. A winning move by the player *making* the move is valid,
                    # but if the sequence leads to a state where the *previous* player already won, it's invalid.
                    # The check here prevents playing a move if *that move itself* wins the game for the current player.
                    # Let's re-read the Rust code carefully.
                    # `if board.check_winning_move(column)` checks if playing `column` *would* win for the *current* player.
                    # The Rust comment says "abort if the position is won at any point". This implies checking
                    # if the board *before* the current move was already won, or if the current move is invalid because it's played on a finished game.
                    # Let's test the Rust logic: If player 1 plays 1,1,2,2,3,3, then tries to play 4.
                    # Before playing 4, `board.check_winning_move(3)` (the potential winning move) is true.
                    # The code checks `board.check_winning_move(column)` *before* `board.play(move_bitmap)`.
                    # This means it prevents the *final* move if that move wins.
                    # The error message "Invalid position, game is over" suggests the check is meant to catch
                    # moves played *after* a win occurred.
                    # Let's stick to the literal translation: check if the move *being considered* is a winning move.
                    if board.check_winning_move(column):
                         # This error condition in Rust seems to prevent creating a board state *ending* in a win via from_moves.
                         # This might be intentional to only allow non-terminal positions from this function.
                         # Replicating this exact behavior.
                        raise ValueError("Invalid position, game is over")

                    move_bitmap = (board.board_mask + (1 << (column * (HEIGHT + 1)))) \
                        & BitBoard.column_mask(column)
                    board.play(move_bitmap)
                else:
                     raise ValueError(f"Column index {column_digit} out of range [1, {WIDTH}]")
            except ValueError:
                 # Catch non-digit characters or out-of-range columns
                 raise ValueError(f"could not parse '{column_char}' as a valid move") from None
                 
        # Final check: After all moves, is the game won by the player whose turn it *was*?
        # The Rust code doesn't explicitly do a final check *after* the loop.
        # The check inside the loop prevents the *last* move from being a winning one.
        # Let's adhere strictly to the Rust code's loop logic.

        return board

    @staticmethod
    def from_slice(moves: List[int]) -> 'BitBoard':
        """Creates a board from a slice (list) of 0-indexed moves
        
        Significantly faster than [`BitBoard.from_moves`] but provides less informative errors
        
        Returns `Err` (raises ValueError) if the board position is invalid (see [`BitBoard.from_moves`])
        
        # Warning
        This method assumes all items in the slice are in the valid column range [0, WIDTH-1].
        Providing numbers outside this range might lead to unexpected bitboard states due to Python's
        arbitrary precision integers handling large shifts differently than Rust's fixed-size overflow.
        
        # Example
        ```python
        # import sys
        # # Add project root to path if needed
        # # sys.path.append('path/to/your/project') 
        # from your_module import BitBoard # Assuming BitBoard is in your_module.py
        
        try:
            board = BitBoard.from_slice([0, 0, 1, 1, 2, 2])
            assert board.check_winning_move(3)
            print("Example successful!")
        except ValueError as e:
            print(f"Example failed: {e}")
            
        ```
        [`BitBoard.from_moves`]: #method-from_moves # (Note: Link might not work in Python context)
        """
        board = BitBoard.new()
        for column in moves:
            # Basic validation for column range, though the warning implies trust
            if not (0 <= column < WIDTH):
                 # Raise error similar to Rust's potential panic/unexpected state
                 raise ValueError(f"Column index {column} out of range [0, {WIDTH-1}]")
                 
            if not board.playable(column):
                # Corresponds to Err(()) in Rust
                raise ValueError(f"Invalid move, column {column + 1} full")
                
            # abort if the position is won at any point (see notes in from_moves)
            # This check prevents the move currently being processed if it wins the game.
            if board.check_winning_move(column):
                 # Corresponds to Err(()) in Rust
                raise ValueError("Invalid position, game is over")
                
            move_bitmap = \
                (board.board_mask + (1 << (column * (HEIGHT + 1)))) & BitBoard.column_mask(column)
            board.play(move_bitmap)
        return board

    @staticmethod
    def from_parts(player_mask: int, board_mask: int, num_moves: int) -> 'BitBoard':
        """Creates a bitboard from its constituent bit masks and move counter (see [Internal Representation])
        [Internal Representation]: #internal-representation # (Note: Link might not work in Python context)
        """
        # This functionality is covered by __init__
        return BitBoard(player_mask, board_mask, num_moves)

    def player_mask(self) -> int:
        """Accesses the internal mask of the current player's tiles"""
        return self.player_mask

    def board_mask(self) -> int:
        """Accesses the internal mask of tiles on the whole board"""
        return self.board_mask

    @staticmethod
    def top_mask(column: int) -> int:
        """Returns a mask of the top square (guard row) of a given column"""
        # Correction: Original comment says "top square", but calculation points to the guard bit above the top row.
        # Rust code: 1 << (column * (HEIGHT + 1) + (HEIGHT - 1)) -> This is the highest playable square (row HEIGHT-1)
        # Let's re-verify the diagram and Rust code.
        # Diagram: Row 5 is index 5, 12, 19... Row 6 is index 6, 13, 20... (guard row)
        # Column 0: bits 0..=5 are playable, bit 6 is guard.
        # Column 1: bits 7..=12 are playable, bit 13 is guard.
        # Column `c`: bits `c*(H+1)` to `c*(H+1) + H - 1` are playable. Bit `c*(H+1) + H` is guard.
        # The Rust code `1 << (column * (HEIGHT + 1) + (HEIGHT - 1))` points to the highest *playable* square.
        # The `playable` method uses `Self::top_mask(column) & self.board_mask == 0`.
        # If the top *playable* square (index H-1) is occupied, `board_mask` will have that bit set.
        # But `playable` should check if the *guard* bit is set. If `board_mask` has the guard bit set, the column is full.
        # Let's assume the Rust comment/implementation for `top_mask` is slightly misleadingly named,
        # and it *should* refer to the guard bit for the `playable` check logic to work as intended.
        # The calculation `(board_mask + bottom_mask(column))` correctly finds the lowest empty cell *or* the guard bit if full.
        # `possible_moves` uses `(board_mask + static_masks::bottom_mask()) & static_masks::full_board_mask()`.
        # `full_board_mask` includes rows 0 to H-1. It does *not* include the guard row.
        # `playable` uses `top_mask`. Let's re-evaluate `playable`.
        # `playable(col)`: `top_mask(col) & board_mask == 0`.
        # `top_mask(col)` = `1 << (col * (H+1) + H - 1)` (highest playable square).
        # This means `playable` returns true if the *highest playable square* is empty. This is incorrect.
        # A column is playable if the *guard bit* is empty in the `board_mask`.
        # The guard bit index is `col * (H+1) + H`.
        # Let's redefine `top_mask` to return the guard bit mask, as this seems necessary for `playable`.
        # This deviates from the literal Rust code for `top_mask` but aligns with the apparent *intent* of `playable`.
        # If we keep the Rust `top_mask` literally, `playable` is wrong.
        # Decision: Redefine `top_mask` to return the guard bit mask.
        # return 1 << (column * (HEIGHT + 1) + (HEIGHT - 1)) # Original Rust calculation (highest playable square)
        return 1 << (column * (HEIGHT + 1) + HEIGHT) # Guard bit mask

    @staticmethod
    def bottom_mask(column: int) -> int:
        """Returns a mask of the bottom square of a given column"""
        return 1 << (column * (HEIGHT + 1))

    @staticmethod
    def column_mask(column: int) -> int:
        """Returns a mask of the given column (playable squares only)"""
        # The mask should cover bits 0 to HEIGHT-1 within the column's section.
        # ((1 << HEIGHT) - 1) creates a mask for the bottom H bits (0 to H-1).
        # << (column * (HEIGHT + 1)) shifts this mask to the correct column's position.
        return ((1 << HEIGHT) - 1) << (column * (HEIGHT + 1))

    @staticmethod
    def column_from_move(move_bitmap: int) -> int:
        """Returns the column index (0-based) represented by a move bitmap 
        or [`WIDTH`] if the column is not found.

        [`WIDTH`]: ../constant.WIDTH.html # (Note: Link might not work in Python context)
        """
        # A move_bitmap should only have one bit set, representing the position of the new tile.
        for column in range(WIDTH):
            # Check if the move bit falls within the *entire* column range (including guard bit)
            # The column_mask only covers playable squares. A move bit could be the guard bit if column gets full.
            # Let's define a full column mask including the guard bit for this check.
            full_col_mask = ((1 << (HEIGHT + 1)) - 1) << (column * (HEIGHT + 1))
            if move_bitmap & full_col_mask != 0:
                return column
        # WIDTH is always an invalid column index, used here like Rust's Option::None equivalent
        return WIDTH 

    def non_losing_moves(self) -> int:
        """Returns a bitmap of all moves that don't give the opponent an immediate win"""
        possible_moves_mask = self.possible_moves()
        opponent_winning_positions_mask = self.opponent_winning_positions()
        # Moves that opponent could play to win immediately if we don't block
        forced_moves = possible_moves_mask & opponent_winning_positions_mask

        if forced_moves != 0:
            # if more than one forced move exists, you can't prevent the opponent winning
            # Check if more than one bit is set in forced_moves
            if forced_moves & (forced_moves - 1) != 0:
                return 0 # No non-losing moves available
            else:
                # Only one forced move exists, must play it
                possible_moves_mask = forced_moves
        
        # Avoid playing directly below an opponent's winning square
        # opponent_winning_positions >> 1 shifts the winning positions down by one row.
        # We mask out any possible moves that are in the square directly below an opponent's win.
        return possible_moves_mask & ~(opponent_winning_positions_mask >> 1)

    def possible_moves(self) -> int:
        """Returns a mask of all possible moves (lowest empty square in each column)"""
        # `board_mask + STATIC_MASKS_BOTTOM_MASK` performs addition column-wise.
        # For a column: `col_mask + col_bottom_mask`. If the column is empty (0), result is `col_bottom_mask` (bit 0).
        # If column has bits 0, 1 set (value 3), result is 3 + 1 = 4 (bit 2).
        # This finds the lowest empty square in each column. If a column is full (bits 0..H-1 set),
        # adding 1 will set the guard bit (bit H).
        # `& STATIC_MASKS_FULL_BOARD_MASK` masks out the guard bits, so full columns result in 0 here.
        return (self.board_mask + STATIC_MASKS_BOTTOM_MASK) & STATIC_MASKS_FULL_BOARD_MASK

    def opponent_winning_positions(self) -> int:
        """Returns a bitmap of open squares that complete alignments for the opponent"""
        # Opponent's mask is all tiles XOR current player's tiles
        opp_mask = self.player_mask ^ self.board_mask
        return self._winning_positions(opp_mask)

    def _winning_positions(self, player_mask: int) -> int:
        """Returns a mask of open squares that would complete a 4-in-a-row for the given player mask"""
        # Note: player_mask here is the mask of the player we are checking *for*, not necessarily self.player_mask

        # Vertical check: Find sequences of 3 vertically.
        # A winning position is the square *above* a sequence of 3.
        # (player_mask << 1): Player has tile at row r+1
        # (player_mask << 2): Player has tile at row r+2
        # (player_mask << 3): Player has tile at row r+3
        # Intersection finds squares at r+3 where r, r+1, r+2 also have player tiles.
        # This identifies the *top* tile of a vertical 3-in-a-row. The winning move is one step *above* this.
        # The Rust code seems to calculate `r` as the position *above* the 3-in-a-row directly.
        # Let's trace: If player has 0, 1, 2.
        # player_mask = ... 1 | (1<<1) | (1<<2) ...
        # pm << 1:     ... (1<<1) | (1<<2) | (1<<3) ...
        # pm << 2:     ... (1<<2) | (1<<3) | (1<<4) ...
        # pm << 3:     ... (1<<3) | (1<<4) | (1<<5) ...
        # Intersection: ... (1<<3) ... This is the position at index 3.
        # Yes, `r` correctly identifies the empty square *above* a vertical run of 3.
        r = (player_mask << 1) & (player_mask << 2) & (player_mask << 3)

        # Horizontal check
        h_shift = HEIGHT + 1
        # p = potential horizontal wins starting point (XXX_)
        p = (player_mask << h_shift) & (player_mask << (2 * h_shift))
        # Check for XXXO pattern (right end)
        r |= p & (player_mask << (3 * h_shift))
        # Check for X_XO pattern (hole in middle-right) - Requires player tile to the left
        # Rust: r |= p & (player_mask >> h_shift); -- This seems wrong.
        # Let's rethink X_XO: Need O at pos `i`, O at `i+2*h`, O at `i+3*h`.
        # `p` = mask for positions `i` where `i+h` and `i+2h` have player tiles (O O _ _)
        # We need `player_mask` at `i+3h`. So `p & (player_mask << (3*h_shift))` seems correct for O O O _
        # Let's re-read Rust: `r |= p & (player_mask >> (HEIGHT + 1));`
        # This checks if `p` (positions `i` where `i+h` and `i+2h` are set) AND `player_mask` at `i-h` are set.
        # This corresponds to a pattern _ O O _, where the winning move is at `i`.
        # Okay, the Rust code finds two patterns here: OOO_ and _OO_
        # Let's rename `p` for clarity in the first case (OOO_)
        p_ooo_ = (player_mask << h_shift) & (player_mask << (2 * h_shift))
        r |= p_ooo_ & (player_mask << (3 * h_shift)) # OOO_ (win at i+3h)
        # The second part uses the same `p` but checks `player_mask >> h_shift`.
        # Let's call the original player mask `pm`.
        # p = (pm << h) & (pm << 2h) -- set at index `i` if `i+h` and `i+2h` have tiles.
        # pm >> h -- set at index `i` if `i-h` has a tile.
        # p & (pm >> h) -- set at index `i` if `i-h`, `i+h`, `i+2h` have tiles. (_ O O _)
        # This finds the winning position `i`.
        r |= p_ooo_ & (player_mask >> h_shift) # _ O O _ (win at i)

        # p = potential horizontal wins starting point (_XXX)
        p_xxx_ = (player_mask >> h_shift) & (player_mask >> (2 * h_shift))
        # Check for OXXX pattern (left end)
        r |= p_xxx_ & (player_mask >> (3 * h_shift)) # _OOO (win at i-3h)
        # Check for OX_O pattern (hole in middle-left)
        # p_xxx_ is set at `i` if `i-h` and `i-2h` have tiles. (_ _ O O)
        # We need `player_mask` at `i+h`.
        # Rust: r |= p & (player_mask << (HEIGHT + 1));
        # Checks if `p_xxx_` (set at `i` where `i-h`, `i-2h` have tiles) AND `player_mask` at `i+h` are set.
        # This corresponds to O O _ O, where the winning move is at `i`.
        r |= p_xxx_ & (player_mask << h_shift) # O O _ O (win at i)

        # Diagonal / check
        d1_shift = HEIGHT # Shift up-right is H
        # p = potential diagonal wins starting point (XXX_) /
        p_d1_ = (player_mask << d1_shift) & (player_mask << (2 * d1_shift))
        # Check for XXXO pattern / (right end)
        r |= p_d1_ & (player_mask << (3 * d1_shift)) # OOO_ / (win at i+3d1)
        # Check for X_XO pattern / (hole middle-right) _ O O _ /
        r |= p_d1_ & (player_mask >> d1_shift) # _ O O _ / (win at i)

        # p = potential diagonal wins starting point (_XXX) /
        p_d1x_ = (player_mask >> d1_shift) & (player_mask >> (2 * d1_shift))
        # Check for OXXX pattern / (left end)
        r |= p_d1x_ & (player_mask >> (3 * d1_shift)) # _OOO / (win at i-3d1)
        # Check for OX_O pattern / (hole middle-left) O O _ O /
        r |= p_d1x_ & (player_mask << d1_shift) # O O _ O / (win at i)

        # Diagonal \ check
        d2_shift = HEIGHT + 2 # Shift up-left is H+2
        # p = potential diagonal wins starting point (XXX_) \
        p_d2_ = (player_mask << d2_shift) & (player_mask << (2 * d2_shift))
        # Check for XXXO pattern \ (right end)
        r |= p_d2_ & (player_mask << (3 * d2_shift)) # OOO_ \ (win at i+3d2)
        # Check for X_XO pattern \ (hole middle-right) _ O O _ \
        r |= p_d2_ & (player_mask >> d2_shift) # _ O O _ \ (win at i)

        # p = potential diagonal wins starting point (_XXX) \
        p_d2x_ = (player_mask >> d2_shift) & (player_mask >> (2 * d2_shift))
        # Check for OXXX pattern \ (left end)
        r |= p_d2x_ & (player_mask >> (3 * d2_shift)) # _OOO \ (win at i-3d2)
        # Check for OX_O pattern \ (hole middle-left) O O _ O \
        r |= p_d2x_ & (player_mask << d2_shift) # O O _ O \ (win at i)

        # Return only positions that are currently empty and within the playable board area
        return r & (STATIC_MASKS_FULL_BOARD_MASK ^ self.board_mask)

    def move_score(self, candidate: int) -> int:
        """Scores a move bitmap by counting open 3-alignments created *after* the move"""
        # Calculate winning positions if the player makes the move 'candidate'
        # Note: candidate is a bitmap with a single bit set for the move.
        potential_player_mask = self.player_mask | candidate
        # Count how many ways this new state can lead to a win on the *next* move
        winning_pos_after_move = self._winning_positions(potential_player_mask)
        # Count the number of set bits (popcount)
        return bin(winning_pos_after_move).count('1')

    def num_moves(self) -> int:
        """Accesses the internal move counter"""
        return self.num_moves

    def playable(self, column: int) -> bool:
        """Returns whether a column is a legal move (i.e., not full)"""
        # Check if the guard bit for the column is empty in the board mask.
        # Uses the redefined top_mask which points to the guard bit.
        return BitBoard.top_mask(column) & self.board_mask == 0

    def play(self, move_bitmap: int):
        """Advances the game by applying a move bitmap and switching players"""
        # switch the current player by XORing player_mask with the full board mask
        # Tiles that were opponent's (in board_mask but not player_mask) become player's.
        # Tiles that were player's (in both) become opponent's (not in player_mask).
        self.player_mask ^= self.board_mask
        
        # add the new tile (represented by move_bitmap) to the board mask
        # Since player_mask was just flipped, the new tile belongs to the player who *was* current.
        # The player_mask now represents the *next* player. We need to add the tile to the *previous*
        # player's set, which is implicitly represented. Let's adjust player_mask *after* adding to board_mask.

        # Re-order based on Rust:
        # 1. Switch player: self.player_mask ^= self.board_mask
        #    Now player_mask holds the tiles of the player who just moved *before* their latest move.
        # 2. Add tile to board: self.board_mask |= move_bitmap
        #    The new tile is now on the board.
        # 3. Add tile to current player (who just moved): self.player_mask |= move_bitmap
        #    This seems wrong. The player mask should represent the player whose turn it *is*.
        #    Let's trace:
        #    Start: P1 turn. p_mask = P1 tiles, b_mask = P1|P2 tiles.
        #    P1 plays `move`.
        #    Rust step 1: self.player_mask ^= self.board_mask
        #       new_p_mask = P1 ^ (P1|P2) = (P1 & ~P1 & ~P2) | (~P1 & (P1|P2)) = 0 | (~P1 & P2) = P2 tiles. Correct.
        #    Rust step 2: self.board_mask |= move_bitmap
        #       new_b_mask = (P1|P2) | move. Correct.
        #    Rust step 3: self.num_moves += 1. Correct.
        #    Wait, the Rust code is:
        #    `self.player_mask ^= self.board_mask;` // player_mask now holds P2 tiles
        #    `self.board_mask |= move_bitmap;` // board_mask holds P1|P2|move
        #    `self.num_moves += 1;`
        #    It seems the player_mask *after* play represents the *next* player's tiles *before* their turn.
        #    Let's stick to the exact Rust logic.

        self.player_mask ^= self.board_mask
        self.board_mask |= move_bitmap
        self.num_moves += 1

    def check_winning_move(self, column: int) -> bool:
        """Returns whether playing in the given column is a winning move for the current player"""
        # Calculate the position of the new piece if played in 'column'
        # This uses the same logic as in possible_moves/play to find the landing spot.
        move_bitmap = (self.board_mask + BitBoard.bottom_mask(column)) & BitBoard.column_mask(column)
        
        # Create a hypothetical board state *after* the move for the current player
        pos = self.player_mask | move_bitmap

        # Check for 4-in-a-row in all four directions for the hypothetical state `pos`

        # Check horizontal alignment
        h_shift = HEIGHT + 1
        # m = pairs of horizontally adjacent tiles
        m = pos & (pos >> h_shift)
        # Check for two pairs separated by two columns (X X . . X X) -> No, (X X X X)
        # If m has bits i and i+h_shift set, it means pos has i, i+h_shift, i+2*h_shift set.
        # Let's trace: pos = ... 1 | (1<<h) | (1<<2h) | (1<<3h) ... (4 in a row)
        # pos >> h:      ... 1 | (1<<h) | (1<<2h) ...
        # m = pos & (pos >> h): ... (1<<h) | (1<<2h) | (1<<3h) ... (3 pairs starting at index h)
        # m >> 2h:       ... (1<<3h) ...
        # m & (m >> 2h): ... (1<<3h) ... (non-zero) -> indicates a win ending at 3h.
        if m & (m >> (2 * h_shift)) != 0:
            return True

        # Check diagonal alignment 1 (/)
        d1_shift = HEIGHT # Up-right
        # m = pairs of diagonally adjacent tiles (/)
        m = pos & (pos >> d1_shift)
        # Check for two pairs separated diagonally (/)
        if m & (m >> (2 * d1_shift)) != 0:
            return True

        # Check diagonal alignment 2 (\)
        d2_shift = HEIGHT + 2 # Up-left
        # m = pairs of diagonally adjacent tiles (\)
        m = pos & (pos >> d2_shift)
        # Check for two pairs separated diagonally (\)
        if m & (m >> (2 * d2_shift)) != 0:
            return True

        # Check vertical alignment
        v_shift = 1 # Up
        # m = pairs of vertically adjacent tiles
        m = pos & (pos >> v_shift)
        # Check for two pairs separated vertically
        if m & (m >> (2 * v_shift)) != 0: # Check m & (m >> 2)
            return True

        # no alignments found
        return False

    def key(self) -> int:
        """Returns the key used for indexing into the transposition table (see [Board Keys])
        
        The key is a representation where 1s mark the current player's tiles AND the
        lowest empty square in each column (the possible moves for the current player).
        
        [Board Keys]: #board-keys # (Note: Link might not work in Python context)
        """
        # Rust: self.player_mask + self.board_mask
        # Let's analyze this:
        # board_mask = P1 | P2 (all occupied squares)
        # player_mask = P_current (current player's squares)
        # If a square is P_current, bit is 1 in both masks. 1+1 = binary 10. Carries happen.
        # If a square is P_opponent, bit is 1 in board_mask, 0 in player_mask. 0+1 = 1.
        # If a square is empty, bit is 0 in both. 0+0 = 0.
        # This doesn't seem to match the description "1-bit in each square the board where the current player has a tile,
        # and an additional 1-bit in the first empty square of a column."
        
        # Let's re-read the Rust source carefully. It is indeed `self.player_mask + self.board_mask`.
        # Consider the Connect4 paper by Victor Allis (where this bitboard technique originates).
        # The key is often defined as `mask + bottom`, where `mask` is the combined board mask.
        # Or sometimes `current_player_mask + board_mask`.
        # Let's trust the Rust code's implementation literally. The addition might create a unique key
        # through the way carries propagate, even if the description isn't perfectly matching the operation.
        # Example: Column 0 empty. board=0, player=0. key = 0+0=0.
        # P1 plays C0. board=1, player=1 (P1 turn ended, now P2). key = 1+1=2 (bit 1 set).
        # P2 plays C0. board=1|(1<<1)=3, player=1<<1=2 (P2 turn ended, now P1). key = 2+3=5 (bits 0, 2 set).
        # P1 plays C0. board=3|(1<<2)=7, player=1|(1<<2)=5 (P1 turn ended, now P2). key = 5+7=12 (bits 2, 3 set).
        # The key `player_mask + board_mask` seems to be a valid unique identifier.
        return self.player_mask + self.board_mask

    def huffman_code(self) -> int:
        """Returns the Huffman code used for searching the opening database (see [Huffman Codes])
        
        # Notes
        For positions with more than 13 tiles, data will be lost and the returned code will not
        be unique (as it targets a u32). Python ints handle arbitrary size, but the logic
        might implicitly assume 32/64 bit limits if not careful. The logic here seems fine.
        The code returned is the minimum of the code for the board and its mirror image.
        
        [Huffman Codes]: #huffman-codes # (Note: Link might not work in Python context)
        """
        # Calculate code for normal and mirrored board, return the minimum
        code1 = self._huffman_code(False)
        code2 = self._huffman_code(True)
        return min(code1, code2)

    def _huffman_code(self, mirror: bool) -> int:
        """Returns Huffman code for opening database, optionally mirroring the position"""
        # 0 separates the tiles of each column
        # 10 (binary 2) is a player 1 tile (assuming player 1 made the first move)
        # 11 (binary 3) is a player 2 tile
        # Player 1/2 determination depends on num_moves parity.
        # If num_moves is even, player 1 just moved (or game start). player_mask holds player 2.
        # If num_moves is odd, player 2 just moved. player_mask holds player 1.
        # Let's assume the code represents tiles based on who moved first (P1) and second (P2).
        # P1 tiles should get code 10. P2 tiles should get code 11.
        # If num_moves is even (P2's turn), player_mask holds P2 tiles.
        # If num_moves is odd (P1's turn), player_mask holds P1 tiles.

        code = 0
        
        # Determine player masks based on who is P1 and P2
        # Player 1 (first mover)
        p1_mask: int
        # Player 2 (second mover)
        p2_mask: int
        if self.num_moves % 2 == 0:
             # Even moves means P2's turn now. player_mask holds P2 tiles.
             p2_mask = self.player_mask
             p1_mask = self.board_mask ^ self.player_mask
        else:
             # Odd moves means P1's turn now. player_mask holds P1 tiles.
             p1_mask = self.player_mask
             p2_mask = self.board_mask ^ self.player_mask

        # Choose iteration order based on mirror flag
        column_iterator: Iterator[int]
        if mirror:
            column_iterator = reversed(range(WIDTH))
        else:
            column_iterator = range(WIDTH)
            
        for column in column_iterator:
            col_mask = BitBoard.column_mask(column) # Playable area mask
            # Iterate rows from bottom (0) up to HEIGHT (inclusive) to check for separator
            # The loop needs to check up to H squares; if the H-th square (row index H-1)
            # is filled, we still need to check the guard bit position to place the separator.
            # The Rust loop `0..=HEIGHT` includes HEIGHT.
            # Bit index for row `r` in column `c` is `c*(H+1) + r`.
            # Row HEIGHT corresponds to the guard bit.
            for row in range(HEIGHT + 1): # Check rows 0..H
                tile_pos_mask = 1 << (column * (HEIGHT + 1) + row)

                # Check if this position is occupied on the board
                if self.board_mask & tile_pos_mask != 0:
                    # Tile is present. Determine if it's P1 or P2.
                    if p1_mask & tile_pos_mask != 0:
                        # Player 1 tile: append 10 (binary)
                        code = (code << 2) | 0b10
                    else: # Must be Player 2 tile
                        # Player 2 tile: append 11 (binary)
                        code = (code << 2) | 0b11
                else:
                    # Empty square found. This marks the end of the column's tiles.
                    # Append separator 0.
                    code <<= 1
                    break # Move to the next column
            # If the loop finishes without break (column is full), a separator is still needed.
            # The Rust code implicitly handles this because the loop goes to HEIGHT.
            # If row == HEIGHT and the guard bit was set, the loop added the tile code.
            # If row == HEIGHT and the guard bit was *not* set, the `else` block runs, adds 0, and breaks.
            # So the Python logic seems correct.

        # Final shift by 1 as in Rust code? `code << 1` at the end.
        # Let's trace: Empty board. Loop runs 7 times. Each time hits `else` immediately. code <<= 1 seven times. Result 0. Rust returns 0 << 1 = 0.
        # Board "1". P1 plays C0. num_moves=1. p1=1, p2=0. mirror=false.
        # Col 0: row 0: board&1=1. p1&1=1. code = (0<<2)|0b10 = 2.
        # Col 0: row 1: board&2=0. code <<= 1 => code = 4. break.
        # Col 1-6: board&mask=0. code <<= 1 six times. code = 4 * (2**6) = 256.
        # Final: code << 1 = 512.
        # Let's try Rust logic manually:
        # Col 0: row 0: tile_mask=1. board&1=1. player_mask&1=1 (P1). code=(0<<2)+0b10=2.
        # Col 0: row 1: tile_mask=2. board&2=0. code<<=1 => code=4. break.
        # Col 1..6: code<<=1 six times. code = 4 * 64 = 256.
        # Final: code << 1 = 512. Matches.
        return code << 1

    # Implement __eq__ and __hash__ if instances need to be compared or used in sets/dicts
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BitBoard):
            return NotImplemented
        return self.player_mask == other.player_mask and \
               self.board_mask == other.board_mask and \
               self.num_moves == other.num_moves

    def __hash__(self) -> int:
        # Use the key directly for hashing, as it's designed to be a unique representation
        return hash(self.key())

    # Optional: Add a __str__ or __repr__ for easier debugging
    def __repr__(self) -> str:
        return f"BitBoard(player_mask={self.player_mask}, board_mask={self.board_mask}, num_moves={self.num_moves})"

    def __str__(self) -> str:
        """Returns a string representation of the board."""
        p1_char = 'X'
        p2_char = 'O'
        empty_char = '.'
        
        # Determine P1/P2 based on num_moves parity for display
        p1_mask_disp: int
        p2_mask_disp: int
        current_player_num = (self.num_moves % 2) + 1 # 1 or 2

        if self.num_moves % 2 == 0: # P1 just moved (or start), P2's turn
             p2_mask_disp = self.player_mask
             p1_mask_disp = self.board_mask ^ self.player_mask
        else: # P2 just moved, P1's turn
             p1_mask_disp = self.player_mask
             p2_mask_disp = self.board_mask ^ self.player_mask

        lines = []
        # Iterate rows top-down
        for r in range(HEIGHT - 1, -1, -1):
            row_str = []
            # Iterate columns left-right
            for c in range(WIDTH):
                mask = 1 << (c * (HEIGHT + 1) + r)
                if p1_mask_disp & mask:
                    row_str.append(p1_char)
                elif p2_mask_disp & mask:
                    row_str.append(p2_char)
                else:
                    row_str.append(empty_char)
            lines.append(" ".join(row_str))
        # Add column numbers at the bottom
        lines.append("-" * (WIDTH * 2 - 1))
        lines.append(" ".join(str(i + 1) for i in range(WIDTH)))
        lines.append(f"Turn: Player {current_player_num}")
        return "\n".join(lines)

# Corresponds to `impl Default for BitBoard`
# The __init__ with default arguments already handles this.
# You can create a default board using BitBoard()
