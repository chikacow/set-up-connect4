# Required dependencies:
# Assuming the following modules exist in the same directory or are importable:
# - bitboard.py (containing BitBoard class, WIDTH, HEIGHT constants)
# - opening_database.py (containing OpeningDatabase class, DATABASE_DEPTH constant)
# - transposition_table.py (containing TranspositionTable class)

# Import necessary components from assumed modules
from bitboard import BitBoard, WIDTH, HEIGHT
from opening_database import OpeningDatabase, DATABASE_DEPTH
from transposition_table import TranspositionTable

import math
import copy # Used for deep copying the board state

# Note: Python's standard integers have arbitrary precision, unlike Rust's i32.
# This translation assumes standard Python integers are sufficient and that
# overflow/underflow behavior matching Rust's i32 is not critical for the algorithm's logic.
# The u8 type for transposition table values requires ensuring values are clamped or handled appropriately if necessary.

# The minimum possible score of a position
MIN_SCORE: int = -((WIDTH * HEIGHT)) // 2 + 3
# The maximum possible score of a postion
MAX_SCORE: int = ((WIDTH * HEIGHT) + 1) // 2 - 3

class MoveSorter:
    def __init__(self):
        self.size: int = 0
        # move bitmap, column and score
        # Initialize with placeholder tuples; size is managed by self.size
        self.moves: list[tuple[int, int, int]] = [(0, 0, 0)] * WIDTH

    def push(self, new_move: int, column: int, score: int):
        pos = self.size
        self.size += 1
        # Ensure we don't exceed the allocated list size (though Python lists grow, we mimic the fixed-size array behavior)
        if pos >= len(self.moves):
             # This case should ideally not happen if WIDTH is correct, but added for safety.
             # In Rust, this would be an out-of-bounds error. Python lists would resize.
             # To strictly mimic Rust, we might raise an error or pre-allocate exactly WIDTH.
             # Pre-allocating with [(0,0,0)] * WIDTH is done in __init__.
             pass # Or raise IndexError("MoveSorter capacity exceeded")

        # Insertion sort logic (descending score for Rust's pop logic)
        # Rust code sorts ascending and pops from the end (highest score first).
        # Python's pop() also removes from the end.
        # Let's re-read the Rust code:
        # `while pos != 0 && self.moves[pos - 1].2 > score` -> shifts elements with *higher* scores up
        # `self.moves[pos] = (new_move, column, score);` -> inserts the new element
        # This implements insertion sort in *ascending* order of score.
        # The iterator `next` then pops from the end (`self.size -= 1`), returning the highest score first.
        while pos != 0 and self.moves[pos - 1][2] > score:
            self.moves[pos] = self.moves[pos - 1]
            pos -= 1
        self.moves[pos] = (new_move, column, score)

    def __iter__(self):
        # Reset internal state for iteration if needed, though this implementation
        # consumes the sorter state directly like the Rust version.
        # A copy of size is needed if the sorter is iterated multiple times,
        # but the Rust version consumes the original.
        self._iter_size = self.size
        return self

    def __next__(self) -> tuple[int, int]:
        if self.size == 0:
            raise StopIteration
        else:
            self.size -= 1
            # Return the element with the highest score (last element in sorted array)
            return (self.moves[self.size][0], self.moves[self.size][1])

# Returns a list ordering the columns from the middle outwards, as
# the middle columns are often better moves
def move_order() -> list[int]:
    move_order_list = [0] * WIDTH
    i = 0
    while i < WIDTH:
        # Original Rust logic: move_order[i] = (WIDTH / 2) + (i % 2) * (i / 2 + 1) - (1 - i % 2) * (i / 2);
        # Python equivalent using integer division //
        center = WIDTH // 2
        offset_magnitude = (i // 2) + 1
        if i % 2 == 1: # Odd i: (i % 2) * (i / 2 + 1) part
            offset = offset_magnitude
        else: # Even i: -(1 - i % 2) * (i / 2) part -> -(i // 2)
             # Careful: Rust i/2 is integer division.
             offset = -(i // 2)

        move_order_list[i] = center + offset
        i += 1
    return move_order_list

# Precompute move order
_MOVE_ORDER = move_order()


# An agent to solve Connect 4 positions
#
# Notes
# This agent uses a classical game tree search with various optimisations to
# find the mathematically best move(s) in any position, thus 'solving' the game
#
# Position Scoring
# A position is scored by how far a forced win is from the start of the game for either player.
# If the first player wins with their final placed tile (their 21st tile in a 7x6 board)
# the score is 1, or -1 if the the second player wins with their final tile. Earlier wins
# have scores further from 0, up to 18/-18, where a player wins with their 4th tile. A drawn position
# has a score of 0
class Solver:
    # board: BitBoard # Type hint
    # node_count: int # Type hint
    # transposition_table: TranspositionTable # Type hint
    # opening_database: Optional[OpeningDatabase] # Type hint

    # Creates a new `Solver` from a bitboard
    # Overload __init__ using default arguments to simulate multiple constructors
    def __init__(self, board: BitBoard, transposition_table: TranspositionTable = None, opening_database: OpeningDatabase = None):
        self.board: BitBoard = board
        # The number of nodes searched by this `Solver` so far (for diagnostics only)
        self.node_count: int = 0
        self.transposition_table: TranspositionTable = transposition_table if transposition_table is not None else TranspositionTable()
        self.opening_database: OpeningDatabase | None = opening_database # Use | None for Optional type hint

    # Creates a new `Solver` from a bitboard with a given transposition table
    # This is handled by the default argument in __init__
    # def new_with_transposition_table(board: BitBoard, transposition_table: TranspositionTable) -> 'Solver':
    #     return Solver(board, transposition_table)
    # We can add a class method for clarity if desired:
    @classmethod
    def new_with_transposition_table(cls, board: BitBoard, transposition_table: TranspositionTable) -> 'Solver':
         return cls(board, transposition_table=transposition_table)

    # Adds an opening database to an existing `Solver`
    def with_opening_database(self, opening_database: OpeningDatabase) -> 'Solver':
        self.opening_database = opening_database
        return self # Return self to allow chaining like in Rust

    # Performs game tree search
    #
    # Returns the score of the position (see [Position Scoring])
    #
    # [Position Scoring]: #position-scoring
    def negamax(self, alpha: int, beta: int) -> int:
        # Make alpha and beta mutable within the function scope (Python parameters are mutable if they are mutable types, but ints are immutable, reassignment creates a new binding)
        current_alpha = alpha
        current_beta = beta

        self.node_count += 1

        # check for next-move win for current player
        for column in range(WIDTH):
            if self.board.playable(column) and self.board.check_winning_move(column):
                # Rust: ((WIDTH * HEIGHT + 1 - self.board.num_moves()) / 2) as i32;
                return (WIDTH * HEIGHT + 1 - self.board.num_moves()) // 2

        # look for moves that don't give the opponent a next turn win
        non_losing_moves = self.board.non_losing_moves()
        if non_losing_moves == 0:
            # Rust: -((WIDTH * HEIGHT) as i32 - self.board.num_moves() as i32) / 2;
            return - (WIDTH * HEIGHT - self.board.num_moves()) // 2

        # check for draw
        if self.board.num_moves() == WIDTH * HEIGHT:
            return 0

        # check opening table at appropriate depth
        # Ensure DATABASE_DEPTH is imported or defined
        if self.board.num_moves() == DATABASE_DEPTH:
            if self.opening_database is not None:
                # Use the board's Huffman code representation for lookup
                score = self.opening_database.get(self.board.huffman_code())
                if score is not None:
                    return score

        # upper bound of score
        # Rust: let mut max = (((WIDTH * HEIGHT) - 1 - self.board.num_moves()) / 2) as i32;
        max_score_bound = ((WIDTH * HEIGHT) - 1 - self.board.num_moves()) // 2

        # try to fetch the upper/lower bound of the score from the transposition table
        key = self.board.key()
        # Rust: let value = self.transposition_table.get(key) as i32;
        # Python TT get should return int or None. Assuming 0 means not found like in Rust.
        value = self.transposition_table.get(key) # Assuming get returns int, 0 if not found
        if value != 0:
            # check if lower bound
            # Rust: if value > MAX_SCORE - MIN_SCORE + 1
            if value > MAX_SCORE - MIN_SCORE + 1:
                # Rust: let min_bound = value + 2 * MIN_SCORE - MAX_SCORE - 2;
                min_bound = value + 2 * MIN_SCORE - MAX_SCORE - 2
                if current_alpha < min_bound:
                    current_alpha = min_bound
                    if current_alpha >= current_beta:
                        # prune the exploration
                        return current_alpha
            # else upper bound
            else:
                # Rust: let max_bound = value + MIN_SCORE - 1;
                max_bound = value + MIN_SCORE - 1
                if current_beta > max_bound:
                    current_beta = max_bound
                    if current_alpha >= current_beta:
                        # prune the exploration
                        return current_beta
            # This line seems redundant in Rust? max_score_bound is updated below anyway.
            # Let's translate it directly for exactness.
            # Rust: max_score_bound = value + MIN_SCORE - 1;
            # This seems to be updating the general max_score_bound based on TT entry,
            # regardless of whether it was lower or upper bound info.
            # Let's re-evaluate the Rust logic:
            # If it's a lower bound (value > ...), it means score >= min_bound.
            # If it's an upper bound (value <= ...), it means score <= max_bound = value + MIN_SCORE - 1.
            # The line `max = value + MIN_SCORE - 1;` seems to only apply if it was an upper bound.
            # Let's adjust the Python code to match this interpretation.
            if not (value > MAX_SCORE - MIN_SCORE + 1): # If it was an upper bound
                 max_score_bound = value + MIN_SCORE - 1


        if current_beta > max_score_bound:
            # clamp beta to calculated upper bound
            current_beta = max_score_bound
            # if the upper bound is lower than alpha, we can prune the exploration
            if current_alpha >= current_beta:
                return current_beta

        moves = MoveSorter()
        # reversing move order to put edges first reduces the amount of sorting
        # as these moves are worse on average
        # Rust: for i in (0..WIDTH).rev()
        for i in range(WIDTH - 1, -1, -1):
            column = _MOVE_ORDER[i]
            # Assuming BitBoard.column_mask is a static/class method or accessible
            candidate = non_losing_moves & BitBoard.column_mask(column)
            # Check playability after getting the candidate move mask
            if candidate != 0 and self.board.playable(column):
                moves.push(candidate, column, self.board.move_score(candidate))

        # search the next level of the tree
        # The MoveSorter is consumed here
        for move_bitmap, _column in moves:
            # Rust: let mut next = self.clone();
            # Python: Need to create a new state for the recursive call.
            # The board needs to be copied. TT and DB are shared.
            # We don't need a full Solver clone, just the board state.
            next_board = self.board.copy() # Assuming BitBoard has a copy method
            next_board.play(move_bitmap)

            # Create a temporary solver instance for the recursive call, sharing TT and DB
            # Or, more simply, modify self.board temporarily and revert, but that's not what clone does.
            # Let's simulate clone by creating a new Solver instance with the copied board
            # and shared resources. This is closer to Rust's behavior if Solver itself isn't huge.
            # However, the node_count accumulation suggests modifying the *current* solver's count
            # based on the recursive call's count. This implies the recursive call should perhaps
            # be on a temporary object or a method that takes the board state.
            # Let's stick to the clone simulation first.

            # Create a temporary solver instance for the recursive call
            # Pass the *same* transposition table and opening database references
            recursive_solver = Solver(next_board, self.transposition_table, self.opening_database)
            recursive_solver.node_count = 0 # Reset node count for the sub-search

            # the search window is flipped for the other player
            score = -recursive_solver.negamax(-current_beta, -current_alpha)

            # Accumulate node count from the recursive call
            self.node_count += recursive_solver.node_count

            # if a child node's score is better than beta, we can prune the tree
            # here because a perfect opponent will not pick this branch
            if score >= current_beta:
                # save a lower bound of the score
                # Rust: self.transposition_table.set(key, (score + MAX_SCORE - 2 * MIN_SCORE + 2) as u8);
                # Ensure the value fits in u8 (0-255). Python ints handle large numbers,
                # so explicit clamping/modulo might be needed if the TT implementation expects u8.
                tt_value_lower = score + MAX_SCORE - 2 * MIN_SCORE + 2
                # Assuming transposition_table.set handles potential u8 conversion/clamping if necessary
                self.transposition_table.set(key, tt_value_lower)
                return score # Return beta (fail-high)

            if score > current_alpha:
                current_alpha = score

        # offset of one to prevent putting a 0, which represents an empty entry
        # Rust: self.transposition_table.set(self.board.key(), (alpha - MIN_SCORE + 1) as u8);
        # Use the final alpha value for the upper bound store
        tt_value_upper = current_alpha - MIN_SCORE + 1
        # Assuming transposition_table.set handles potential u8 conversion/clamping if necessary
        self.transposition_table.set(self.board.key(), tt_value_upper)
        return current_alpha # Return alpha (best score found)

    # Performs a top-level search, bypassing transposition table and opening database
    #
    # Returns the score of the position and the calculated best move
    def top_level_search(self, alpha: int, beta: int) -> tuple[int, int]:
        # Make alpha mutable
        current_alpha = alpha

        self.node_count += 1

        # check for win for current player on this move
        for column in range(WIDTH):
            if self.board.playable(column) and self.board.check_winning_move(column):
                # Rust: ((WIDTH * HEIGHT + 1 - self.board.num_moves()) / 2) as i32
                score = (WIDTH * HEIGHT + 1 - self.board.num_moves()) // 2
                return (score, column)

        # look for moves that don't give the opponent a next turn win
        non_losing_moves = self.board.non_losing_moves()
        if non_losing_moves == 0:
            # all moves lose, return the first legal move found
            # Rust: let first = (0..WIDTH).find(|&i| self.board.playable(i)).unwrap();
            first_legal_move = -1 # Should always find one if non_losing_moves is 0 but board isn't full
            for i in range(WIDTH):
                if self.board.playable(i):
                    first_legal_move = i
                    break
            # Rust code unwraps, assuming a move is always possible if not drawn/won
            # Add an assertion or proper error handling if needed
            if first_legal_move == -1:
                 # This case implies board is full but wasn't caught earlier, or playable is inconsistent
                 raise Exception("No playable moves found in a non-terminal state")

            # Rust: -((WIDTH * HEIGHT) as i32 - self.board.num_moves() as i32) / 2
            score = -(WIDTH * HEIGHT - self.board.num_moves()) // 2
            return (score, first_legal_move)

        # check for draw (no valid moves) - This check seems redundant given the non_losing_moves check above?
        # If non_losing_moves > 0, there are moves. If num_moves == WIDTH*HEIGHT, it's a draw.
        # The Rust code checks for draw *after* checking non_losing_moves. Let's keep it.
        if self.board.num_moves() == WIDTH * HEIGHT:
             # Return score 0 and an invalid column index (like WIDTH) to indicate draw?
             # Rust returns (0, WIDTH). Let's match that.
            return (0, WIDTH)

        moves = MoveSorter()
        # Rust: for i in (0..WIDTH).rev()
        for i in range(WIDTH - 1, -1, -1):
            column = _MOVE_ORDER[i]
            candidate = non_losing_moves & BitBoard.column_mask(column)
            if candidate != 0 and self.board.playable(column):
                moves.push(candidate, column, self.board.move_score(candidate))

        # search the next level of the tree and keep track of the best move
        best_score = MIN_SCORE # Initialize with the lowest possible score
        best_move = WIDTH # Initialize with an invalid column index

        # Initialize best_move with the first move from the sorter if available
        # This ensures a valid move is returned even if all scores are <= alpha initially.
        # The Rust code implicitly handles this by iterating and updating best_move.
        # Let's peek at the first move without consuming it from the iterator.
        # The MoveSorter iterator consumes state, so we need a way to get the first move.
        # Or, initialize best_move inside the loop on the first iteration.
        first_move_peek = None
        if moves.size > 0:
             # The highest score move is at index moves.size - 1
             first_move_peek = moves.moves[moves.size - 1][1] # Get column
             best_move = first_move_peek # Initialize best_move

        for move_bitmap, column in moves: # Consumes the MoveSorter
            # Simulate clone for recursive call
            next_board = self.board.copy()
            next_board.play(move_bitmap)

            recursive_solver = Solver(next_board, self.transposition_table, self.opening_database)
            recursive_solver.node_count = 0

            # the search window is flipped for the other player
            # Call negamax for deeper search, not top_level_search
            score = -recursive_solver.negamax(-beta, -current_alpha)

            self.node_count += recursive_solver.node_count

            # if the actual score is better than beta, we can prune the tree
            # because the other player will not pick this branch
            if score >= beta:
                return (score, column) # Return beta-cutoff score and the move causing it

            if score > current_alpha:
                current_alpha = score # Update alpha

            # Update best score and move found so far
            if score > best_score:
                best_score = score
                best_move = column

        # After checking all moves, return the best score found (which is <= alpha)
        # and the corresponding best move.
        # The final score returned should be the refined alpha.
        return (current_alpha, best_move)


    # Calculate the score and best move of the current position with iterative deepening
    def solve(self) -> tuple[int, int]:
        return self._solve(True) # silent = True

    # Calculate the score and best move of the current position with iterative deepening, logging progress to stdout
    def solve_verbose(self) -> tuple[int, int]:
        return self._solve(False) # silent = False

    # Performs the iterative deepening search, returning position score and best move
    def _solve(self, silent: bool) -> tuple[int, int]:
        # Rust: let mut min = -(((WIDTH * HEIGHT) as i32) - self.board.num_moves() as i32) / 2;
        min_bound = -(WIDTH * HEIGHT - self.board.num_moves()) // 2
        # Rust: let mut max = (WIDTH * HEIGHT + 1 - self.board.num_moves()) as i32 / 2;
        max_bound = (WIDTH * HEIGHT + 1 - self.board.num_moves()) // 2

        next_move = WIDTH # Initialize with invalid column

        # iteratively narrow the search window for iterative deepening
        while min_bound < max_bound:
            # Rust: let mut mid = min + (max - min) / 2;
            mid = min_bound + (max_bound - min_bound) // 2
            # tweak the search value for both negative and positive searches
            if mid <= 0 and min_bound // 2 < mid:
                mid = min_bound // 2
            elif mid >= 0 and max_bound // 2 > mid:
                mid = max_bound // 2

            # log progress to stdout
            if not silent:
                # Calculate search depth and uncertainty
                current_ply = self.board.num_moves()
                total_ply = WIDTH * HEIGHT
                # Depth can be seen as remaining plies
                remaining_ply = total_ply - current_ply
                # Score relates to distance from end; abs(score) relates to how quick the win/loss is.
                # A higher abs(score) means fewer moves remaining *after* the win.
                # Let's use the Rust formula's components:
                # score = +/- ( (W*H+1)/2 - moves_to_win )
                # moves_to_win = (W*H+1)/2 - abs(score)
                # The number in the print seems to be: total_plies - current_plies - abs(bound)
                # This doesn't seem right. Let's re-read the Rust print:
                # (WIDTH * HEIGHT - self.board.num_moves()) as i32 - min.abs().min(max.abs())
                # This is: remaining_plies - min(abs(min_bound), abs(max_bound))
                # This represents roughly how many plies deep the search is *certain* about the outcome.
                # Let's try to replicate this:
                certainty_depth_offset = min(abs(min_bound), abs(max_bound))
                search_depth_metric = remaining_ply - certainty_depth_offset
                uncertainty = max_bound - min_bound

                print(
                    f"Search depth: {search_depth_metric}/{remaining_ply}, uncertainty: {uncertainty}"
                )

            # use a null-window search (alpha=mid, beta=mid+1) to determine if the actual score is greater or less than mid
            # Call top_level_search for the root node exploration
            (r_score, best_move) = self.top_level_search(mid, mid + 1)
            # Store the best move found during this iteration
            # The best move from the last iteration where min_bound == max_bound is the true best move.
            # The move returned by top_level_search corresponds to the score `r_score`.
            # If r_score > mid, it means the true score is > mid, and `best_move` is a candidate for the overall best move.
            # If r_score <= mid, it means the true score is <= mid. The returned `best_move` might not be optimal globally.
            # The Rust code updates `next_move = best_move` unconditionally in the loop. Let's follow that.
            next_move = best_move

            # r is not necessarily the exact true score, but its value indicates
            # whether the true score is above or below the search target (mid)
            if r_score <= mid:
                # actual score <= mid, so narrow the upper bound
                max_bound = r_score
            else:
                # actual score > mid, so narrow the lower bound
                min_bound = r_score
                # Note: In fail-soft PVS/NWS, if score > alpha (mid here), the returned score `r`
                # is a lower bound. If score <= alpha, the returned score `r` is an upper bound.
                # top_level_search returns alpha if score <= alpha, and score if score >= beta.
                # If top_level_search returns r > mid (alpha), it means the score was >= mid+1 (beta).
                # The returned r is a lower bound. So `min_bound = r` is correct.
                # If top_level_search returns r <= mid (alpha), the score was <= mid.
                # The returned r is an upper bound. So `max_bound = r` is correct.

        # min and max should be equal here
        # The final value (min_bound or max_bound) is the true score.
        # The `next_move` holds the best move found in the iteration that determined the final score.
        return (min_bound, next_move)

    # Converts a position score to a win distance in a single player's moves
    def score_to_win_distance(self, score: int) -> int:
        # Rust: match score.cmp(&0)
        if score == 0: # Draw
             # Rust: WIDTH * HEIGHT - self.board.num_moves()
             # This seems to return remaining moves? A draw takes all moves.
             # Let's return remaining moves as per Rust code.
            return WIDTH * HEIGHT - self.board.num_moves()
        elif score > 0: # Current player wins
            # Rust: (WIDTH * HEIGHT / 2 + 1 - score as usize) - self.board.num_moves() / 2
            # Win distance = moves for current player until win
            # Score = (W*H+1)/2 - total_moves_by_winner
            # total_moves_by_winner = (W*H+1)/2 - score
            # current_moves_by_winner = ceil(num_moves / 2) if current player is P1, floor if P2?
            # Let's assume num_moves starts at 0. P1 plays at 0, 2, 4... P2 plays at 1, 3, 5...
            # Moves by current player = ceil(num_moves / 2)
            # Rust uses integer division: self.board.num_moves() / 2
            # Let's use integer division //
            # Rust formula seems complex. Let's re-derive.
            # Max score = (W*H+1)/2 - 3 (win on move 4)
            # Min score = -(W*H)/2 + 3 (loss on move 4)
            # Score 1 = win on last move (move 21 for P1 on 7x6)
            # Score -1 = loss on last move (move 21 for P2 on 7x6)
            # Total moves played = self.board.num_moves()
            # If score > 0, current player wins.
            # Let total_moves_to_win be the total number of moves on the board when the win occurs.
            # If winner is P1 (played ceil(total_moves_to_win/2) moves), score = (W*H+1)/2 - ceil(total_moves_to_win/2) ? No, score definition is simpler.
            # Score = (BoardSize + 1) / 2 - MovesPlayedByWinner
            # BoardSize = W*H = 42
            # Max Score = (42+1)/2 - 4 = 21.5 - 4 = 17.5 -> 18? (Rust uses integer division)
            # MAX_SCORE = (42+1)//2 - 3 = 21 - 3 = 18
            # MIN_SCORE = -(42)//2 + 3 = -21 + 3 = -18
            # Score = (W*H+1)//2 - moves_by_winner
            # moves_by_winner = (W*H+1)//2 - score
            # Win distance = moves_by_winner - current_moves_by_this_player
            # current_moves_by_this_player = ceil(self.board.num_moves() / 2)
            # Win distance = (W*H+1)//2 - score - ceil(self.board.num_moves() / 2)
            # Let's check the Rust formula:
            # (WIDTH * HEIGHT // 2 + 1 - score) - self.board.num_moves() // 2
            # This is ( (W*H)//2 + 1 - score ) - floor(current_moves / 2)
            # It seems slightly off from the derivation. Let's trust the Rust code for now.
            # Note: usize conversion is implicit in Python for positive numbers.
            return (WIDTH * HEIGHT // 2 + 1 - score) - self.board.num_moves() // 2
        else: # score < 0, Current player loses
            # Rust: (WIDTH * HEIGHT / 2 + 1) - (-score as usize) - self.board.num_moves() / 2
            # This is equivalent to: (W*H)//2 + 1 + score - floor(current_moves / 2)
            # Let's verify: Opponent wins.
            # Score = - ( (W*H+1)//2 - moves_by_opponent )
            # -Score = (W*H+1)//2 - moves_by_opponent
            # moves_by_opponent = (W*H+1)//2 + score
            # Win distance for opponent = moves_by_opponent - current_moves_by_opponent
            # current_moves_by_opponent = floor(self.board.num_moves() / 2)
            # Win distance for opponent = (W*H+1)//2 + score - floor(self.board.num_moves() / 2)
            # The Rust formula calculates this opponent win distance.
            return (WIDTH * HEIGHT // 2 + 1) + score - self.board.num_moves() // 2


    # Emulating Deref: Provide direct access to the board attribute.
    # Python doesn't have Deref, users will access `solver.board` directly.
    # No explicit code needed here for that, just ensure `self.board` is public.

# Example Placeholder Implementations (replace with actual modules)

# --- Placeholder bitboard.py ---
class BitBoard:
    # Define WIDTH and HEIGHT here if not imported globally
    # WIDTH = 7
    # HEIGHT = 6
    def __init__(self):
        # Minimal state for placeholder methods
        self._num_moves = 0
        self._current_player_mask = 0 # Example state
        self._opponent_mask = 0       # Example state
        self._heights = [0] * WIDTH   # Example: track column heights

    def playable(self, column: int) -> bool:
        # Placeholder: Assume column is playable if not full
        return self._heights[column] < HEIGHT

    def check_winning_move(self, column: int) -> bool:
        # Placeholder: Assume no immediate win
        return False

    def num_moves(self) -> int:
        return self._num_moves

    def non_losing_moves(self) -> int:
        # Placeholder: Return a bitmask of all non-full columns
        # Assuming a loss doesn't happen immediately by playing there
        mask = 0
        for i in range(WIDTH):
            if self.playable(i):
                 # Check if playing in column 'i' creates an immediate threat for the opponent
                 # For placeholder, assume no move loses immediately unless it's the only option
                 # A more realistic placeholder might return all playable columns.
                 mask |= (1 << i) # Use column index as bit position for simplicity
        # A better placeholder might be based on column masks if the real BitBoard uses them
        # return (1 << WIDTH) - 1 # Assume all columns are non-losing for now
        return mask if mask != 0 else ((1 << WIDTH) -1) # Simplistic: return all if mask is 0

    def huffman_code(self) -> int:
        # Placeholder: Return a dummy code
        return hash((self._current_player_mask, self._opponent_mask))

    def key(self) -> int:
        # Placeholder: Return a dummy key for TT
        # Key should ideally include current player information
        player_bit = self.num_moves() % 2
        return hash((self._current_player_mask, self._opponent_mask, player_bit))

    @staticmethod
    def column_mask(column: int) -> int:
        # Placeholder: Return a simple mask, assuming it represents the column index
        # The actual implementation likely returns a bitmask for the lowest available cell in that column.
        # Let's return 1 shifted by column index for simplicity in the solver logic.
        return 1 << column # Simple placeholder, real one depends on board representation

    def move_score(self, move_bitmap: int) -> int:
        # Placeholder: Return a neutral score
        # Actual score likely depends on heuristics (center control, threats, etc.)
        return 0

    def play(self, move_bitmap: int):
        # Placeholder: Update minimal state
        self._num_moves += 1
        # Find column from move_bitmap (assuming simple 1 << column format)
        column = -1
        for i in range(WIDTH):
            if (move_bitmap >> i) & 1:
                column = i
                break
        if column != -1 and self._heights[column] < HEIGHT:
             self._heights[column] += 1
        # Swap player masks (conceptual)
        # self._current_player_mask, self._opponent_mask = self._opponent_mask, self._current_player_mask
        # Add the move (conceptual)
        # In a real bitboard, update the masks based on the actual cell played.

    def copy(self) -> 'BitBoard':
        # Placeholder: Return a deep copy
        new_board = BitBoard()
        new_board._num_moves = self._num_moves
        new_board._current_player_mask = self._current_player_mask
        new_board._opponent_mask = self._opponent_mask
        new_board._heights = list(self._heights) # Copy the list
        return new_board

# --- Placeholder opening_database.py ---
DATABASE_DEPTH = 10 # Example value

class OpeningDatabase:
    def __init__(self):
        # Placeholder: Use a dictionary for storage
        self._data = {}

    def get(self, huffman_code: int) -> int | None:
        # Placeholder: Return value from dict or None
        return self._data.get(huffman_code)

    def add_entry(self, huffman_code: int, score: int):
         # Helper to populate the placeholder DB if needed
         self._data[huffman_code] = score


# --- Placeholder transposition_table.py ---
class TranspositionTable:
    def __init__(self, size: int = 1000000): # Example size
        # Placeholder: Use a dictionary
        # A real TT would use hashing, handle collisions, and possibly fixed size arrays.
        self._table = {}
        # Store count for diagnostics if needed
        self._entry_count = 0

    def get(self, key: int) -> int:
        # Placeholder: Return value or 0 if not found
        return self._table.get(key, 0)

    def set(self, key: int, value: int):
        # Placeholder: Store value in dict
        # The Rust code uses u8. If strict emulation is needed, clamp/check value.
        # value_u8 = max(0, min(255, value)) # Clamp to 0-255 if TT expects u8 range
        # self._table[key] = value_u8
        # For now, store the int directly, assuming consumer handles it.
        if key not in self._table:
             self._entry_count += 1
        self._table[key] = value
        # Optional: Implement replacement strategy if size-limited

    def __len__(self) -> int:
        return self._entry_count # Or len(self._table)

# Example Usage (requires WIDTH and HEIGHT defined, e.g., in bitboard.py or globally)
# Assuming bitboard.py defines WIDTH=7, HEIGHT=6
if __name__ == '__main__':
    # Need to define WIDTH and HEIGHT if not imported from bitboard
    # WIDTH = 7
    # HEIGHT = 6
    # MIN_SCORE = -((WIDTH * HEIGHT)) // 2 + 3
    # MAX_SCORE = ((WIDTH * HEIGHT) + 1) // 2 - 3

    print(f"Board: {WIDTH}x{HEIGHT}")
    print(f"Score Range: [{MIN_SCORE}, {MAX_SCORE}]")
    print(f"Move Order: {_MOVE_ORDER}")

    # Create a board (using placeholder)
    initial_board = BitBoard()

    # Create a solver
    solver = Solver(initial_board)

    # Optionally add an opening database (using placeholder)
    # db = OpeningDatabase()
    # solver = solver.with_opening_database(db)

    print("\nSolving initial position (verbose)...")
    # Reset node count before solving
    solver.node_count = 0
    score, best_move = solver.solve_verbose()

    print(f"\nSolver finished.")
    print(f"Nodes searched: {solver.node_count}")
    print(f"Position Score: {score}")
    if best_move < WIDTH:
        print(f"Best Move (column index): {best_move}")
        win_dist = solver.score_to_win_distance(score)
        if score > 0:
            print(f"Predicted win in {win_dist} moves for current player.")
        elif score < 0:
            print(f"Predicted loss in {win_dist} moves for current player (opponent wins).")
        else:
            print(f"Predicted draw.")
    else:
         print("Position is a draw (no moves possible or returned).")

    # Example of making a move and solving again
    if best_move < WIDTH:
        print(f"\nMaking move in column {best_move}...")
        # Need a way to get the move bitmap for the best move column
        # Placeholder: Assume simple mask
        move_mask = BitBoard.column_mask(best_move)
        solver.board.play(move_mask) # Modify the solver's board state

        print("Solving new position...")
        solver.node_count = 0
        score, best_move = solver.solve() # Use non-verbose solve

        print(f"\nSolver finished.")
        print(f"Nodes searched: {solver.node_count}")
        print(f"Position Score: {score}")
        if best_move < WIDTH:
            print(f"Best Move (column index): {best_move}")
            win_dist = solver.score_to_win_distance(score)
            if score > 0:
                print(f"Predicted win in {win_dist} moves for current player.")
            elif score < 0:
                print(f"Predicted loss in {win_dist} moves for current player (opponent wins).")
            else:
                print(f"Predicted draw.")
        else:
             print("Position is a draw (no moves possible or returned).")

