# Required dependencies:
# pip install tqdm
import struct
import time
import os
import io
import math
from multiprocessing import Process, Queue, Pool, cpu_count
from queue import Empty as QueueEmpty # Renamed to avoid conflict with multiprocessing.Queue
import threading # Used for the progress bar updater thread in generate
from typing import List, Tuple, Optional, Any, Dict
from tqdm import tqdm
import copy # Although Rc is about shared ownership, Python handles this by default. copy might be needed if mutation were involved, but it's not here.

# --- Placeholder implementations for external dependencies ---
# These should be replaced with the actual translated code from
# crate::{bitboard::*, solver::*, HEIGHT, WIDTH}

WIDTH = 7
HEIGHT = 6

def move_order() -> List[int]:
    """Returns the preferred order of columns to check."""
    # Original implementation likely centers columns: 3, 2, 4, 1, 5, 0, 6
    return [WIDTH // 2 + (1 - 2 * (i % 2)) * (i + 1) // 2 for i in range(WIDTH)]

class BitBoard:
    """Placeholder for the BitBoard struct."""
    def __init__(self, player_mask: int = 0, board_mask: int = 0, num_moves: int = 0):
        self.player_mask = player_mask
        self.board_mask = board_mask
        self.num_moves = num_moves
        # Add other necessary state based on the actual BitBoard implementation

    @staticmethod
    def from_slice(moves: List[int]) -> Optional['BitBoard']:
        """Creates a BitBoard from a sequence of moves. Returns None if invalid."""
        # This needs the actual logic for building the board state from moves.
        # For the placeholder, assume valid if moves list is not empty.
        if not moves:
            return None

        # Simulate board creation (highly simplified)
        bb = BitBoard()
        bb.num_moves = len(moves)
        # In a real implementation, update player_mask and board_mask based on moves
        # For placeholder purposes, generate some deterministic masks based on moves
        bb.player_mask = sum(1 << (i * 3 + m) for i, m in enumerate(moves) if i % 2 == 0)
        bb.board_mask = sum(1 << (i * 3 + m) for i, m in enumerate(moves))

        # Simulate potential failure (e.g., invalid move sequence)
        # Real implementation would check for column overflows, etc.
        if any(m < 0 or m >= WIDTH for m in moves):
             return None # Invalid move

        # Check if number of moves matches DATABASE_DEPTH for generation context
        if len(moves) == DATABASE_DEPTH:
             return bb
        else:
             # If used outside generation, might need different logic
             # For now, return a board if moves seem plausible
             return bb


    @staticmethod
    def from_parts(player_mask: int, board_mask: int, num_moves: int) -> 'BitBoard':
        """Creates a BitBoard from its constituent parts."""
        return BitBoard(player_mask, board_mask, num_moves)

    def huffman_code(self) -> int:
        """Returns the Huffman code for the board position."""
        # Placeholder: Use a combination of masks. Real implementation is complex.
        # Ensure it returns a 32-bit unsigned integer equivalent.
        # Using a simple hash for placeholder purposes.
        code = hash((self.player_mask, self.board_mask)) & 0xFFFFFFFF
        return code

    def player_mask(self) -> int:
        """Returns the player's mask."""
        return self.player_mask

    def board_mask(self) -> int:
        """Returns the combined board mask."""
        return self.board_mask

    def playable(self, col: int) -> bool:
        """Checks if a move in the given column is possible."""
        # Placeholder: Assume playable if column is valid. Real check involves height.
        return 0 <= col < WIDTH
        # Real check: (self.board_mask & top_mask(col)) == 0

    def check_winning_move(self, col: int) -> bool:
        """Checks if playing in the given column wins the game."""
        # Placeholder: Assume no winning move for simplicity.
        # Real implementation involves checking for 4-in-a-row after the move.
        return False

class Solver:
    """Placeholder for the Solver struct."""
    def __init__(self, board: BitBoard):
        self.board = board
        # Add other necessary state like transposition tables etc.

    def solve(self) -> Tuple[int, Any]:
        """Solves the position, returning score and potentially other info."""
        # Placeholder: Return a dummy score. Real implementation involves search.
        # Score range seems to be small enough for i8 in Rust (-128 to 127).
        # Let's return a score based on masks for deterministic placeholder behavior.
        score = (self.board.player_mask % 10) - (self.board.board_mask % 10)
        # Ensure score fits within i8 range for consistency
        score = max(-128, min(127, score))
        return score, None # Returning None for the second element (move info?)

# --- End of Placeholder implementations ---


# Helper function for human-readable duration (similar to Rust's HumanDuration)
def format_duration(seconds: float) -> str:
    """Formats duration in seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    else:
        return f"{seconds / 3600:.1f}hr"

#! A searchable store of Connect 4 positions to speed up early-game searches
#!

# Note: anyhow::Result is replaced by standard Python exception handling (try/except)

# Note: byteorder::{BigEndian, ReadBytesExt, WriteBytesExt} is replaced by Python's struct module

# Note: indicatif is replaced by tqdm

# Note: rayon::prelude is replaced by Python's multiprocessing module

# Note: std::cmp::Ordering is replaced by Python's standard comparison operators

# Note: std::fs::{File, OpenOptions} is replaced by Python's open()

# Note: std::io::{BufReader, BufWriter, Read} is replaced by Python's io module and file objects

# Note: std::rc::Rc is handled implicitly by Python's reference counting / garbage collection

# Note: std::sync::mpsc is replaced by multiprocessing.Queue

# Note: std::thread is replaced by multiprocessing.Process

# Note: std::time is replaced by Python's time module

# Note: crate::{bitboard::*, solver::*, HEIGHT, WIDTH} are imported or defined above as placeholders



DATABASE_PATH: str = "opening_database.bin"

TEMP_FILE_PATH: str = "temp_positions.bin"

DATABASE_DEPTH: int = 12

DATABASE_NUM_POSITIONS: int = 4200899 # Expected number of positions after filtering and deduplication

# Note: This value seems specific to the Rust implementation's filtering.
# The Python generation might yield a different number if filtering differs slightly.
# The Rust code uses this size to pre-allocate vectors when loading.
# Python lists grow dynamically, but we use it for progress bars and loading checks.
# The Rust code also hardcodes a total generation count (8532690438) for the progress bar.
# We will estimate this or use total=None if the exact number isn't critical for Python's tqdm.
# Let's keep the Rust value for now for the progress bar target.
TOTAL_GENERATION_TARGET = 8532690438


class OpeningDatabaseStorage:
    """Internal storage for the opening database."""
    def __init__(self, positions: List[int], values: List[int]):
        # Note: Rust uses Vec<u32> and Vec<i8>. Python uses lists of ints.
        self.positions: List[int] = positions
        self.values: List[int] = values # Store as int, was i8

    @staticmethod
    def load() -> 'OpeningDatabaseStorage':
        """Loads the database from the specified path."""
        # Note: BufReader is implicitly handled by io.BufferedReader or default file buffering
        try:
            with open(DATABASE_PATH, 'rb') as f:
                # Determine actual size based on file content if possible, or use constant
                # For simplicity, let's read until EOF or use the constant size
                positions = []
                values = []
                # Read pairs of (u32, i8)
                entry_format = '>Ii' # Big-endian u32, big-endian i8 (struct uses 'b' for signed char)
                entry_size = struct.calcsize('>Ib') # 4 bytes for u32, 1 byte for i8

                # Check file size consistency if needed
                # file_size = os.path.getsize(DATABASE_PATH)
                # if file_size != DATABASE_NUM_POSITIONS * entry_size:
                #     print(f"Warning: File size {file_size} does not match expected size {DATABASE_NUM_POSITIONS * entry_size}")
                #     # Adjust num_positions or handle error

                num_read = 0
                while True:
                    data = f.read(entry_size)
                    if not data:
                        break
                    if len(data) < entry_size:
                         raise IOError("Incomplete data read from database file.")
                    # Unpack u32 (I) and i8 (b)
                    pos_code, val = struct.unpack('>Ib', data)
                    positions.append(pos_code)
                    values.append(val)
                    num_read += 1

                # Optional: Verify against DATABASE_NUM_POSITIONS if strictly required
                if num_read != DATABASE_NUM_POSITIONS:
                     print(f"Warning: Read {num_read} entries, expected {DATABASE_NUM_POSITIONS}")

                return OpeningDatabaseStorage(positions, values)
        except FileNotFoundError:
            print(f"Error: Database file not found at {DATABASE_PATH}")
            raise # Re-raise the exception
        except Exception as e:
            print(f"Error loading database: {e}")
            raise # Re-raise

    def get(self, position_code: int) -> Optional[int]:
        """Retrieves the score for a position using binary search."""
        # variables for binary search state
        n = len(self.positions)
        if n == 0:
            return None

        step = n - 1
        pos1 = step

        # invalid value placeholder
        value = -99

        # Binary search - replicating the Rust logic carefully
        while step > 0:
            # divide step by 2, always rounding up apart from at 0.5
            step = (step + (step & 1)) >> 1 if step != 1 else 0

            # Check bounds before accessing list element
            code1 = 0 # Default value like unwrap_or(&0)
            if 0 <= pos1 < n:
                code1 = self.positions[pos1]
            # else: pos1 is out of bounds, code1 remains 0

            # Compare position_code with code1
            if position_code < code1:
                # Note: No wrapping needed in Python if indices are managed correctly
                pos1 = pos1 - step
            elif position_code > code1:
                pos1 = pos1 + step
            else: # position_code == code1
                # Found the code, get the value (check bounds again just in case)
                if 0 <= pos1 < n:
                    value = self.values[pos1]
                else:
                    # This case should ideally not happen if logic is correct
                    # but mirrors the potential issue if pos1 became invalid
                    # after the assignment but before value retrieval.
                    value = -99 # Indicate not found
                break # Exit loop

        if value != -99:
            return int(value) # Return as standard int
        else:
            return None


# /// A shared, immutable, non-thread-safe opening database
# ///
# /// # Notes
# /// The database stores all 'unique' positions with exactly 12 tiles played and their scores.
# /// In this case 'unique' means a position whose mirror image is not already in the database
# /// and does not have any moves that end the game on the next turn, as the game-tree search
# /// short-circuits in these cases before checking the database.
# ///
# /// Positions are stored using a Huffman code of the board (4 bytes) and 1 byte representing
# /// the signed score, for a total size of ~20MB. The entries are stored in ascending numeric order
# /// of the Huffman code to allow binary search.
# ///
# /// For details of the Huffman code and score, see [`BitBoard`] and [`Solver`].
# ///
# /// The database contains a `Rc` internally, allowing cheap cloning.
# /// In Python, standard object references provide similar cheap copying of references.
# ///
# /// [`BitBoard`]: ../bitboard/struct.BitBoard.html#huffman-codes -> See placeholder BitBoard class
# /// [`Solver`]: ../solver/struct.Solver.html#position-scoring -> See placeholder Solver class
# # Note: @derive(Clone) is achieved by Python's default reference assignment behavior
class OpeningDatabase:
    def __init__(self, storage: OpeningDatabaseStorage):
        # Store the single storage object; assignments will share this reference.
        self._storage = storage

    # /// Try to load a database from the hard-coded file path into memory
    @staticmethod
    def load() -> 'OpeningDatabase':
        # Note: Result is handled by exceptions
        try:
            storage = OpeningDatabaseStorage.load()
            return OpeningDatabase(storage)
        except Exception as e:
            # Propagate the exception or handle it as needed
            print(f"Failed to load OpeningDatabase: {e}")
            raise

    # /// Retrieve the score for a position, given as a huffman code
    # ///
    # /// Returns `None` if the position is not found in the database,
    # /// see [Notes] for details of stored positions
    # ///
    # /// [Notes]: #Notes
    def get(self, position_code: int) -> Optional[int]:
        # Note: position_code is u32 in Rust, standard int in Python
        return self._storage.get(position_code)

    # --- Generation Logic ---

    # Helper function for the generation process running in a separate process
    @staticmethod
    def _generate_positions_task(start_col: int, output_queue: Queue, start_time: float):
        # Note: Corresponds to the thread::spawn closure in Rust
        moves = [0] * DATABASE_DEPTH
        moves[0] = start_col
        positions: List[Tuple[int, int, int]] = [] # List of (huffman_code, player_mask, board_mask)
        generated = 0
        last_size = 0
        # Note: Using time.monotonic() for interval checks is generally better than time.time()
        next_report_time = time.monotonic() + 0.1 # Report quickly at first

        # Loop condition needs careful translation
        # The Rust loop iterates through all move combinations of length DATABASE_DEPTH
        # starting with `start_col`. It increments the last move, carrying over like addition.
        # It stops when the second move (index 1) onwards are all WIDTH - 1,
        # which signifies the end for that starting column branch.

        while True:
            # Check termination condition: moves[1] through moves[HEIGHT+1] are all WIDTH-1
            # The Rust code checks `moves.iter().skip(1).take(HEIGHT + 1)`.
            # This seems slightly off, maybe it should be `take(DATABASE_DEPTH - 1)`?
            # Let's assume it means all moves *after* the first one.
            # If DATABASE_DEPTH is 12, it checks indices 1 through 11.
            # Let's replicate the termination check based on the likely intent:
            # Stop if all moves from index 1 onwards have reached their maximum value (WIDTH - 1).
            if all(m == WIDTH - 1 for m in moves[1:]):
                # Send final results and break
                # Sort and deduplicate one last time before sending
                positions.sort(key=lambda x: x[0])
                unique_positions = []
                if positions:
                    last_code = -1
                    for p in positions:
                        if p[0] != last_code:
                            unique_positions.append(p)
                            last_code = p[0]
                output_queue.put(('finish', (generated, unique_positions)))
                break

            # Try to create a board from the current move sequence
            # Note: The slice passed to from_slice should represent the actual moves played.
            # If moves array represents the sequence, pass it directly.
            board = BitBoard.from_slice(moves[:DATABASE_DEPTH]) # Ensure correct slice length

            if board is not None:
                # don't include next-turn wins, the tree search short-circuits these
                # before searching the database
                is_next_turn_win = False
                for i in move_order():
                    if board.playable(i) and board.check_winning_move(i):
                        is_next_turn_win = True
                        break

                if not is_next_turn_win:
                    # both mirrors will push the same huffman code, we will dedup later
                    # Note: Deduplication happens later in the main thread or here before finishing.
                    # The Rust code deduplicates periodically and finally.
                    positions.append((
                        board.huffman_code(),
                        board.player_mask(),
                        board.board_mask(),
                    ))
                    generated += 1

            # Increment moves sequence (like base-WIDTH addition)
            moves[DATABASE_DEPTH - 1] += 1
            # carry the addition
            for d in range(DATABASE_DEPTH - 1, 0, -1): # Iterate from depth-1 down to 1
                if moves[d] >= WIDTH:
                    moves[d] = 0
                    moves[d - 1] += 1
                else:
                    # No carry needed for lower indices if current digit is valid
                    break # Optimization: stop carrying over

            # Check if the first move needs incrementing (shouldn't happen if loop logic is correct)
            # The loop should break before moves[0] needs to change.

            # Periodic reporting and deduplication
            current_time = time.monotonic()
            if current_time > next_report_time:
                # Optional: Periodic deduplication like in Rust
                if len(positions) - last_size > 10_000_000: # Arbitrary threshold
                     positions.sort(key=lambda x: x[0])
                     unique_positions_temp = []
                     if positions:
                         last_code_temp = -1
                         for p in positions:
                             if p[0] != last_code_temp:
                                 unique_positions_temp.append(p)
                                 last_code_temp = p[0]
                     positions = unique_positions_temp
                     last_size = len(positions)

                if generated > 0:
                    output_queue.put(('count', generated))
                    generated = 0 # Reset count after reporting
                next_report_time = current_time + 0.5 # Report every 500ms


    # Helper function for the score calculation task
    @staticmethod
    def _calculate_score_task(position_data: Tuple[int, int, int]) -> Tuple[int, int]:
        # Note: Corresponds to the closure in par_iter().for_each_with()
        huffman_code, player_mask, board_mask = position_data
        # Recreate the board
        # The number of moves is fixed at DATABASE_DEPTH for these entries
        board = BitBoard.from_parts(player_mask, board_mask, DATABASE_DEPTH)

        # Create a solver and solve
        solver = Solver(board)
        score, _ = solver.solve() # Ignore the second return value (move info?)

        # Ensure score fits i8 range (-128 to 127) as in Rust
        score_i8 = max(-128, min(127, score))

        return (huffman_code, score_i8)


    # /// Generate an opening database at the hard-coded depth and path
    # ///
    # /// # Warning
    # /// This procedure is very computationally intensive; tested on a
    # /// Ryzen 5 1600 @ 3.2GHz generation took 23 hours at 100% CPU usage on all cores
    @staticmethod
    def generate() -> None:
        # Note: Result<()> becomes None, errors are raised as exceptions
        start_time = time.monotonic()
        print("Starting database generation...")

        all_positions: List[Tuple[int, int, int]] = [] # List of (huffman_code, player_mask, board_mask)

        # try to read positions from temp file
        if os.path.exists(TEMP_FILE_PATH):
            print(f"Loading stored positions from {TEMP_FILE_PATH}")
            try:
                with open(TEMP_FILE_PATH, 'rb') as positions_file:
                    # Read triples of (u32, u64, u64)
                    entry_format = '>IQQ' # Big-endian u32, u64, u64
                    entry_size = struct.calcsize(entry_format)
                    while True:
                        data = positions_file.read(entry_size)
                        if not data:
                            break
                        if len(data) < entry_size:
                            raise IOError("Incomplete data read from temp file.")
                        all_positions.append(struct.unpack(entry_format, data))
                print(f"Loaded {len(all_positions)} positions.")
                # Optional: Verify against DATABASE_NUM_POSITIONS if needed
                # if len(all_positions) != DATABASE_NUM_POSITIONS:
                #    print(f"Warning: Loaded count {len(all_positions)} differs from expected {DATABASE_NUM_POSITIONS}")

            except Exception as e:
                print(f"Error loading temp file: {e}. Regenerating positions.")
                all_positions = [] # Clear potentially corrupt data
                # Optionally remove the corrupt temp file
                # try: os.remove(TEMP_FILE_PATH) except OSError: pass
        else:
            print("No temp file found. Generating positions...")
            # Phase 1: Generate positions using multiprocessing
            # enum Message -> Use tuples like ('type', data)
            # let (tx, rx) = channel(); -> Use multiprocessing.Queue
            output_queue = Queue()
            processes: List[Process] = []

            for i in range(WIDTH):
                # Create and start a process for each starting column
                p = Process(target=OpeningDatabase._generate_positions_task, args=(i, output_queue, start_time))
                processes.append(p)
                p.start()

            # Progress bar setup
            # Note: TOTAL_GENERATION_TARGET is a very large number from Rust, might not be accurate here.
            # Using total=None might be safer if the exact count is unknown. Let's use the constant for now.
            progress = tqdm(total=TOTAL_GENERATION_TARGET, desc="[1/2] Generating positions", unit="pos", ncols=80)

            total_generated_count = 0
            finished_processes = 0
            temp_positions_buffer: List[Tuple[int, int, int]] = []

            while finished_processes < WIDTH:
                try:
                    # Wait for messages from worker processes
                    msg_type, data = output_queue.get(timeout=0.1) # Timeout to allow progress update

                    if msg_type == 'count':
                        count = data
                        total_generated_count += count
                        progress.update(count)
                        progress.set_postfix_str(f"({total_generated_count // 1_000_000}M / {TOTAL_GENERATION_TARGET // 1_000_000}M)", refresh=False)
                    elif msg_type == 'finish':
                        thread_generated, thread_positions = data
                        total_generated_count += thread_generated
                        progress.update(thread_generated) # Update progress with final count from thread
                        temp_positions_buffer.extend(thread_positions)
                        finished_processes += 1
                        # Optional: Intermediate sort/dedup of buffer if memory becomes an issue
                        # if len(temp_positions_buffer) > SOME_LARGE_NUMBER: process buffer...

                except QueueEmpty:
                    # Timeout occurred, just continue (allows checking loop condition)
                    pass
                except (EOFError, BrokenPipeError):
                     # Handle cases where a child process might have exited unexpectedly
                     print("Warning: A generation worker process seems to have exited prematurely.")
                     # We might need more robust error handling here, e.g., count missing processes
                     # For now, assume we just wait for the remaining ones.
                     # To prevent hanging, we could check process liveness:
                     active_processes = sum(1 for p in processes if p.is_alive())
                     if active_processes == 0 and finished_processes < WIDTH:
                         print("Error: All generation workers finished but not all reported completion.")
                         # Decide how to proceed: maybe raise error, maybe continue with partial data
                         break # Exit collection loop
                     pass # Continue waiting if some processes are still alive

            # Ensure all processes are joined
            for p in processes:
                p.join()

            progress.close()

            # Final sort and deduplication
            print("Sorting and deduplicating generated positions...")
            temp_positions_buffer.sort(key=lambda x: x[0])
            if temp_positions_buffer:
                 last_code = -1
                 for p in temp_positions_buffer:
                     if p[0] != last_code:
                         all_positions.append(p)
                         last_code = p[0]

            generation_duration = time.monotonic() - start_time
            print(
                f"Position generation complete in {format_duration(generation_duration)}, "
                f"found {len(all_positions)} unique positions"
            )

            # Write positions to temp file
            print(f"Writing out positions to {TEMP_FILE_PATH} ... ", end='', flush=True)
            try:
                # Note: BufWriter equivalent is using buffered IO, default for open()
                with open(TEMP_FILE_PATH, 'wb') as positions_file:
                    entry_format = '>IQQ' # Big-endian u32, u64, u64
                    for position in all_positions:
                        positions_file.write(struct.pack(entry_format, *position))
                print("Complete")
            except IOError as e:
                print(f"Error writing temp file: {e}")
                # Decide if we should continue without the temp file or stop

        # --- Phase 2: Calculate scores ---
        print("Starting score calculation...")
        # enum Message2 -> Use results directly from Pool or a Queue if needed
        # let (tx, rx) = channel(); -> Handled by multiprocessing.Pool

        # Use multiprocessing Pool for parallel execution
        # Note: Using cpu_count() might utilize all cores, adjust if needed
        num_workers = cpu_count()
        print(f"Using {num_workers} worker processes for score calculation.")

        calculated_entries: List[Tuple[int, int]] = [] # List of (huffman_code, score_i8)

        # Setup progress bar for score calculation
        score_progress = tqdm(total=len(all_positions), desc="[2/2] Calculating scores", unit="pos", ncols=80)

        # Define a simple callback for tqdm updates
        def update_progress(*args):
            score_progress.update()

        try:
            with Pool(processes=num_workers) as pool:
                # Use imap_unordered for potentially better performance as results come in
                # Need to wrap the task function if using imap
                results_iterator = pool.imap_unordered(OpeningDatabase._calculate_score_task, all_positions, chunksize=100) # Adjust chunksize as needed

                for result in results_iterator:
                    calculated_entries.append(result)
                    score_progress.update(1)
                    score_progress.set_postfix_str(f"({score_progress.n} / {score_progress.total})", refresh=False) # Update message less frequently if needed

        except Exception as e:
            score_progress.close()
            print(f"\nError during score calculation: {e}")
            raise # Propagate the error

        score_progress.close()

        calculation_duration = time.monotonic() - start_time - generation_duration # Time for this phase
        print(f"Score calculation complete in {format_duration(calculation_duration)}")


        # --- Final Step: Write database file ---
        print(f"Calculations complete, writing out to {DATABASE_PATH} ... ", end='', flush=True)

        # Sort entries by Huffman code
        calculated_entries.sort(key=lambda x: x[0])

        try:
            # Note: BufWriter equivalent is using buffered IO
            with open(DATABASE_PATH, 'wb') as db_file:
                entry_format = '>Ib' # Big-endian u32, signed i8
                for entry in calculated_entries:
                    db_file.write(struct.pack(entry_format, entry[0], entry[1]))
            print("Complete")
        except IOError as e:
            print(f"Error writing database file: {e}")
            raise

        # Optional: Clean up temp file
        try:
            os.remove(TEMP_FILE_PATH)
            print(f"Removed temporary file {TEMP_FILE_PATH}")
        except OSError:
            pass # Ignore if file doesn't exist or cannot be removed

        finish_time = time.monotonic()
        total_duration = finish_time - start_time
        print(f"Opening database generation completed in {format_duration(total_duration)}")

        # Note: Ok(()) is implicit in Python if no exception is raised

# Example Usage (Optional)
if __name__ == '__main__':
    # To generate the database (takes a long time!)
    # try:
    #     print("Attempting to generate the opening database...")
    #     OpeningDatabase.generate()
    #     print("Database generation finished successfully.")
    # except Exception as e:
    #     print(f"Database generation failed: {e}")

    # To load and use the database
    try:
        print(f"Attempting to load the opening database from {DATABASE_PATH}...")
        db = OpeningDatabase.load()
        print("Database loaded successfully.")

        # Example lookup (replace with actual Huffman codes)
        test_code_1 = 12345678 # Example Huffman code
        test_code_2 = 87654321 # Another example
        test_code_not_found = 99999999

        score1 = db.get(test_code_1)
        if score1 is not None:
            print(f"Score for code {test_code_1}: {score1}")
        else:
            print(f"Code {test_code_1} not found in database.")

        score2 = db.get(test_code_2)
        if score2 is not None:
            print(f"Score for code {test_code_2}: {score2}")
        else:
            print(f"Code {test_code_2} not found in database.")

        score_nf = db.get(test_code_not_found)
        if score_nf is not None:
            print(f"Score for code {test_code_not_found}: {score_nf}")
        else:
            print(f"Code {test_code_not_found} not found in database.")

    except FileNotFoundError:
         print(f"Database file {DATABASE_PATH} not found. Generate it first.")
    except Exception as e:
        print(f"An error occurred: {e}")

