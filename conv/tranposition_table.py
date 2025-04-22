# Required dependencies:
# No external libraries are strictly required beyond standard Python.
# The `threading` module is used for the shared table implementation.

import threading
import math # Used for calculating TABLE_MAX_SIZE if needed, but value is hardcoded

# A transposition table to cache the results of Connect 4 game tree searches.

# Note: Python doesn't have direct equivalents of Rust's u32, u64, u8 types.
# Standard Python integers (`int`) are used. We will manually handle
# potential overflows or specific bit-size requirements where necessary,
# particularly the 32-bit key truncation.
# Python integers have arbitrary precision, so overflow isn't an issue,
# but we need `& 0xFFFFFFFF` to simulate u32 truncation.
# Values intended as u8 should be kept within 0-255.

class Entry:
    """
    Equivalent to the Rust struct Entry.
    Using default values that match Rust's default struct initialization.
    """
    def __init__(self):
        self.key: int = 0 # Corresponds to u32 in Rust
        self.value: int = 0 # Corresponds to u8 in Rust

    # The new() method is effectively the __init__ in Python.
    # Adding a class method for stylistic similarity if desired, but __init__ is standard.
    @classmethod
    def new(cls) -> 'Entry':
        """Creates a new Entry with default values."""
        return cls()

# The capacity of the transposition table in entries. Prime values minimise hash collisions
# Using integer literal directly. Python handles large integers.
TABLE_MAX_SIZE: int = (1 << 23) + 9 # prime value minimises hash collisions
# TABLE_MAX_SIZE: int = (1 << 24) + 13; # prime value minimises hash collisions

class TranspositionTableStorage:
    """
    Equivalent to the Rust struct TranspositionTableStorage.
    Manages the underlying storage for the single-threaded transposition table.
    """
    def __init__(self):
        # Initialize the list with default Entry objects
        self.entries: list[Entry] = [Entry.new() for _ in range(TABLE_MAX_SIZE)]

    # The new() method is effectively the __init__ in Python.
    @classmethod
    def new(cls) -> 'TranspositionTableStorage':
        """Creates a new TranspositionTableStorage."""
        return cls()

    def set(self, key: int, value: int):
        """
        Sets a key-value pair in the storage.
        key: Corresponds to u64 in Rust.
        value: Corresponds to u8 in Rust.
        """
        # Truncate the key to 32 bits, simulating Rust's `key as u32`
        key_u32: int = key & 0xFFFFFFFF
        # Ensure value fits within u8 range (0-255) - assuming valid input as in Rust
        value_u8: int = value & 0xFF

        # Create a new Entry object
        entry = Entry.new()
        entry.key = key_u32
        entry.value = value_u8

        # Calculate the index using modulo operator
        # Python's % operator handles negative numbers differently than Rust's,
        # but keys (hashes) are typically non-negative, so it should be equivalent.
        index: int = key % len(self.entries)
        self.entries[index] = entry

    def get(self, key: int) -> int:
        """
        Retrieves a value from the storage based on the key.
        key: Corresponds to u64 in Rust.
        Returns the value (u8) if the key matches, otherwise 0.
        """
        # Calculate the index
        index: int = key % len(self.entries)
        entry: Entry = self.entries[index]

        # Truncate the lookup key to 32 bits for comparison
        key_u32: int = key & 0xFFFFFFFF

        # Check if the stored key (already truncated) matches the truncated lookup key
        if entry.key == key_u32:
            return entry.value
        else:
            # Return 0 if the key doesn't match (cache miss or collision overwrite)
            return 0

# A shared, non-thread-safe transposition table
#
# # Notes
#
# This table uses standard Python object references internally to allow cheap cloning
# and sharing between `Solver` instances on a single thread. In Rust, this used `Rc<RefCell<...>>`.
# Python's reference counting and garbage collection provide similar shared ownership.
# Direct mutation is allowed unless the object is made immutable.
#
# **The table has a fixed capacity (TABLE_MAX_SIZE) and key collisions will overwrite the previous
# value**
#
# See `BitBoard` for a description of the key values and `Solver` for a description of the values
# (Assuming these are concepts from the larger project context)
#
# [`BitBoard`]: ../bitboard/struct.BitBoard.html#board-keys (External link preserved)
# [`Solver`]: ../solver/struct.Solver.html#position-scoring (External link preserved)
class TranspositionTable:
    """
    A wrapper providing access to the TranspositionTableStorage.
    Mimics the Rust version's use of Rc<RefCell> for shared, mutable access
    in a single-threaded context through standard Python object references.
    """
    # The storage is instantiated directly. Python references handle the sharing.
    def __init__(self, storage: TranspositionTableStorage = None):
        # If no storage is provided, create a new one.
        # This allows sharing storage if needed, similar to Rc.
        if storage is None:
            self._storage = TranspositionTableStorage.new()
        else:
            self._storage = storage

    # Creates an empty transposition table
    @classmethod
    def new(cls) -> 'TranspositionTable':
        """Creates a new TranspositionTable with its own storage."""
        return cls()

    # Set a key-value pair in the transposition table
    def set(self, key: int, value: int):
        """Sets a key-value pair in the underlying storage."""
        # No need for borrow_mut(), just call the method directly.
        self._storage.set(key, value)

    # Retrieve a value from the transposition table
    def get(self, key: int) -> int:
        """Retrieves a value from the underlying storage."""
        # No need for borrow(), just call the method directly.
        return self._storage.get(key)

    # In Python, standard assignment creates a new reference to the same object,
    # effectively "cloning" the reference like Rc::clone.
    # An explicit clone method can be added for clarity if desired.
    def clone(self) -> 'TranspositionTable':
        """Creates another TranspositionTable sharing the same underlying storage."""
        # Return a new instance sharing the same _storage object
        return TranspositionTable(self._storage)

    # Default implementation: __init__ serves as the default constructor.
    # The Default trait in Rust is often mapped to a parameterless __init__.


# --- Thread-Safe Version ---

class SharedEntry:
    """
    Equivalent to the Rust struct SharedEntry, but using threading.Lock for safety
    instead of atomics.
    """
    def __init__(self):
        # Initialize key and value
        self._key: int = 0 # Corresponds to AtomicU32
        self._value: int = 0 # Corresponds to AtomicU8
        # Each entry gets its own lock for fine-grained locking
        self._lock = threading.Lock()

    # The new() method is effectively the __init__ in Python.
    @classmethod
    def new(cls) -> 'SharedEntry':
        """Creates a new SharedEntry with default values and a lock."""
        return cls()

    def store(self, key: int, value: int):
        """
        Atomically stores the key and value.
        key: The value to store in the key field (u32).
        value: The value to store in the value field (u8).
        Uses the lock to ensure atomicity of the update.
        """
        # Ensure key is treated as u32 and value as u8
        key_u32: int = key & 0xFFFFFFFF
        value_u8: int = value & 0xFF
        with self._lock:
            self._key = key_u32
            self._value = value_u8

    def load(self) -> tuple[int, int]:
        """
        Atomically loads the key and value.
        Returns a tuple (key, value).
        Uses the lock to ensure atomicity of the read.
        """
        with self._lock:
            return self._key, self._value

    def load_value(self) -> int:
        """Atomically loads only the value."""
        with self._lock:
            return self._value

    def load_key(self) -> int:
        """Atomically loads only the key."""
        with self._lock:
            return self._key


# Hidden documentation equivalent - often denoted by a leading underscore in Python
# or simply omitted from explicit documentation generation.
# #[doc(hidden)]
class SharedTranspositionTable:
    """
    A thread-safe transposition table using locks for synchronization.
    Equivalent to Rust's SharedTranspositionTable which used Arc<Vec<SharedEntry>> and atomics.
    Python's list references and per-entry locks provide thread-safety.
    """
    def __init__(self, entries: list[SharedEntry] = None):
        if entries is None:
            # Create a list of SharedEntry objects
            self._entries: list[SharedEntry] = [SharedEntry.new() for _ in range(TABLE_MAX_SIZE)]
        else:
            # Allow sharing the list of entries if provided externally
            self._entries = entries
        # Note: No direct Arc equivalent needed, Python's list reference works for sharing.

    # The new() method is effectively the __init__ in Python.
    @classmethod
    def new(cls) -> 'SharedTranspositionTable':
        """Creates a new SharedTranspositionTable."""
        return cls()

    def set(self, key: int, value: int):
        """
        Sets a key-value pair in the shared table using a lock.
        key: Corresponds to u64 in Rust.
        value: Corresponds to u8 in Rust.
        Implements the key ^ value trick for consistency check during get.
        """
        index: int = key % len(self._entries)
        entry: SharedEntry = self._entries[index]

        # Truncate key to 32 bits and value to 8 bits
        key_u32: int = key & 0xFFFFFFFF
        value_u8: int = value & 0xFF

        # Store key' = (key ^ value) and value' = value
        # This is done atomically using the entry's lock via the store method.
        entry.store(key_u32 ^ value_u8, value_u8)

    def get(self, key: int) -> int:
        """
        Retrieves a value from the shared table using a lock.
        key: Corresponds to u64 in Rust.
        Returns the value (u8) if the key check passes, otherwise 0.
        Uses the key ^ value trick for consistency check.
        """
        index: int = key % len(self._entries)
        entry: SharedEntry = self._entries[index]

        # Truncate the lookup key to 32 bits
        key_u32: int = key & 0xFFFFFFFF

        # Atomically load the stored key' and value'
        # We need both values, loaded atomically relative to each other.
        # The load method in SharedEntry handles locking.
        stored_key_xor_value, data = entry.load()

        # Check if stored_key' == (key ^ value')
        # stored_key_xor_value is key'
        # data is value'
        if stored_key_xor_value == (key_u32 ^ data):
            return data
        else:
            # Key mismatch or data corruption (e.g., read during concurrent write
            # without proper atomicity, though the lock prevents this here).
            return 0

    # In Python, standard assignment creates a new reference to the same object,
    # effectively "cloning" the reference like Arc::clone.
    def clone(self) -> 'SharedTranspositionTable':
        """Creates another SharedTranspositionTable sharing the same underlying storage."""
        # Return a new instance sharing the same _entries list object
        return SharedTranspositionTable(self._entries)

    # Default implementation: __init__ serves as the default constructor.

