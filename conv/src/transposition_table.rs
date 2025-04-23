//! A transposition table to cache the results of Connect 4 game tree searches.

use std::sync::{atomic::*, Arc, Mutex};

#[derive(Copy, Clone)]
struct Entry {
    key: u32,
    value: u8,
}

impl Entry {
    pub fn new() -> Self {
        Self { key: 0, value: 0 }
    }
}

/// The capacity of the transposition table in entries. Prime values minimise hash collisions
pub const TABLE_MAX_SIZE: usize = (1 << 23) + 9; // prime value minimises hash collisions

struct TranspositionTableStorage {
    entries: Vec<Entry>,
}

impl TranspositionTableStorage {
    pub fn new() -> Self {
        Self {
            entries: vec![Entry::new(); TABLE_MAX_SIZE],
        }
    }

    pub fn set(&mut self, key: u64, value: u8) {
        let mut entry = Entry::new();
        entry.key = key as u32;
        entry.value = value;

        let len = self.entries.len();
        self.entries[key as usize % len] = entry;
    }

    pub fn get(&self, key: u64) -> u8 {
        let entry = self.entries[key as usize % self.entries.len()];
        if entry.key == key as u32 {
            entry.value
        } else {
            0
        }
    }
}

/// A thread-safe transposition table
#[derive(Clone)]
pub struct TranspositionTable(Arc<Mutex<TranspositionTableStorage>>);

impl TranspositionTable {
    /// Creates an empty transposition table
    pub fn new() -> Self {
        Self(Arc::new(Mutex::new(TranspositionTableStorage::new())))
    }

    /// Set a key-value pair in the transposition table
    pub fn set(&self, key: u64, value: u8) {
        self.0.lock().unwrap().set(key, value);
    }

    /// Retrieve a value from the transposition table
    pub fn get(&self, key: u64) -> u8 {
        self.0.lock().unwrap().get(key)
    }
    pub fn with_capacity(size_mb: usize) -> Self {
        let num_entries = (size_mb * 1024 * 1024) / std::mem::size_of::<Entry>();
        let storage = TranspositionTableStorage {
            entries: vec![Entry::new(); num_entries],
        };
        Self(Arc::new(Mutex::new(storage)))
    }
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new()
    }
}

struct SharedEntry {
    key: AtomicU32,
    value: AtomicU8,
}

impl SharedEntry {
    pub fn new() -> Self {
        Self {
            key: AtomicU32::new(0),
            value: AtomicU8::new(0),
        }
    }

    pub fn store(&self, key: u32, value: u8) {
        self.key.store(key, Ordering::Relaxed);
        self.value.store(value, Ordering::Relaxed);
    }
}

#[doc(hidden)]
#[derive(Clone)]
pub struct SharedTranspositionTable {
    entries: Arc<Vec<SharedEntry>>,
}

impl SharedTranspositionTable {
    pub fn new() -> Self {
        let mut entries = Vec::with_capacity(TABLE_MAX_SIZE);
        for _ in 0..TABLE_MAX_SIZE {
            entries.push(SharedEntry::new());
        }
        Self {
            entries: Arc::new(entries),
        }
    }

    pub fn set(&self, key: u64, value: u8) {
        let i = key as usize % self.entries.len();
        self.entries[i].store(key as u32 ^ value as u32, value);
    }

    pub fn get(&self, key: u64) -> u8 {
        let entry = &self.entries[key as usize % self.entries.len()];
        let data = entry.value.load(Ordering::Relaxed);
        if entry.key.load(Ordering::Relaxed) == key as u32 ^ data as u32 {
            data
        } else {
            0
        }
    }
}

impl Default for SharedTranspositionTable {
    fn default() -> Self {
        Self::new()
    }
}