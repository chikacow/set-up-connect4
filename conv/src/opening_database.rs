//! A searchable store of Connect 4 positions to speed up early-game searches

use anyhow::Result;
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
// use std::{
//     cmp::Ordering,
//     fs::{File, OpenOptions},
//     io::{BufReader, BufWriter, Read, Write},
//     sync::{Arc, Mutex},
//     time::{Duration, Instant},
// };
use std::{
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter},
    sync::{Arc, Mutex},
    time::Instant,
};
use crate::{bitboard::*, solver::*, HEIGHT, WIDTH};

/// Hard-coded database path
pub const DATABASE_PATH: &str = "opening_database.bin";
/// Hard-coded temp file path
pub const TEMP_FILE_PATH: &str = "temp_positions.bin";
/// Hard-coded database depth
pub const DATABASE_DEPTH: usize = 12;
/// Hard-coded database size
pub const DATABASE_NUM_POSITIONS: usize = 4200899;

/// A thread-safe opening database
#[derive(Clone)]
pub struct OpeningDatabase(Arc<OpeningDatabaseStorage>);

impl OpeningDatabase {
    /// Try to load a database from the hard-coded file path into memory
    pub fn load() -> Result<Self> {
        Ok(Self(Arc::new(OpeningDatabaseStorage::load()?)))
    }

    /// Retrieve the score for a position, given as a huffman code
    pub fn get(&self, position_code: u32) -> Option<i32> {
        self.0.get(position_code)
    }

    /// Generate an opening database
    pub fn generate() -> Result<()> {
        let start = Instant::now();
        let positions = Arc::new(Mutex::new(Vec::new()));

        // Load or generate positions
        if std::path::Path::new(TEMP_FILE_PATH).exists() {
            println!("Loading stored positions from {}", TEMP_FILE_PATH);
            let mut positions_file = BufReader::new(File::open(TEMP_FILE_PATH)?);
            let mut temp_positions = Vec::new();
            for _ in 0..DATABASE_NUM_POSITIONS {
                temp_positions.push((
                    positions_file.read_u32::<BigEndian>()?,
                    positions_file.read_u64::<BigEndian>()?,
                    positions_file.read_u64::<BigEndian>()?,
                ));
            }
            *positions.lock().unwrap() = temp_positions;
        } else {
            Self::generate_positions(Arc::clone(&positions))?;
            Self::save_temp_positions(&positions.lock().unwrap())?;
        }

        // Calculate scores
        let entries = Arc::new(Mutex::new(Vec::new()));
        Self::calculate_scores(Arc::clone(&positions), Arc::clone(&entries))?;

        // Save final database
        Self::save_database(&entries.lock().unwrap())?;

        println!(
            "Opening database generation completed in {:?}",
            start.elapsed()
        );
        Ok(())
    }

    fn generate_positions(positions: Arc<Mutex<Vec<(u32, u64, u64)>>>) -> Result<()> {
        let progress = ProgressBar::new(8532690438);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("[1/2] Generating positions: {bar:40.cyan/blue} {msg} ~{eta} remaining")?
                .progress_chars("█▓▒░  "),
        );

        let positions_clone = Arc::clone(&positions);
        rayon::scope(|s| {
            for i in 0..WIDTH {
                let progress = progress.clone();
                let positions = Arc::clone(&positions_clone);
                s.spawn(move |_| {
                    let mut moves = [0; DATABASE_DEPTH];
                    moves[0] = i;
                    let mut thread_positions = Vec::new();
                    let mut generated = 0;

                    loop {
                        if moves.iter().skip(1).take(HEIGHT + 1).all(|&x| x == WIDTH - 1) {
                            break;
                        }

                        if let Ok(board) = BitBoard::from_slice(&moves) {
                            if !move_order()
                                .iter()
                                .any(|&i| board.playable(i) && board.check_winning_move(i))
                            {
                                thread_positions.push((
                                    board.huffman_code(),
                                    board.player_mask(),
                                    board.board_mask(),
                                ));
                                generated += 1;
                            }
                        }

                        // Increment move sequence
                        moves[DATABASE_DEPTH - 1] += 1;
                        for d in (0..DATABASE_DEPTH).rev() {
                            if moves[d] >= WIDTH {
                                moves[d] = 0;
                                if d > 0 {
                                    moves[d - 1] += 1;
                                }
                            }
                        }

                        if generated % 10_000 == 0 {
                            progress.inc(10_000);
                        }
                    }

                    let mut positions = positions.lock().unwrap();
                    positions.extend(thread_positions);
                    progress.inc(generated as u64);
                });
            }
        });

        progress.finish();
        let mut positions = positions.lock().unwrap();
        positions.sort_unstable();
        positions.dedup_by(|a, b| a.0 == b.0);
        Ok(())
    }

    fn save_temp_positions(positions: &[(u32, u64, u64)]) -> Result<()> {
        let mut positions_file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(TEMP_FILE_PATH)?;

        for position in positions {
            positions_file.write_u32::<BigEndian>(position.0)?;
            positions_file.write_u64::<BigEndian>(position.1)?;
            positions_file.write_u64::<BigEndian>(position.2)?;
        }
        Ok(())
    }

    fn calculate_scores(
        positions: Arc<Mutex<Vec<(u32, u64, u64)>>>,
        entries: Arc<Mutex<Vec<(u32, i8)>>>,
    ) -> Result<()> {
        let progress = ProgressBar::new(positions.lock().unwrap().len() as u64);
        progress.set_style(
            ProgressStyle::default_bar()
                .template("[2/2] Calculating scores: {bar:40.cyan/blue} {msg} ~{eta} remaining")?
                .progress_chars("█▓▒░  "),
        );

        let positions = positions.lock().unwrap();
        positions.par_iter().for_each(|(huffman_code, player_mask, board_mask)| {
            let board = BitBoard::from_parts(*player_mask, *board_mask, DATABASE_DEPTH);
            let mut solver = Solver::new(board);
            let (score, _) = solver.solve();
            
            let mut entries = entries.lock().unwrap();
            entries.push((*huffman_code, score as i8));
            progress.inc(1);
        });

        progress.finish();
        Ok(())
    }

    fn save_database(entries: &[(u32, i8)]) -> Result<()> {
        let mut file = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .create(true)
                .open(DATABASE_PATH)?,
        );

        for entry in entries {
            file.write_u32::<BigEndian>(entry.0)?;
            file.write_i8(entry.1)?;
        }
        Ok(())
    }
}

struct OpeningDatabaseStorage {
    positions: Vec<u32>,
    values: Vec<i8>,
}

impl OpeningDatabaseStorage {
    pub fn load() -> Result<Self> {
        let mut file = BufReader::new(File::open(DATABASE_PATH)?);
        let mut positions = Vec::with_capacity(DATABASE_NUM_POSITIONS);
        let mut values = Vec::with_capacity(DATABASE_NUM_POSITIONS);

        for _ in 0..DATABASE_NUM_POSITIONS {
            positions.push(file.read_u32::<BigEndian>()?);
            values.push(file.read_i8()?);
        }

        Ok(Self { positions, values })
    }

    pub fn get(&self, position_code: u32) -> Option<i32> {
        match self.positions.binary_search(&position_code) {
            Ok(pos) => Some(self.values[pos] as i32),
            Err(_) => None,
        }
    }
}