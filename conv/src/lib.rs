//! A perfect agent for playing or analysing the board game 'Connect 4'

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyModule;
use pyo3::PyResult;
use std::sync::{Arc, Mutex};
use static_assertions::*;
pub use anyhow;

pub mod transposition_table;
pub mod bitboard;
pub mod opening_database;
pub mod solver;

/// The width of the game board in tiles
pub const WIDTH: usize = 7;

/// The height of the game board in tiles
pub const HEIGHT: usize = 6;

// ensure that the given dimensions fit in a u64 for the bitboard representation
const_assert!(WIDTH * (HEIGHT + 1) < 64);

/// Python interface for the Connect4 solver
#[pymodule]
fn connect4_ai(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySolver>()?;
    m.add_function(wrap_pyfunction!(solve_position, m)?)?;
    Ok(())
}

/// Python wrapper for the Solver
#[pyclass]
struct PySolver {
    inner: Arc<Mutex<solver::Solver>>,
}

#[pymethods]
impl PySolver {
    #[new]
    fn new(moves: &str) -> PyResult<Self> {
        let board = bitboard::BitBoard::from_moves(moves)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PySolver {
            inner: Arc::new(Mutex::new(solver::Solver::new(board))),
        })
    }

    /// Solve the current position
    fn solve(&mut self) -> PyResult<(i32, usize)> {
        let mut solver = self.inner.lock().unwrap();
        Ok(solver.solve())
    }
}

/// Solve a position directly from a move string
#[pyfunction]
fn solve_position(moves: &str) -> PyResult<(i32, usize)> {
    let board = bitboard::BitBoard::from_moves(moves)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut solver = solver::Solver::new(board);
    Ok(solver.solve())
}