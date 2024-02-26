use crate::error::PsetRes;
use serde::{Deserialize, Serialize};
use std::{fs::File, io::BufReader};

const SQUARE_MATRIX_FILES: [&str; 3] = [
    "./src/test_data/3x3.json",
    "./src/test_data/8x8.json",
    "./src/test_data/64x64.json",
];

const ASYMM_MATRIX_FILES: [&str; 1] = ["./src/test_data/3x4.json"];

#[derive(Debug, Serialize, Deserialize)]
pub struct SquareTestMatrix {
    /// n * n matrix
    pub left: Vec<Vec<i64>>,
    /// n * n matrix
    pub right: Vec<Vec<i64>>,
    /// left + right
    pub sum: Vec<Vec<i64>>,
    /// left - right
    pub diff: Vec<Vec<i64>>,
    /// left * right
    pub prod: Vec<Vec<i64>>,
}

/// Returns a list of TestMatrix data stored in json files.
pub fn get_square_test_matrices() -> PsetRes<Vec<SquareTestMatrix>> {
    SQUARE_MATRIX_FILES
        .iter()
        .map(|path| Ok(serde_json::from_reader(BufReader::new(File::open(*path)?))?))
        .collect()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AsymmTestMatrix {
    /// n * n matrix
    pub left: Vec<Vec<i64>>,
    /// n * n matrix
    pub right: Vec<Vec<i64>>,
    /// left * right
    pub prod: Vec<Vec<i64>>,
}

pub fn get_asymm_test_matrices() -> PsetRes<Vec<AsymmTestMatrix>> {
    ASYMM_MATRIX_FILES
        .iter()
        .map(|path| Ok(serde_json::from_reader(BufReader::new(File::open(*path)?))?))
        .collect()
}
