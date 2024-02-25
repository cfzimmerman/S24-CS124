use std::{
    fs::{self, File},
    io::BufReader,
};

use crate::error::PsetRes;
use serde::{Deserialize, Serialize};

const MATRIX_FILES: [&str; 1] = ["./src/test_data/3x3.json"];

#[derive(Debug, Serialize, Deserialize)]
pub struct TestMatrix {
    /// n * m matrix
    pub left: Vec<Vec<i64>>,
    /// m * k matrix
    pub right: Vec<Vec<i64>>,
    /// left + right
    pub sum: Vec<Vec<i64>>,
    /// left - right
    pub diff: Vec<Vec<i64>>,
    /// left * right
    pub prod: Vec<Vec<i64>>,
}

/// Returns a list of TestMatrix data stored in json files.
pub fn get_test_matrices() -> PsetRes<Vec<TestMatrix>> {
    MATRIX_FILES
        .iter()
        .map(|path| Ok(serde_json::from_reader(BufReader::new(File::open(*path)?))?))
        .collect()
}
