use crate::error::PsetRes;
use std::fs;

const ISIZE_ARR: &str = "./src/test_data/rng_isize_1k.json";

/// Returns a vector of static isize values that were previously
/// randomly generated.
pub fn get_isize_arr() -> PsetRes<Vec<isize>> {
    let file = fs::read_to_string(ISIZE_ARR)?;
    let nums: Vec<isize> = serde_json::from_str(file.as_str())?;
    Ok(nums)
}
