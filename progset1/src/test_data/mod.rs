use crate::{error::PsetRes, prim_heap::Weight};
use serde::Deserialize;
use std::fs;

const ISIZE_ARR: &str = "./src/test_data/rng_isize_1k.json";
const WEIGHTED_ARR: &str = "./src/test_data/rng_weighted.json";

/// Returns a vector of static isize values that were previously
/// randomly generated.
pub fn get_isize_arr() -> PsetRes<Vec<isize>> {
    let file = fs::read_to_string(ISIZE_ARR)?;
    let nums: Vec<isize> = serde_json::from_str(file.as_str())?;
    Ok(nums)
}

#[derive(Deserialize)]
struct WeightedVal {
    weight: f64,
    val: i64,
}

pub fn get_weighted_nums() -> PsetRes<Vec<(i64, Weight<f64>)>> {
    let file = fs::read_to_string(WEIGHTED_ARR)?;
    let nums: Vec<WeightedVal> = serde_json::from_str(file.as_str())?;
    Ok(nums
        .into_iter()
        .map(|wv| (wv.val, Weight::new(wv.weight)))
        .collect())
}
