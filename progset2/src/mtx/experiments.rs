use super::{matrix::Matrix, padding::PadPow2};
use crate::error::PsetRes;
use csv::Writer;
use rand::{distributions::Uniform, thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::BufReader,
    ops::RangeInclusive,
    path::Path,
    time::{Duration, Instant},
};

/// The minimum cutoff a strassen multiplication will ever have before
/// switching to iterative multiplication.
const MIN_STRASSEN_CUTOFF: usize = 4;

/// CSV row containing the results of a base cutoff experiment
#[derive(Serialize)]
struct BaseExpRes {
    graph_dim: usize,
    base_cutoff: usize,
    time_ms: f64,
}

/// Configures and runs experiments on the affect of
/// different base_cutoff values on Strassen multiplication.
#[derive(Serialize, Deserialize)]
pub struct BaseExperiment {
    input_sizes: Vec<usize>,
    val_min: i32,
    val_max: i32,
}

/// Generates a new square matrix of the given dimension with values
/// sampled from the given range.
fn new_rand_sq(dim: usize, vals: RangeInclusive<i32>) -> Matrix<i32> {
    let mut rng = thread_rng();
    let range = Uniform::from(vals);
    let mut mtx: Vec<Vec<i32>> = Vec::with_capacity(dim);
    for _ in 0..dim {
        mtx.push((&mut rng).sample_iter(range).take(dim).collect());
    }
    mtx.into()
}

impl BaseExperiment {
    /// Builds a base-case experiment from an experiment config file.
    pub fn from_cfg(cfg_file: &Path) -> PsetRes<BaseExperiment> {
        Ok(serde_json::from_reader(BufReader::new(File::open(
            cfg_file,
        )?))?)
    }

    /// Runs the experiment configured by self, writing results to
    /// the given csv file.
    pub fn run(&mut self, csv: &mut Writer<File>) -> PsetRes<()> {
        self.input_sizes.sort();
        let Some(largest_dim) = self.input_sizes.last() else {
            eprintln!("Skipping experiments on empty input");
            return Ok(());
        };
        let mut left = new_rand_sq(*largest_dim, self.val_min..=self.val_max);
        let mut right = new_rand_sq(*largest_dim, self.val_min..=self.val_max);

        for dim in self.input_sizes.iter().rev().copied() {
            PadPow2::trim_dims(&mut left, dim, dim);
            PadPow2::trim_dims(&mut right, dim, dim);
            let mut cutoff = dim;
            while MIN_STRASSEN_CUTOFF <= cutoff {
                println!("dim: {dim}, cutoff: {cutoff}");
                let start = Instant::now();
                Matrix::mul_strassen(&mut left, &mut right, cutoff)?;
                csv.serialize(BaseExpRes {
                    graph_dim: dim,
                    base_cutoff: cutoff,
                    time_ms: Duration::as_secs_f64(&start.elapsed()),
                })?;
                cutoff /= 2;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::mtx::experiments::new_rand_sq;

    /// Verifies rng matrix generation yields correct output.
    #[test]
    fn gen_rand_sq() {
        let dim = 4;
        let mtx = new_rand_sq(dim, 0..=10);
        assert_eq!(mtx.num_rows(), dim, "Rng matrix should have dim rows");
        assert_eq!(mtx.num_cols(), dim, "Rng matrix should have dim cols");
    }
}
