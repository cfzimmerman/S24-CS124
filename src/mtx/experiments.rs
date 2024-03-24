use super::{matrix::Matrix, padding::PadPow2};
use crate::error::{PsetErr, PsetRes};
use csv::Writer;
use rand::{
    distributions::{Uniform, WeightedIndex},
    prelude::Distribution,
    thread_rng, Rng,
};
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
    time_secs: f64,
}

/// Configures and runs experiments on the effect of
/// different base_cutoff values on Strassen multiplication.
#[derive(Serialize, Deserialize)]
pub struct BaseExperiment {
    input_sizes: Vec<usize>,
    val_min: i32,
    val_max: i32,
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
        let mut left = Self::rand_sq(*largest_dim, self.val_min..=self.val_max);
        let mut right = Self::rand_sq(*largest_dim, self.val_min..=self.val_max);

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
                    time_secs: Duration::as_secs_f64(&start.elapsed()),
                })?;
                cutoff /= 2;
            }
        }
        Ok(())
    }

    /// Generates a new square matrix of the given dimension with values
    /// sampled from the given range.
    fn rand_sq(dim: usize, vals: RangeInclusive<i32>) -> Matrix<i32> {
        let mut rng = thread_rng();
        let range = Uniform::from(vals);
        let mut mtx: Vec<Vec<i32>> = Vec::with_capacity(dim);
        for _ in 0..dim {
            mtx.push((&mut rng).sample_iter(range).take(dim).collect());
        }
        mtx.into()
    }
}

/// CSV row containing the results of a triangle experiment
#[derive(Serialize)]
struct TriangleExpRes {
    graph_dim: usize,
    edge_prob: f32,
    trial: usize,
    num_triangles: i32,
}

/// Configures and runs experiments on the number of triangles in a
/// graph with the given dimensions and edge probabilities.
#[derive(Serialize, Deserialize)]
pub struct TriangleExperiment {
    num_vertices: usize,
    num_trials: usize,
    edge_probabilities: Vec<f32>,
}

impl TriangleExperiment {
    /// Builds a triangle experiment from an experiment config file.
    pub fn from_cfg(cfg_file: &Path) -> PsetRes<TriangleExperiment> {
        Ok(serde_json::from_reader(BufReader::new(File::open(
            cfg_file,
        )?))?)
    }

    /// Given the config in self, uses matrix multiplication to compute the
    /// number of triangles in a graph with edge probability P and writes
    /// results in CSV form to the writer.
    pub fn run(&self, csv: &mut Writer<File>) -> PsetRes<()> {
        let dim = self.num_vertices;
        for prob in self.edge_probabilities.iter().copied() {
            for trial in 1..=self.num_trials {
                println!("num_vertices: {dim}, edge_prob: {prob}, trial: {trial}");
                // the clone is inefficient, but I don't have time right now for
                // a more elegant solution. Good place for a future optimization.
                let mut left = Self::edge_graph(dim, prob)?;
                let mut right = left.clone();
                let mut sq = Matrix::mul_strassen(&mut left, &mut right, 64)?;
                drop(left);
                let cube = Matrix::mul_strassen(&mut sq, &mut right, 64)?;
                let mut diag_sum = 0;
                for offset in 0..dim {
                    diag_sum += cube.inner[offset][offset];
                }
                let num_triangles = diag_sum / 6;
                csv.serialize(TriangleExpRes {
                    graph_dim: dim,
                    edge_prob: prob,
                    num_triangles,
                    trial,
                })?;
            }
        }
        Ok(())
    }

    /// Generates a new square matrix representing an adjacency matrix of
    /// a graph where vertices have a `prob_edge` likelihood of having an
    /// edge between them.
    fn edge_graph(dim: usize, prob_edge: f32) -> PsetRes<Matrix<i32>> {
        if !(0. ..=1.).contains(&prob_edge) {
            return Err(PsetErr::Static("probability of an edge must be in [0, 1]"));
        }
        let choices = [0, 1];
        let weights = [1. - prob_edge, prob_edge];
        let range = WeightedIndex::new(weights)?;
        let mut rng = thread_rng();

        let mut mtx: Vec<Vec<i32>> = Vec::with_capacity(dim);
        for _ in 0..dim {
            mtx.push((0..dim).map(|_| choices[range.sample(&mut rng)]).collect())
        }
        Ok(mtx.into())
    }
}

#[cfg(test)]
mod test {
    use crate::{
        error::PsetRes,
        mtx::experiments::{BaseExperiment, TriangleExperiment},
    };

    /// Verifies matrix generation functions yield correct output.
    #[test]
    fn gen_rand_sqs() -> PsetRes<()> {
        let dim = 4;

        let rng_mtx = BaseExperiment::rand_sq(dim, 0..=10);
        assert_eq!(rng_mtx.num_rows(), dim, "Rng matrix should have dim rows");
        assert_eq!(rng_mtx.num_cols(), dim, "Rng matrix should have dim cols");

        let graph_mtx = TriangleExperiment::edge_graph(dim, 0.5)?;
        assert_eq!(
            graph_mtx.num_rows(),
            dim,
            "Graph matrix should have dim rows"
        );
        assert_eq!(
            graph_mtx.num_cols(),
            dim,
            "Graph matrix should have dim cols"
        );

        Ok(())
    }
}
