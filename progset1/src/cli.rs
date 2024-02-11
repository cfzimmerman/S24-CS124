use crate::error::{PsetErr, PsetRes};
use std::{num::ParseIntError, path::PathBuf, str::FromStr, time::Duration};

/// The various graph dimensions this program supports.
#[derive(Debug)]
pub enum GraphDim {
    OneD,
    TwoD,
    ThreeD,
    FourD,
}

/// Structures the input arguments to a TrialAverage command.
#[derive(Debug)]
pub struct MstAverageArgs {
    pub num_vertices: usize,
    pub num_trials: usize,
    pub graph_dimension: GraphDim,
}

/// A configuration object expected in the config file provided
/// by a stats collection file. Inputs should be in Json.
#[derive(Debug)]
pub struct DataCollectionConfig {
    /// Number of vertices
    pub graph_sizes: Vec<usize>,
    pub graph_dimensions: Vec<usize>,
    pub trials_per_size: usize,
}

#[derive(Debug)]
pub struct CollectedStat {
    pub graph_size: usize,
    pub graph_dimension: usize,
    pub trial_number: usize,
    pub runtime: Duration,
    pub weight: f64,
}

#[derive(Debug)]
pub struct CollectStats {
    pub output_filepath: PathBuf,
    pub config_filepath: PathBuf,
    pub config: DataCollectionConfig,
}

/// The command variants this program supports.
#[derive(Debug)]
pub enum CliCommand {
    /// Generates a `GraphDim` dimension graph of size `num_vertices`
    /// and finds the weight of its MST a total of `num_trials`
    /// times, returning the average weight,
    MstAverage(MstAverageArgs),
}

impl CliCommand {
    fn usage_err(issue: &str) -> PsetErr {
        eprintln!(
            r"Supported commands:
        // Returns the average MST weight given inputs:
        > [cargo run --release] 0 [usize numpoints] [usize numtrials] [usize dimension]

        // Runs the commmands specified in the config filepath and writes results as a 
        // CSV back to the output filepath:
        > [cargo run --release] 1 [config filepath.json] [output filepath.csv] 
        "
        );

        PsetErr::Cxt(format!("Parse error: {}", issue))
    }

    fn get_parsed<T: FromStr<Err = ParseIntError>>(
        args: &[String],
        ind: usize,
        issue: &str,
    ) -> PsetRes<T> {
        let val = match args.get(ind) {
            Some(val) => val,
            None => return Err(CliCommand::usage_err(issue)),
        };
        let parsed = val.parse::<T>()?;
        Ok(parsed)
    }
}

impl TryFrom<&[String]> for CliCommand {
    type Error = PsetErr;

    fn try_from(value: &[String]) -> Result<Self, Self::Error> {
        let mode = match value.get(1) {
            Some(num) => num.parse::<usize>()?,
            None => return Err(CliCommand::usage_err("missing mode")),
        };
        match mode {
            0 => {
                let num_vertices: usize =
                    CliCommand::get_parsed(value, 2, "expected usize num_vertices")?;
                let num_trials: usize =
                    CliCommand::get_parsed(value, 3, "expected usize num_trials")?;
                let dimension: usize =
                    CliCommand::get_parsed(value, 4, "expected usize dimension")?;
                Ok(Self::MstAverage(MstAverageArgs {
                    num_vertices,
                    num_trials,
                    graph_dimension: dimension.try_into()?,
                }))
            }
            /*
            1 => {
                let config_filepath = value.get(2).ok_or_else(|| PsetErr::Static("expected config filepath"))?;
                let output_filepath = value.get(3).ok_or_else(|| PsetErr::Static("expected output filepath"))?;
            },
            */
            _ => Err(CliCommand::usage_err("unsupported mode num")),
        }
    }
}

impl TryFrom<usize> for GraphDim {
    type Error = PsetErr;
    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::OneD),
            2 => Ok(Self::TwoD),
            3 => Ok(Self::ThreeD),
            4 => Ok(Self::FourD),
            _ => Err(PsetErr::Cxt(format!(
                "{} does not correspond to a supported graph dimension: 1, 2, 3, 4.",
                value
            ))),
        }
    }
}
