use crate::error::{PsetErr, PsetRes};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    num::ParseIntError,
    path::{Path, PathBuf},
    str::FromStr,
};

/// The various graph dimensions this program supports.
#[derive(Debug)]
pub enum GraphDim {
    ZeroD,
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
#[derive(Debug, Deserialize)]
pub struct DataCollectionConfig {
    /// Number of vertices
    pub graph_sizes: Vec<usize>,
    pub graph_dimensions: Vec<usize>,
    pub trials_per_size: usize,
}

#[derive(Debug, Serialize)]
pub struct CollectedStat {
    pub graph_size: usize,
    pub graph_dimension: usize,
    pub num_trials: usize,
    pub runtime_secs: u64,
    pub weight: f64,
}

#[derive(Debug)]
pub struct CollectStatsArgs {
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

    /// Runs experiments configured by an input file, writing outputs to the specified CSV.
    CollectStats(CollectStatsArgs),
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
                    graph_dimension: (&dimension).try_into()?,
                }))
            }
            1 => {
                let config_filepath = value
                    .get(2)
                    .ok_or_else(|| PsetErr::Static("expected config filepath"))?;
                let config_filepath = PathBuf::from_str(config_filepath.as_str())?;
                let output_filepath = value
                    .get(3)
                    .ok_or_else(|| PsetErr::Static("expected output filepath"))?;
                let output_filepath = PathBuf::from_str(output_filepath.as_str())?;
                let config = DataCollectionConfig::try_from_path(config_filepath.as_path())?;
                Ok(Self::CollectStats(CollectStatsArgs {
                    output_filepath,
                    config_filepath,
                    config,
                }))
            }
            _ => Err(CliCommand::usage_err("unsupported mode num")),
        }
    }
}

impl DataCollectionConfig {
    /// Attempts to retrieve and parse config from the provided file path.
    fn try_from_path(path: &Path) -> PsetRes<Self> {
        let file = fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&file)?;
        Ok(config)
    }
}

impl TryFrom<&usize> for GraphDim {
    type Error = PsetErr;
    fn try_from(value: &usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::ZeroD),
            2 => Ok(Self::TwoD),
            3 => Ok(Self::ThreeD),
            4 => Ok(Self::FourD),
            _ => Err(PsetErr::Cxt(format!(
                "{} does not correspond to a supported graph dimension: 0, 2, 3, 4.",
                value
            ))),
        }
    }
}
