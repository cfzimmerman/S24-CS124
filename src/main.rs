use csv::Writer;
use progset2::{
    error::PsetRes,
    mtx::{
        cli::{CliInput, MtxCli},
        experiments::{BaseExperiment, TriangleExperiment},
        matrix::Matrix,
    },
};
use std::{
    env,
    fs::File,
    io::{stdin, stdout, BufReader},
};

const DEFAULT_BASE: usize = 64;
type InputT = i32;

fn main() -> PsetRes<()> {
    let args: Vec<String> = env::args().collect();
    let mut base = DEFAULT_BASE;
    let mut grading_output = false;

    let (mut left, mut right): (Matrix<InputT>, Matrix<InputT>) = match MtxCli::parse_args(&args)? {
        CliInput::Grading { dim, file_path } => {
            grading_output = true;
            MtxCli::read_sq_matrices(BufReader::new(File::open(file_path)?), dim)?
        }
        CliInput::Stdin { dim } => MtxCli::read_sq_matrices(BufReader::new(stdin().lock()), dim)?,
        CliInput::Cutoff {
            dim,
            base_cutoff,
            file_path,
        } => {
            base = base_cutoff;
            MtxCli::read_sq_matrices(BufReader::new(File::open(file_path)?), dim)?
        }
        CliInput::BaseExperiment {
            input_file,
            output_file,
        } => {
            println!("Running base experiment");
            let mut expr = BaseExperiment::from_cfg(&input_file)?;
            let mut csv = Writer::from_path(output_file)?;
            expr.run(&mut csv)?;
            return Ok(());
        }
        CliInput::TriangleExperiment {
            input_file,
            output_file,
        } => {
            println!("Running triangle experiment");
            let expr = TriangleExperiment::from_cfg(&input_file)?;
            let mut csv = Writer::from_path(output_file)?;
            expr.run(&mut csv)?;
            return Ok(());
        }
    };

    let res = Matrix::mul_strassen(&mut left, &mut right, base)?;
    if grading_output {
        MtxCli::write_diagonal(&res, &mut stdout().lock())?;
    } else {
        println!("{res}");
    }
    Ok(())
}
