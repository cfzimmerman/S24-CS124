use progset2::{
    error::PsetRes,
    mtx::{
        cli::{CliInput, MtxCli},
        matrix::Matrix,
    },
};
use std::{
    env,
    fs::File,
    io::{stdin, stdout, BufReader},
};

const DEFAULT_BASE: usize = 3;
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
    };

    let res = Matrix::mul_strassen(&mut left, &mut right, base)?;
    if grading_output {
        MtxCli::write_result(&res, &mut stdout().lock())?;
    } else {
        println!("{res}");
    }
    Ok(())
}
