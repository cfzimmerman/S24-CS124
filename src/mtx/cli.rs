use super::matrix::Matrix;
use crate::error::{PsetErr, PsetRes};
use std::{
    fmt::Display,
    io::{BufRead, Write},
    path::PathBuf,
    str::FromStr,
};

/// Each CLI input configuration is declared by a variant of this enum.
#[derive(Debug)]
pub enum CliInput {
    Grading {
        dim: usize,
        file_path: PathBuf,
    },
    Stdin {
        dim: usize,
    },
    Cutoff {
        dim: usize,
        base_cutoff: usize,
        file_path: PathBuf,
    },
    BaseExperiment {
        input_file: PathBuf,
        output_file: PathBuf,
    },
    TriangleExperiment {
        input_file: PathBuf,
        output_file: PathBuf,
    },
}

/// Manages CLI interaction with user input.
pub struct MtxCli;

impl MtxCli {
    /// Prints information for the user on how to use this tool.
    fn usage() -> PsetErr {
        let msg = r#"
        This program performs Matrix multiplication using Strassen's algorithm.
        Stdin and file input expects newline-separated integer entries. 
        If multiplying two N-dimensional (square) matrices, the input file should 
        have 2 * n^2 entries. `cutoff` configures the matrix dimension at which 
        Strassen's algorithm switches to iterative. 

        Mode = 0, grading variant:
        ./** 0 [uint dimension] [file path]
        ex: ./** 0 4 m4x4.txt 

        Mode = 1, stdin variant: 
        ./** 1 [uint dimension] 
        ex: cat m4x4.txt | ./** 1 4

        Mode = 2, cutoff variant:
        ./** 2 [uint dimension] [uint > 2 cutoff] [file path]
        ex: ./** 2 4 3 m4x4.txt

        Mode = 3, base experiments:
        ./** 3 [input file path] [output file path]
        ex: ./** 3 ./base_experiment.json ./base_result.csv

        Mode = 4, triangle experiments:
        ./** 4 [input file path] [output file path] 
        ex: ./** 4 ./tri_experiment.json ./tri_result.csv"#;

        eprintln!("{msg}");
        PsetErr::InvalidInput
    }

    /// Retrieves a usize argument from an index of the given args. Prints and returns
    /// usage if out of bounds.
    fn get_usize(args: &[String], ind: usize) -> PsetRes<usize> {
        let Some(val) = args.get(ind).and_then(|val| val.parse::<usize>().ok()) else {
            return Err(Self::usage());
        };
        Ok(val)
    }

    /// Retrieves a file path argument from the given index in args. Validates the
    /// path and gives feedback to the user if the argument is invalid.
    fn get_path(args: &[String], int: usize) -> PsetRes<PathBuf> {
        let Some(path_str) = args.get(int) else {
            return Err(Self::usage());
        };
        let path: PathBuf = path_str.into();
        if !path.try_exists()? {
            eprintln!("Failed to verify path: {:?}", path);
            return Err(PsetErr::InvalidInput);
        }
        Ok(path)
    }

    /// Reads user CLI input and attempts to match that to a CliInput variant.
    pub fn parse_args(args: &[String]) -> PsetRes<CliInput> {
        let mode = Self::get_usize(args, 1)?;
        let mut dim = 0;
        if mode < 3 {
            dim = Self::get_usize(args, 2)?;
        };

        match mode {
            0 => Ok(CliInput::Grading {
                dim,
                file_path: Self::get_path(args, 3)?,
            }),
            1 => Ok(CliInput::Stdin { dim }),
            2 => Ok(CliInput::Cutoff {
                dim,
                base_cutoff: Self::get_usize(args, 3)?,
                file_path: Self::get_path(args, 4)?,
            }),
            3 => Ok(CliInput::BaseExperiment {
                input_file: Self::get_path(args, 2)?,
                output_file: Self::get_path(args, 3)?,
            }),
            4 => Ok(CliInput::TriangleExperiment {
                input_file: Self::get_path(args, 2)?,
                output_file: Self::get_path(args, 3)?,
            }),
            other => {
                eprintln!("Unrecognized mode: {other}. Modes 0 and 1 are supported");
                Err(PsetErr::InvalidInput)
            }
        }
    }

    /// Prints to the writer each entry on the diagonal of mtx with one entry
    /// per line (as requested by the grading spec).
    pub fn write_diagonal<W, T>(mtx: &Matrix<T>, writer: &mut W) -> PsetRes<()>
    where
        W: Write,
        T: Display,
    {
        for (ind, row) in mtx.inner.iter().enumerate() {
            writeln!(writer, "{}", row[ind])?;
        }
        writer.flush()?;
        Ok(())
    }

    /// Given a buffer over newline-separated entries of type T, attempts to
    /// build and return two dim x dim square matrices.
    pub fn read_sq_matrices<B, T>(buf: B, dim: usize) -> PsetRes<(Matrix<T>, Matrix<T>)>
    where
        B: BufRead,
        T: FromStr,
        Matrix<T>: From<Vec<Vec<T>>>,
    {
        let mut lines = buf
            .lines()
            .filter_map(|line| line.ok().and_then(|txt| txt.parse::<T>().ok()));
        let mut matrices: [_; 2] = [Vec::with_capacity(dim), Vec::with_capacity(dim)];
        for matrix in &mut matrices {
            for _ in 0..dim {
                let mut row = Vec::with_capacity(dim);
                for _ in 0..dim {
                    row.push(lines.next().ok_or_else(|| {
                        PsetErr::Cxt(format!(
                            "Expected dim {dim} reader to have {} entries",
                            dim.pow(2) * 2
                        ))
                    })?);
                }
                matrix.push(row);
            }
        }
        let mut mtx_iter = matrices.into_iter();
        Ok((
            // Unwrapping is okay because these are guaranteed by array init above.
            mtx_iter.next().unwrap().into(),
            mtx_iter.next().unwrap().into(),
        ))
    }
}

/*
    let mut res: Vec<SquareTestMatrix> = Vec::with_capacity(RAW_SQUARES.len());
    for (path, dim) in RAW_SQUARES {

        let mut left_right_prod = vec![
            Vec::with_capacity(dim),
            Vec::with_capacity(dim),
            Vec::with_capacity(dim),
        ];
        for matrix in &mut left_right_prod {
        }

        let mut lrp = left_right_prod.into_iter();
        res.push(SquareTestMatrix {
            // unwrapping will never panic because the three entries are added just above
            left: lrp.next().unwrap(),
            right: lrp.next().unwrap(),
            prod: lrp.next().unwrap(),
            sum: Vec::new(),
            diff: Vec::new(),
        });
    }
    Ok(res)

*/
