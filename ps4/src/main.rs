use ps4::PrettyPrint;
use std::{
    env,
    io::{self},
};

fn usage() {
    println!("usage: cargo run [int preference] [normal (default), verbose]");
    println!("ex: cat test_data/yes.txt | cargo run 17");
    println!("ex: cat test_data/yes.txt | cargo run 17 verbose");
}

#[derive(Debug, PartialEq)]
enum OutputLen {
    Normal,
    Verbose,
}

/// Takes a CLI argument for the preferred character width of a line.
/// Reads text from stdin and pretty prints it given the character preference.
fn main() -> anyhow::Result<()> {
    let mut args = env::args().skip(1);
    let Some(pref_len) = args.next().map(|arg| arg.parse::<usize>().ok()).flatten() else {
        usage();
        return Ok(());
    };
    let verbosity: OutputLen = args
        .next()
        .map(|arg| arg.as_str().try_into().ok())
        .flatten()
        .unwrap_or(OutputLen::Normal);

    let input: Vec<String> = io::stdin().lines().filter_map(Result::ok).collect();
    let trimmed: Vec<&str> = input
        .iter()
        .flat_map(|txt| txt.split_whitespace())
        .collect();

    let mut pretty = PrettyPrint::new(pref_len, &trimmed);
    pretty.find_pretty()?;
    pretty.print_preference()?;
    pretty.print()?;
    if verbosity == OutputLen::Verbose {
        pretty.print_line_breaks();
    }

    Ok(())
}

impl TryFrom<&str> for OutputLen {
    type Error = anyhow::Error;
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "normal" => Ok(Self::Normal),
            "verbose" => Ok(Self::Verbose),
            _ => Err(anyhow::Error::msg(format!(
                "Please choose verbosity 'normal' or 'verbose'"
            ))),
        }
    }
}
