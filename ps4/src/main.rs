use ps4::PrettyPrint;
use std::{
    env::args,
    io::{self},
};

fn usage() {
    println!("usage: cargo run [int preference]");
    println!("ex: cat test_data/yes.txt | cargo run 17");
}

/// Takes a CLI argument for the preferred character width of a line.
/// Reads text from stdin and pretty prints it given the character preference.
fn main() -> anyhow::Result<()> {
    let Some(pref_len) = args()
        .skip(1)
        .next()
        .map(|arg| arg.parse::<usize>().ok())
        .flatten()
    else {
        usage();
        return Ok(());
    };

    let input: Vec<String> = io::stdin().lines().filter_map(Result::ok).collect();
    let trimmed: Vec<&str> = input
        .iter()
        .flat_map(|txt| txt.split_whitespace())
        .collect();

    let mut pretty = PrettyPrint::new(pref_len, &trimmed);
    pretty.find_pretty()?;
    pretty.print_preference()?;
    pretty.print()?;

    Ok(())
}
