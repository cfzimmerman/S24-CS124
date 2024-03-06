use std::{fmt::Display, ops::RangeInclusive};

#[derive(Debug)]
pub struct PrettyPrint<'a, T> {
    pref_len: i32,
    text: &'a [T],
    new_lines: Vec<usize>,
    total_cost: i32,
}

#[derive(Clone, Copy, Debug)]
struct Cached {
    cost: i32,
    next_line: Option<usize>,
}

impl<'a, T> PrettyPrint<'a, T>
where
    T: AsRef<str> + Display,
{
    /// Runs a DP algorithm on the input paragraph to determine the most
    /// optimal placement of line breaks.
    pub fn build(preferred_len: usize, text: &'a [T]) -> anyhow::Result<Self> {
        let mut pretty = PrettyPrint {
            pref_len: preferred_len.try_into()?,
            text,
            new_lines: Vec::new(),
            total_cost: 0,
        };
        let mut dp = vec![Cached::default(); text.len()];
        if let (Some(last_word), Some(last_dp)) = (text.last(), dp.last_mut()) {
            // base case
            let last_ind = text.len() - 1;
            last_dp.cost = Self::cost(
                last_ind..=last_ind,
                last_word.as_ref().len(),
                pretty.pref_len,
                text,
            )?;
        }

        for startline in (0..text.len()).rev() {
            let mut chars_in_range: usize = 0;
            for (eol, end_word) in text.iter().enumerate().skip(startline) {
                chars_in_range += end_word.as_ref().len();
                let cost_of_interval = Self::cost(
                    startline..=eol,
                    chars_in_range,
                    pretty.pref_len,
                    pretty.text,
                )?;
                if cost_of_interval >= dp[startline].cost {
                    continue;
                }
                let cost_of_rest = dp.get(eol + 1).map(|c| c.cost).unwrap_or(0);
                let cost_if_break_here = cost_of_interval + cost_of_rest;
                if cost_if_break_here < dp[startline].cost {
                    dp[startline] = Cached {
                        cost: cost_if_break_here,
                        next_line: dp.get(eol + 1).map(|_| eol + 1),
                    };
                }
            }
        }
        pretty.add_line_breaks(&dp);
        Ok(pretty)
    }

    /// Pretty-prints the input. If find_pretty has not been called, there will be
    /// no added line breaks.
    pub fn print(&self) -> anyhow::Result<()> {
        let mut formatted = String::with_capacity(self.new_lines.len() * self.pref_len as usize);
        let mut line_break_iter = self.new_lines.iter().skip(1);
        let mut next_line_break = line_break_iter.next();

        for (ind, word) in self.text.iter().enumerate() {
            if let Some(br) = next_line_break {
                if &ind == br {
                    formatted.push('\n');
                    next_line_break = line_break_iter.next();
                }
            }
            formatted.push_str(word.as_ref());
            formatted.push(' ');
        }
        println!("{formatted}");
        println!("\ntotal cost: {}", self.total_cost);
        Ok(())
    }

    /// Prints a benchmark for how wide the preferred window of text is
    pub fn print_preference(&self) -> anyhow::Result<()> {
        let chars: Vec<u8> = vec!['#'.try_into()?; self.pref_len as usize];
        println!("\n{}", String::from_utf8(chars)?);
        Ok(())
    }

    /// Prints the list of line break indices
    pub fn print_new_lines(&self) {
        println!("new line indices: {:?}", self.new_lines);
    }

    /// Returns the cost of a line bounded by the given range of words with words
    /// in the range having a sum of `char_ct` characters.
    fn cost(
        words: RangeInclusive<usize>,
        char_ct: usize,
        pref_len: i32,
        text: &[T],
    ) -> anyhow::Result<i32> {
        let (start, end, char_ct): (i32, i32, i32) = (
            (*words.start()).try_into()?,
            (*words.end()).try_into()?,
            char_ct.try_into()?,
        );
        let closeness: i32 = pref_len - end + start - char_ct;
        if closeness >= 0 && words.end() + 1 == text.len() {
            // "The last line has no penalties if A >= 0"
            return Ok(0);
        }
        if closeness < 0 {
            let first_term = 2i32.saturating_pow(closeness.abs().try_into()?);
            let second_term = closeness.saturating_pow(3);
            return Ok(first_term.saturating_sub(second_term) - 1);
        }
        Ok(closeness.saturating_pow(3))
    }

    /// Given the DP array for the current paragraph, searches the array and extracts
    /// line breaks for printing from the first word.
    fn add_line_breaks(&mut self, dp: &[Cached]) {
        let mut next_line = dp.first().and_then(|entry| {
            self.total_cost = entry.cost;
            self.new_lines.push(0);
            entry.next_line
        });
        while let Some(ind) = next_line {
            self.new_lines.push(ind);
            next_line = dp.get(ind).and_then(|entry| entry.next_line);
        }
    }
}

impl Default for Cached {
    fn default() -> Self {
        Self {
            cost: i32::MAX,
            next_line: None,
        }
    }
}

#[cfg(test)]
mod test_pretty {
    use crate::PrettyPrint;
    use std::fs;

    /// Retrieves strings from text files
    fn get_test_texts() -> anyhow::Result<Vec<String>> {
        let files = ["./test_data/bee.txt", "./test_data/yes.txt"];
        let mut res = Vec::with_capacity(files.len());
        for path in files {
            res.push(fs::read_to_string(path)?);
        }
        Ok(res)
    }

    /// Verifies that the number of line breaks decreases as the
    /// line length preference increases.
    #[test]
    fn breaks_correspond_to_pref() -> anyhow::Result<()> {
        for text in get_test_texts()? {
            let mut prev_break_ct = usize::MAX;
            let text: Vec<&str> = text.split(' ').collect();
            for preference in [17, 32, 64] {
                let pretty = PrettyPrint::build(preference, &text)?;
                assert!(
                    pretty.new_lines.len() < prev_break_ct,
                    "line break counts should decrease as line preference increases"
                );
                prev_break_ct = pretty.new_lines.len();
                pretty.print_preference()?;
                pretty.print()?;
            }
        }
        Ok(())
    }

    #[test]
    fn cost() -> anyhow::Result<()> {
        let text = ["aa", "bb", "cc", "dd", "ee"];
        {
            let pretty = PrettyPrint::build(2, &text)?;
            assert!(
                pretty.total_cost == 0,
                "preference 2: input should each have its own line"
            );
        }
        {
            let pretty = PrettyPrint::build(5, &text)?;
            // pretty.print_preference()?;
            // pretty.print()?;
            assert!(
                pretty.total_cost == 0,
                "preference 5: input should still print perfectly"
            );
        }
        Ok(())
    }
}
