use std::{fmt::Display, ops::RangeInclusive};

#[derive(Debug)]
pub struct PrettyPrint<'a, T> {
    pref_len: usize,
    text: &'a [T],
    line_breaks: Vec<usize>,
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
    pub fn new(preferred_len: usize, text: &'a [T]) -> Self {
        PrettyPrint {
            pref_len: preferred_len,
            text,
            line_breaks: Vec::new(),
            total_cost: 0,
        }
    }

    /// Runs a DP algorithm on the input paragraph to determine the most
    /// optimal placement of line breaks.
    pub fn find_pretty(&mut self) -> anyhow::Result<()> {
        let mut dp = vec![Cached::default(); self.text.len()];

        if let (Some(last_word), Some(last_dp)) = (self.text.last(), dp.last_mut()) {
            // base case
            let last_ind = self.text.len() - 1;
            last_dp.cost = self.cost(last_ind..=last_ind, last_word.as_ref().len())?;
        }
        for startline in (0..self.text.len()).rev() {
            let mut chars_in_range: usize = 0;
            for (eol, end_word) in self.text.iter().enumerate().skip(startline) {
                chars_in_range += end_word.as_ref().len();
                let cost_of_interval = self.cost(startline..=eol, chars_in_range)?;
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
        self.add_line_breaks(&dp);
        Ok(())
    }

    /// Pretty-prints the input. If find_pretty has not been called, there will be
    /// no added line breaks.
    pub fn print(&self) -> anyhow::Result<()> {
        let mut formatted = String::with_capacity(self.line_breaks.len() * self.pref_len);
        let mut line_break_iter = self.line_breaks.iter();
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
        println!("total cost: {}", self.total_cost);
        Ok(())
    }

    /// Given the DP array for the current paragraph, searches the array and extracts
    /// line breaks for printing from the first word.
    fn add_line_breaks(&mut self, dp: &[Cached]) {
        let mut next_line = dp
            .get(0)
            .map(|entry| {
                self.total_cost += entry.cost;
                entry.next_line
            })
            .flatten();
        while let Some(ind) = next_line {
            self.total_cost += dp[ind].cost;
            self.line_breaks.push(ind);
            next_line = dp.get(ind).map(|entry| entry.next_line).flatten();
        }
    }

    /// Prints a benchmark for how wide the preferred window of text is
    pub fn print_preference(&self) -> anyhow::Result<()> {
        let chars: Vec<u8> = vec!['#'.try_into()?; self.pref_len];
        println!("\n{}", String::from_utf8(chars)?);
        Ok(())
    }

    /// Returns the cost of a line bounded by the given range of words with words
    /// in the range having a sum of `char_ct` characters.
    fn cost(&self, words: RangeInclusive<usize>, char_ct: usize) -> anyhow::Result<i32> {
        let (start, end, char_ct, pref_len): (i32, i32, i32, i32) = (
            (*words.start()).try_into()?,
            (*words.end()).try_into()?,
            char_ct.try_into()?,
            self.pref_len.try_into()?,
        );
        let closeness: i32 = pref_len - end + start - char_ct;
        if closeness >= 0 && words.end() + 1 == self.text.len() {
            // "The last line has no penalties if A >= 0"
            return Ok(0);
        }
        if closeness < 0 {
            let first_term = 2i32.saturating_pow(closeness.abs().try_into()?);
            let second_term = closeness.saturating_pow(3);
            return Ok(first_term.saturating_sub(second_term).saturating_sub(1));
        }
        Ok(closeness.saturating_pow(3).into())
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
                let mut pretty = PrettyPrint::new(preference, &text);
                pretty.find_pretty()?;
                assert!(
                    pretty.line_breaks.len() < prev_break_ct,
                    "line break counts should decrease as line preference increases"
                );
                prev_break_ct = pretty.line_breaks.len();
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
            let mut pretty = PrettyPrint::new(2, &text);
            pretty.find_pretty()?;
            assert!(
                pretty.total_cost == 0,
                "preference 2: input should each have its own line"
            );
        }
        {
            let mut pretty = PrettyPrint::new(5, &text);
            pretty.find_pretty()?;
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
