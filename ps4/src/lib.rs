use std::ops::RangeInclusive;

#[derive(Debug)]
pub struct PrettyPrint<'a, T> {
    pref_len: usize,
    text: &'a [T],
    line_breaks: Vec<usize>,
}

#[derive(Clone, Copy, Debug)]
struct Cached {
    cost: i32,
    next_line: Option<usize>,
}

impl<'a, T> PrettyPrint<'a, T>
where
    T: AsRef<str>,
{
    pub fn new(preferred_len: usize, text: &'a [T]) -> Self {
        PrettyPrint {
            pref_len: preferred_len,
            text,
            line_breaks: Vec::new(),
        }
    }

    /// Runs a DP algorithm on the input paragraph to determine the most
    /// optimal placement of line breaks.
    pub fn find_pretty(&mut self) -> anyhow::Result<()> {
        let mut dp = vec![Cached::default(); self.text.len()];

        for (startline, word) in self.text.iter().rev().enumerate() {
            let mut chars_in_range: usize = 0;
            for eol in (startline + 1)..self.text.len() {
                chars_in_range += word.as_ref().len();
                let cost_of_interval = self.cost(startline..=eol, chars_in_range)?;
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

    /// Given the DP array for the current paragraph, searches the array and extracts
    /// line breaks for printing from the first word.
    fn add_line_breaks(&mut self, dp: &[Cached]) {
        let mut next_line = dp.get(0).map(|entry| entry.next_line).flatten();
        while let Some(ind) = next_line {
            self.line_breaks.push(ind);
            next_line = dp.get(ind).map(|entry| entry.next_line).flatten();
        }
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
        // end is incremented because the formula assumes inclusive
        let closeness: i32 = pref_len - (end + 1) + start - char_ct;
        if closeness < 0 {
            return Ok(2i32.pow(closeness.abs().try_into()?) - closeness.pow(3) - 1);
        }
        Ok(closeness.pow(3).into())
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
