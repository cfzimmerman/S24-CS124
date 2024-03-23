use super::matrix::Matrix;

/// Manages operations for adding and removing matrix padding.
#[derive(Debug)]
pub struct PadPow2 {
    pub rows_init_ct: usize,
    pub cols_init_ct: usize,
}

impl PadPow2 {
    /// Pads the given matrix so that its dimensions are square and the minimum
    /// power of two possible by only adding rows and columns of default values.
    pub fn new<T>(mtx: &mut Matrix<T>) -> Self
    where
        T: Default + Clone + Copy,
    {
        let pp2 = PadPow2 {
            rows_init_ct: mtx.num_rows(),
            cols_init_ct: mtx.num_cols(),
        };
        Self::pad_pow2(mtx);
        pp2
    }

    /// Undoes padding from the matrix.
    pub fn undo<T>(&self, mtx: &mut Matrix<T>) {
        Self::trim_dims(mtx, self.rows_init_ct, self.cols_init_ct);
    }

    /// Returns the minimum number `n` such that `num <= 2^n`
    fn round_up_nearest_pow2(num: usize) -> u32 {
        (num as f64).log2().ceil() as u32
    }

    /// Pads self with zeroes to the bottom and right so that its dimensions
    /// are a power of 2.
    pub fn pad_pow2<T>(mtx: &mut Matrix<T>)
    where
        T: Default + Clone + Copy,
    {
        let max_dim = mtx.num_rows().max(mtx.num_cols());
        let add_rows = 2usize.pow(Self::round_up_nearest_pow2(max_dim)) - mtx.num_rows();
        let add_cols = 2usize.pow(Self::round_up_nearest_pow2(max_dim)) - mtx.num_cols();
        for curr_row in &mut mtx.inner {
            curr_row.extend(vec![T::default(); add_cols]);
            debug_assert!((curr_row.len() as f64).log2().fract() == 0.);
        }
        for _ in 0..add_rows {
            mtx.inner
                .push(vec![T::default(); mtx.num_cols() + add_cols]);
        }
        debug_assert!((mtx.inner.len() as f64).log2().fract() == 0.);
    }

    /// Removes entries from the bottom and right sides until self reaches the
    /// final dimensions given.
    pub fn trim_dims<T>(mtx: &mut Matrix<T>, final_num_rows: usize, final_num_cols: usize) {
        mtx.inner.truncate(final_num_rows);
        if mtx.num_cols() > final_num_cols {
            for row in &mut mtx.inner {
                row.truncate(final_num_cols);
            }
        }
    }
}
