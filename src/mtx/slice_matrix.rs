use super::matrix::Matrix;
use crate::error::{PsetErr, PsetRes};
use std::{
    fmt::{self, Debug, Display},
    ops::{Add, AddAssign, Mul, Range, Sub},
};

/// A flexible window over data held in some other Matrix.
/// Rows and columns index into the parent matrix.
/// Use Display formatting for a more attractive printing
/// of the sliced matrix.
#[derive(Debug, Clone)]
pub struct SliceMatrix<'a, T> {
    parent: &'a Matrix<T>,
    rows: Range<usize>,
    cols: Range<usize>,
}

impl<'a, T> SliceMatrix<'a, T> {
    /// Returns the number of rows in this matrix WITH PADDING
    pub fn num_rows(&self) -> usize {
        debug_assert!(
            self.rows.end <= self.parent.num_rows(),
            "Slice matrix row range should always be in bounds"
        );
        self.rows.end - self.rows.start
    }

    /// Returns the number of columns in this matrix
    pub fn num_cols(&self) -> usize {
        debug_assert!(
            self.cols.end <= self.parent.num_cols(),
            "Slice matrix col range should always be in bounds"
        );
        self.cols.end - self.cols.start
    }
}

impl<'a, T> SliceMatrix<'a, T>
where
    T: AddAssign + Default + Copy + Debug + 'static,
    for<'b> &'b T: Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    /// Retrieves an entry from a row, col in the slice.
    /// COORDS ARE RELATIVE TO THE SLICE.
    /// Ex, the top left coord in the slice matrix is 0, 0 regardless
    /// of where it falls on the parent matrix.
    pub fn get(&self, req_row: usize, req_col: usize) -> Option<&T> {
        let parent_row = self.rows.start + req_row;
        let parent_col = self.cols.start + req_col;
        if !self.rows.contains(&parent_row) || !self.cols.contains(&parent_col) {
            return None;
        }
        self.parent
            .inner
            .get(parent_row)
            .and_then(|r| r.get(parent_col))
    }

    /// Indexes into the requested row and column of the slice. Indices
    /// are relative to the slice, not the parent matrix.
    /// Panics if the indices are out of bounds of the slice dimensions.
    pub fn index(&self, req_row: usize, req_col: usize) -> &T {
        self.get(req_row, req_col)
            .expect("Accessed out of bounds slice index")
    }

    /// Returns a matrix representing the operation left + right
    pub fn add<'b>(left: &SliceMatrix<'b, T>, right: &SliceMatrix<'b, T>) -> PsetRes<Matrix<T>> {
        Self::op_one_to_one(left, right, |l, r| l + r)
    }

    /// Returns a matrix representing the operation left - right
    pub fn sub<'b>(left: &SliceMatrix<'b, T>, right: &SliceMatrix<'b, T>) -> PsetRes<Matrix<T>> {
        Self::op_one_to_one(left, right, |l, r| l - r)
    }

    /// Performs O(n^3) iterative multiplication on the given matrices.
    /// If the input matrices are padded, the output matrix is computed
    /// without padding.
    pub fn mul_iter<'b>(
        left: &SliceMatrix<'b, T>,
        right: &SliceMatrix<'b, T>,
    ) -> PsetRes<Matrix<T>> {
        if left.num_cols() != right.num_rows() {
            return Err(PsetErr::Static("Matrix dims don't support right multiply"));
        }
        let mut res: Vec<Vec<T>> = Vec::with_capacity(left.num_rows());
        for left_row_off in 0..left.num_rows() {
            let mut row: Vec<T> = Vec::with_capacity(right.num_rows());
            for right_col_off in 0..right.num_cols() {
                let mut sum = T::default();
                for offset in 0..left.num_cols() {
                    // indexing is safe because the initial check guarantees dimensions, and it's
                    // invariant that slices do no exceed the bounds of the parent matrix.
                    sum += left.index(left_row_off, offset) * right.index(offset, right_col_off);
                }
                row.push(sum);
            }
            res.push(row);
        }
        Ok(Matrix { inner: res })
    }

    /// Performs O(n^3) recursive multiplication. Used as a stepping stone before
    /// Strassen's algorithm.
    pub fn mul_naive_rec<'b>(
        left: &SliceMatrix<'b, T>,
        right: &SliceMatrix<'b, T>,
        base_sz: usize,
    ) -> PsetRes<Matrix<T>> {
        if left.num_cols() <= base_sz || left.num_rows() <= base_sz {
            return Self::mul_iter(left, right);
        }

        let left_quad = left.try_split_quad()?;
        let right_quad = right.try_split_quad()?;

        let mut tl1 = Self::mul_naive_rec(&left_quad.top_left, &right_quad.top_left, base_sz)?;
        let tl2 = Self::mul_naive_rec(&left_quad.top_right, &right_quad.bottom_left, base_sz)?;

        let mut bl1 = Self::mul_naive_rec(&left_quad.bottom_left, &right_quad.top_left, base_sz)?;
        let bl2 = Self::mul_naive_rec(&left_quad.bottom_right, &right_quad.bottom_left, base_sz)?;

        let mut tr1 = Self::mul_naive_rec(&left_quad.top_left, &right_quad.top_right, base_sz)?;
        let tr2 = Self::mul_naive_rec(&left_quad.top_right, &right_quad.bottom_right, base_sz)?;

        let mut br1 = Self::mul_naive_rec(&left_quad.bottom_left, &right_quad.top_right, base_sz)?;
        let br2 = Self::mul_naive_rec(&left_quad.bottom_right, &right_quad.bottom_right, base_sz)?;

        tl1.add_in_place(&tl2)?;
        bl1.add_in_place(&bl2)?;
        tr1.add_in_place(&tr2)?;
        br1.add_in_place(&br2)?;
        tl1.merge_neighbors(bl1, tr1, br1)?;

        Ok(tl1)
    }

    /// Performs Strassen's algorithm on the given SliceMatrix instances, switching to iterative
    /// multiplication when sub-matrices are smaller than or equal to base_sz.
    /// Fails if left and right are not a power of 2. Apply padding before multiplication if
    /// needed.
    pub fn mul_strassen<'b>(
        left: &SliceMatrix<'b, T>,
        right: &SliceMatrix<'b, T>,
        base_sz: usize,
    ) -> PsetRes<Matrix<T>> {
        if left.num_cols() <= base_sz || left.num_rows() <= base_sz {
            return Self::mul_iter(left, right);
        }

        let left_quad = left.try_split_quad()?;
        let right_quad = right.try_split_quad()?;

        // Using this algorithm:
        // https://en.wikipedia.org/wiki/Strassen_algorithm
        let mut m1 = {
            let part_a = SliceMatrix::add(&left_quad.top_left, &left_quad.bottom_right)?;
            let part_b = SliceMatrix::add(&right_quad.top_left, &right_quad.bottom_right)?;
            Self::mul_strassen(&(&part_a).into(), &(&part_b).into(), base_sz)?
        };
        let m2 = {
            let part_a = SliceMatrix::add(&left_quad.bottom_left, &left_quad.bottom_right)?;
            Self::mul_strassen(&(&part_a).into(), &right_quad.top_left, base_sz)?
        };
        let mut m3 = {
            let part_b = SliceMatrix::sub(&right_quad.top_right, &right_quad.bottom_right)?;
            Self::mul_strassen(&left_quad.top_left, &(&part_b).into(), base_sz)?
        };
        let mut m4 = {
            let part_b = SliceMatrix::sub(&right_quad.bottom_left, &right_quad.top_left)?;
            Self::mul_strassen(&left_quad.bottom_right, &(&part_b).into(), base_sz)?
        };
        let m5 = {
            let part_a = SliceMatrix::add(&left_quad.top_left, &left_quad.top_right)?;
            Self::mul_strassen(&(&part_a).into(), &right_quad.bottom_right, base_sz)?
        };
        let m6 = {
            let part_a = SliceMatrix::sub(&left_quad.bottom_left, &left_quad.top_left)?;
            let part_b = SliceMatrix::add(&right_quad.top_left, &right_quad.top_right)?;
            Self::mul_strassen(&(&part_a).into(), &(&part_b).into(), base_sz)?
        };
        let m7 = {
            let part_a = SliceMatrix::sub(&left_quad.top_right, &left_quad.bottom_right)?;
            let part_b = SliceMatrix::add(&right_quad.bottom_left, &right_quad.bottom_right)?;
            Self::mul_strassen(&(&part_a).into(), &(&part_b).into(), base_sz)?
        };

        // done in order that minimizes the need for copying
        let final_bottom_left = Self::add(&(&m2).into(), &(&m4).into())?;
        let mut final_top_left = {
            m4.add_in_place(&m1)?.sub_in_place(&m5)?.add_in_place(&m7)?;
            m4
        };
        let final_bottom_right = {
            m1.sub_in_place(&m2)?.add_in_place(&m3)?.add_in_place(&m6)?;
            m1
        };
        let final_top_right = {
            m3.add_in_place(&m5)?;
            m3
        };
        final_top_left.merge_neighbors(final_bottom_left, final_top_right, final_bottom_right)?;
        Ok(final_top_left)
    }

    /// If successful, returns the slice matrix partitioned into four equal
    /// quadrants.
    /// Fails unless the given matrix is an even-dimension square. The method
    /// `pad_even_square` can be helpful getting there.
    fn try_split_quad(&self) -> PsetRes<SplitQuad<T>> {
        SplitQuad::build(self)
    }

    /// Attempts to build a new slice matrix from a range within
    /// the existing one.
    /// RANGES ARE IN RELATION TO THE PARENT SLICE, NOT THE PARENT MATRIX.
    /// Ex. Requesting a range beginning at row 0 yields row 0 of the slice
    /// no matter where that index occurs in the parent matrix.
    fn slice_range(
        &self,
        req_rows: &Range<usize>,
        req_cols: &Range<usize>,
    ) -> PsetRes<SliceMatrix<T>> {
        let req_row_len = req_rows.end - req_rows.start;
        let req_col_len = req_cols.end - req_cols.start;

        if self.num_rows() < req_row_len || self.num_cols() < req_col_len {
            return Err(PsetErr::Static(
                "slice_range: Requested dims exceed parent slice",
            ));
        }

        let sl_row_start = self.rows.start + req_rows.start;
        let sl_col_start = self.cols.start + req_cols.start;

        if self.rows.end <= sl_row_start || self.cols.end <= sl_col_start {
            return Err(PsetErr::Static(
                "slice_range: Requested start exceeds parent end",
            ));
        }

        let sl_rows = sl_row_start..(sl_row_start + req_row_len);
        let sl_cols = sl_col_start..(sl_col_start + req_col_len);
        debug_assert!(sl_rows.end <= self.rows.end);
        debug_assert!(sl_cols.end <= self.cols.end);

        Ok(SliceMatrix {
            parent: self.parent,
            rows: sl_rows,
            cols: sl_cols,
        })
    }

    /// Performs an operation on parallel entries in the given matrices, writing the output
    /// to a new matrix. Useful for addition and subtraction.
    /// Materializes imaginary padding into the result as T::default() values.
    fn op_one_to_one<'b>(
        left: &SliceMatrix<'b, T>,
        right: &SliceMatrix<'b, T>,
        op: fn(&T, &T) -> T,
    ) -> PsetRes<Matrix<T>> {
        if left.num_rows() != right.num_rows() || left.num_cols() != right.num_cols() {
            return Err(PsetErr::Static(
                "Cannot perform a one to one operation on matrices of different dimensions",
            ));
        }
        let mut res = Vec::with_capacity(left.num_rows());
        for row_offset in 0..left.num_rows() {
            let mut row = Vec::with_capacity(left.num_cols());
            for col_offset in 0..left.num_cols() {
                // Indexing is okay because the initial check
                // guarantees that bounds are safe, and slice indices are
                // always in-bounds of the parent matrix.
                row.push(op(
                    left.index(row_offset, col_offset),
                    right.index(row_offset, col_offset),
                ));
            }
            res.push(row);
        }
        Ok(res.into())
    }
}

/// Returns the successful partition of a single SliceMatrix into
/// four sub-slices.
#[derive(Debug)]
struct SplitQuad<'a, T> {
    top_left: SliceMatrix<'a, T>,
    bottom_left: SliceMatrix<'a, T>,
    top_right: SliceMatrix<'a, T>,
    bottom_right: SliceMatrix<'a, T>,
}

impl<'a, T> SplitQuad<'a, T>
where
    T: AddAssign + Default + Copy + Debug + 'static,
    for<'b> &'b T: Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    /// Returns that matrix partitioned into four equal quadrants.
    /// Fails if the given parent slice is not at least 2x2, square, and even.
    fn build(mtx: &'a SliceMatrix<'a, T>) -> PsetRes<Self> {
        if mtx.num_rows() < 2 || mtx.num_cols() < 2 {
            return Err(PsetErr::Static(
                "Matrix must be at least 2x2 to split as quad.",
            ));
        }
        if mtx.num_rows() != mtx.num_cols() || mtx.num_rows() % 2 != 0 {
            return Err(PsetErr::Static("SplitQuad requires an even square matrix."));
        }
        let quad_rows = mtx.num_rows() / 2;
        let quad_cols = mtx.num_cols() / 2;

        let (top_rows, bottom_rows) = (0..quad_rows, quad_rows..(quad_rows * 2));
        let (left_cols, right_cols) = (0..quad_cols, quad_cols..(quad_cols * 2));

        let top_left = mtx.slice_range(&top_rows, &left_cols)?;
        let bottom_left = mtx.slice_range(&bottom_rows, &left_cols)?;
        let top_right = mtx.slice_range(&top_rows, &right_cols)?;
        let bottom_right = mtx.slice_range(&bottom_rows, &right_cols)?;

        Ok(SplitQuad {
            top_left,
            bottom_left,
            top_right,
            bottom_right,
        })
    }
}

impl<'a, T> From<&'a Matrix<T>> for SliceMatrix<'a, T> {
    fn from(parent: &'a Matrix<T>) -> Self {
        SliceMatrix {
            parent,
            rows: 0..parent.num_rows(),
            cols: 0..parent.num_cols(),
        }
    }
}

impl<T> Display for SliceMatrix<'_, T>
where
    T: AddAssign + Default + Copy + Debug + 'static,
    for<'b> &'b T: Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "rows: {:?}, cols: {:?}", self.rows, self.cols)?;
        for row_ind in self.rows.clone() {
            writeln!(f, "{:?}", &self.parent.inner[row_ind][self.cols.clone()])?;
        }
        Ok(())
    }
}
