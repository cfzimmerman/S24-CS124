use crate::error::{PsetErr, PsetRes};
use std::{
    fmt::{self, Debug, Display},
    ops::{Add, AddAssign, Mul, Range, Sub},
};

/// A matrix type that owns its own data.
#[derive(Debug)]
pub struct Matrix<T> {
    inner: Vec<Vec<T>>,
    /// This is for convenience when a function needs to
    /// return a reference that sometimes must be the default value.
    default_val: T,
    /// The number of actually-existing rows and columns of data that came
    /// from slice padding.
    padded_rows: usize,
    padded_cols: usize,
}

// TODO:
// What to do about quad blocks that are entirely padding? How to gracefully
// drop them and avoid that computation?
// When a split has failed:
// - Any multiplication is zero
// - Any addition is identity
//
// Begin by making SplitQuad return Error variants

impl<T> Matrix<T>
where
    T: AddAssign + Default + Copy + Debug + 'static,
    for<'a> &'a T: Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    /// Returns a reference to the inner data of the matrix
    pub fn inspect(&self) -> &Vec<Vec<T>> {
        &self.inner
    }

    /// Consumes the matrix, returning its inner data
    pub fn take(self) -> Vec<Vec<T>> {
        self.inner
    }

    /// Returns a matrix representing the operation left + right
    pub fn add(left: &Matrix<T>, right: &Matrix<T>) -> PsetRes<Matrix<T>> {
        SliceMatrix::add(&left.into(), &right.into())
    }

    /// Returns an error if entries do not have equal width and height
    /// dimensions. Must be Ok() for operations like addition and subtraction.
    /// Note that Matrix does not have any notion of padding.
    fn check_equal_dims(left: &Matrix<T>, right: &Matrix<T>) -> PsetRes<()> {
        if left.num_rows() != right.num_rows() || left.num_cols() != right.num_cols() {
            return Err(PsetErr::Static(
                "Cannot perform a one to one operation on matrices of different dimensions",
            ));
        }
        Ok(())
    }

    /// Adds `other` to `self`. Returns reference for method chaining.
    fn add_in_place(&mut self, other: &Matrix<T>) -> PsetRes<&mut Self> {
        self.op_one_to_one_in_place(other, |l, r| l + r)?;
        Ok(self)
    }

    /// Subtracts `other` from `self`. Performs
    /// (self entry) - (other entry) at every index.
    /// Returns reference for method chaining.
    fn sub_in_place(&mut self, other: &Matrix<T>) -> PsetRes<&mut Self> {
        self.op_one_to_one_in_place(other, |l, r| l - r)?;
        Ok(self)
    }

    /// Performs actions for every parallel entry between the two matrices, storing
    /// the results of operations back in self.
    fn op_one_to_one_in_place(&mut self, other: &Matrix<T>, op: fn(&T, &T) -> T) -> PsetRes<()> {
        Matrix::check_equal_dims(self, other)?;
        for row in 0..self.inner.len() {
            for col in 0..self.inner[0].len() {
                // indexing won't panic because of check_equal_dims
                self.inner[row][col] = op(&self.inner[row][col], &other.inner[row][col]);
            }
        }
        self.padded_rows = self.padded_rows.min(other.padded_rows);
        self.padded_cols = self.padded_cols.min(other.padded_cols);
        Ok(())
    }

    /// Returns a matrix representing the operation left - right
    pub fn sub(left: &Matrix<T>, right: &Matrix<T>) -> PsetRes<Matrix<T>> {
        SliceMatrix::sub(&left.into(), &right.into())
    }

    /// Performs O(n^3) iterative multiplication on the given matrices.
    pub fn mul_iter(left: &Matrix<T>, right: &Matrix<T>) -> PsetRes<Matrix<T>> {
        SliceMatrix::mul_iter(&left.into(), &right.into())
    }

    /// Performs O(n^3) recursive multiplication on the given matrices.
    pub fn mul_naive_rec(
        left: &Matrix<T>,
        right: &Matrix<T>,
        base_cutoff: usize,
    ) -> PsetRes<Matrix<T>> {
        if base_cutoff < 3 {
            return Err(PsetErr::Static("mul_naive_rec base_cutoff must exceed 2"));
        }
        let mut left_sl: SliceMatrix<T> = left.into();
        let mut right_sl: SliceMatrix<T> = right.into();
        left_sl.pad_even_square();
        right_sl.pad_even_square();

        let mut res = SliceMatrix::mul_naive_rec(&left_sl, &right_sl, base_cutoff)?;
        res.trim_dims(left.num_rows(), right.num_cols());
        Ok(res)
    }

    /*
    pub fn mul_strassen(
        left: &Matrix<T>,
        right: &Matrix<T>,
        base_cutoff: usize,
    ) -> PsetRes<Matrix<T>> {
        if base_cutoff < 3 {
            return Err(PsetErr::Static("mul_naive_rec base_cutoff must exceed 2"));
        }
        let mut left_sl: SliceMatrix<T> = left.into();
        let mut right_sl: SliceMatrix<T> = right.into();
        left_sl.pad_even_square();
        right_sl.pad_even_square();

        let mut res = SliceMatrix::mul_strassen(&left_sl, &right_sl, base_cutoff)?;
        res.trim_dims(left.num_rows(), right.num_cols());
        Ok(res)
    }
    */

    /// Builds an identity matrix
    /// For a usize matrix, diagon_val is 1, the value on the diagonal.
    pub fn identity(dim: usize, diagon_val: T) -> Matrix<T> {
        (0..dim)
            .into_iter()
            .map(|row_ind| {
                let mut cols = vec![T::default(); dim];
                cols[row_ind] = diagon_val;
                cols
            })
            .collect::<Vec<Vec<T>>>()
            .into()
    }

    /// Builds an owned matrix from slice parts. Fails unless all given parts
    /// are of equal dimensions (including padding).
    /// Merged output does not include padding.
    /*
    pub fn from_slices(
        top_left: &SliceMatrix<T>,
        bottom_left: &SliceMatrix<T>,
        top_right: &SliceMatrix<T>,
        bottom_right: &SliceMatrix<T>,
    ) -> PsetRes<Matrix<T>> {
        let mut inner: Vec<Vec<T>> = Vec::with_capacity(top_left.num_rows() * 2);
        // Do not merge padding.
        // Add the top half, then the bottom half
        for (left_side, right_side) in &[(top_left, top_right), (bottom_left, bottom_right)] {
            debug_assert!(left_side.row_pad_sz == right_side.row_pad_sz);
            for row_offset in 0..(left_side.num_rows() - left_side.row_pad_sz) {
                let mut row = left_side.parent.inner[left_side.rows.start + row_offset]
                    [left_side.cols.clone()]
                .to_vec();
                row.extend_from_slice(
                    &right_side.parent.inner[right_side.rows.start + row_offset]
                        [right_side.cols.clone()],
                );
                inner.push(row);
            }
        }
        Ok(inner.into())
    }
    */
    /// Assumes self is the top left in a four-part matrix. Adds neighbors into self
    /// in their named positions.
    /// Requires that no matrices involved contain padding.
    fn fill_neighbors(
        &mut self,
        bottom_left: Option<Matrix<T>>,
        top_right: Option<Matrix<T>>,
        bottom_right: Option<Matrix<T>>,
    ) -> PsetRes<()> {
        for mem in [
            Some(&*self),
            bottom_left.as_ref(),
            top_right.as_ref(),
            bottom_right.as_ref(),
        ] {
            if let Some(mtx) = mem {
                if mtx.padded_rows + mtx.padded_cols > 0 {
                    return Err(PsetErr::Static(
                        "Cannot fill neighbors that currently have padding",
                    ));
                }
            }
        }
        if let Some(bl) = bottom_left {
            self.inner.extend(bl.inner);
        };
        match (top_right, bottom_right) {
            (Some(tr), Some(br)) => {
                for (left_row, right_row) in self
                    .inner
                    .iter_mut()
                    .zip(tr.inner.into_iter().chain(br.inner.into_iter()))
                {
                    left_row.extend(right_row);
                }
            }
            (Some(tr), None) => {
                for (left_row, right_row) in self.inner.iter_mut().zip(tr.inner.into_iter()) {
                    left_row.extend(right_row);
                }
            }
            (None, Some(_)) => {
                return Err(PsetErr::Static(
                    "There cannot be a bottom right part without a top right part",
                ));
            }
            (None, None) => (),
        };
        Ok(())
    }

    /// Returns the number of rows in this matrix
    fn num_rows(&self) -> usize {
        self.inner.len()
    }

    /// Returns the number of columns in this matrix
    fn num_cols(&self) -> usize {
        self.inner.get(0).map(|arr| arr.len()).unwrap_or(0)
    }

    /// Removes entries from the bottom and right sides until self reaches the
    /// final dimensions given.
    fn trim_dims(&mut self, final_num_rows: usize, final_num_cols: usize) {
        self.inner.truncate(final_num_rows);
        if self.inner.get(0).map(|row| row.len()).unwrap_or(0) > final_num_cols {
            for row in &mut self.inner {
                row.truncate(final_num_cols);
            }
        }
    }

    /// Trims matrix dimensions to remove internally-tracked padding
    fn trim_padding(&mut self) {
        self.trim_dims(
            self.num_rows() - self.padded_rows,
            self.num_cols() - self.padded_cols,
        );
        self.padded_rows = 0;
        self.padded_cols = 0;
    }
}

/// A flexible window over data held in some other Matrix.
/// Rows and columns index into the parent matrix, and padding
/// implies rows and columns of zero on the bottom and right
/// respectively.
/// Use Display formatting for a more attractive printing
/// of the sliced matrix.
#[derive(Debug, Clone)]
pub struct SliceMatrix<'a, T> {
    parent: &'a Matrix<T>,
    rows: Range<usize>,
    cols: Range<usize>,
    /// The number of rows of zeroes assumed to be at
    /// the bottom of this matrix's range.
    row_pad_sz: usize,
    /// The number of columns of zeroes assumed to be
    /// to the right of this matrix's range.
    col_pad_sz: usize,
}

#[derive(Debug)]
struct SplitQuad<'a, T> {
    top_left: Option<SliceMatrix<'a, T>>,
    bottom_left: Option<SliceMatrix<'a, T>>,
    top_right: Option<SliceMatrix<'a, T>>,
    bottom_right: Option<SliceMatrix<'a, T>>,
}

impl<'a, T> SplitQuad<'a, T>
where
    T: AddAssign + Default + Copy + Debug + 'static,
    for<'b> &'b T: Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    /// Returns that matrix partitioned into four equal quadrants.
    /// Any quadrants that fail partitioning are None.
    /// An all-padding quadrant is returned as None.
    /// Fails if the given parent slice is not at least 2x2, square, and even.
    fn build(mtx: &'a SliceMatrix<'a, T>) -> PsetRes<Self> {
        if mtx.num_rows() < 2 || mtx.num_cols() < 2 {
            return Err(PsetErr::Static(
                "Matrix must be at least 2x2 to split as quad.",
            ));
        }
        if mtx.num_rows() != mtx.num_cols() || mtx.num_rows() % 2 != 0 {
            return Err(PsetErr::Static(
                "SplitQuad requires an even square matrix. Consider padding it first.",
            ));
        }
        let quad_height = mtx.num_rows() / 2;
        let quad_width = mtx.num_cols() / 2;

        let top_left = mtx.slice_range(0..quad_height, 0..quad_width);
        let mut bottom_right = top_left.clone().and_then(|tl| {
            mtx.slice_range(
                tl.rows.end..(tl.rows.end + quad_width),
                tl.cols.end..(tl.cols.end + quad_height),
            )
        });
        let mut bottom_left = None;
        if let (Some(tl), Some(br)) = (&top_left, &bottom_right) {
            bottom_left = mtx.slice_range(br.rows.clone(), tl.cols.clone());
        };
        let mut top_right = None;
        if let (Some(tl), Some(br)) = (&top_left, &bottom_right) {
            top_right = mtx.slice_range(tl.rows.clone(), br.cols.clone());
        }

        // inherit parent padding
        if let Some(br) = &mut bottom_right {
            br.row_pad_sz = mtx.row_pad_sz;
            br.col_pad_sz = mtx.col_pad_sz;
        }
        if let Some(bl) = &mut bottom_left {
            bl.row_pad_sz = mtx.row_pad_sz;
        }
        if let Some(tr) = &mut top_right {
            tr.col_pad_sz = mtx.col_pad_sz;
        }

        Ok(SplitQuad {
            top_left,
            bottom_left,
            top_right,
            bottom_right,
        })
    }
}

impl<'a, T> SliceMatrix<'a, T>
where
    T: AddAssign + Default + Copy + Debug + 'static,
    for<'b> &'b T: Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    /// Returns the number of rows in this matrix WITH PADDING
    fn num_rows(&self) -> usize {
        debug_assert!(
            self.rows.end <= self.parent.num_rows(),
            "Slice matrix row range should always be in bounds"
        );
        self.rows.end + self.row_pad_sz - self.rows.start
    }

    /// Returns the number of columns in this matrix WITH PADDING
    fn num_cols(&self) -> usize {
        debug_assert!(
            self.cols.end <= self.parent.num_cols(),
            "Slice matrix col range should always be in bounds"
        );
        self.cols.end + self.col_pad_sz - self.cols.start
    }

    /// Retrieves an entry in the matrix or none if out of bounds.
    /// INDICES ARE IN TERMS OF THE WINDOW and begin at 0.
    /// Returned values may also reflect padding.
    fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row >= self.num_rows() || col >= self.num_cols() {
            return None;
        }
        if self.rows.end <= self.rows.start + row || self.cols.end <= self.cols.start + col {
            // in padding, case above guarantees in bounds
            return Some(&self.parent.default_val);
        }
        self.parent
            .inner
            .get(self.rows.start + row)
            .and_then(|r| r.get(self.cols.start + col))
    }

    /// If a matrix is not a square with even-length sides, pads it with
    /// zeroes until it is one
    fn pad_even_square(&mut self) {
        self.row_pad_sz = 0;
        self.col_pad_sz = 0;
        if self.num_rows() % 2 != 0 {
            self.row_pad_sz += 1;
        }
        if self.num_cols() % 2 != 0 {
            self.col_pad_sz += 1;
        }
        if self.num_rows() < self.num_cols() {
            self.row_pad_sz += self.num_cols() - self.num_rows();
        }
        if self.num_cols() < self.num_rows() {
            self.col_pad_sz += self.num_rows() - self.num_cols();
        }
        debug_assert!(self.num_rows() == self.num_cols());
        debug_assert!((self.num_rows() + self.num_cols()) % 2 == 0);
    }

    /// If successful, returns the slice matrix partitioned into four equal
    /// quadrants.
    /// Fails unless the given matrix is an even-dimension square. The method
    /// `pad_even_square` can be helpful getting there.
    fn try_split_quad<'b>(&'b self) -> PsetRes<SplitQuad<'b, T>> {
        SplitQuad::build(self)
    }

    /// Attempts to build a new slice matrix from a range within
    /// thwe existing one. The result may have padding.
    /// RANGES ARE IN RELATION TO THE PARENT SLICE, NOT THE PARENT MATRIX.
    pub fn slice_range(
        &self,
        req_rows: Range<usize>,
        req_cols: Range<usize>,
    ) -> Option<SliceMatrix<T>> {
        let req_row_len = req_rows.end - req_rows.start;
        let req_col_len = req_cols.end - req_cols.start;

        if self.num_rows() < req_row_len || self.num_cols() < req_col_len {
            return None;
        }

        let sl_row_start = self.rows.start + req_rows.start;
        let sl_col_start = self.cols.start + req_cols.start;

        if self.rows.end <= sl_row_start || self.cols.end <= sl_col_start {
            return None;
        }

        let sl_rows = sl_row_start..(sl_row_start + req_row_len).min(self.rows.end);
        let sl_cols = sl_col_start..(sl_col_start + req_col_len).min(self.cols.end);

        let row_pad_sz = req_row_len.saturating_sub(sl_rows.end - sl_rows.start);
        let col_pad_sz = req_col_len.saturating_sub(sl_cols.end - sl_cols.start);
        Some(SliceMatrix {
            parent: self.parent,
            rows: sl_rows,
            cols: sl_cols,
            row_pad_sz,
            col_pad_sz,
        })
    }

    /// Returns a matrix representing the operation left + right
    fn add<'b>(left: &SliceMatrix<'b, T>, right: &SliceMatrix<'b, T>) -> PsetRes<Matrix<T>> {
        Self::op_one_to_one(left, right, |l, r| l + r)
    }

    /// Returns a matrix representing the operation left - right
    fn sub<'b>(left: &SliceMatrix<'b, T>, right: &SliceMatrix<'b, T>) -> PsetRes<Matrix<T>> {
        Self::op_one_to_one(left, right, |l, r| l - r)
    }

    /// Performs O(n^3) iterative multiplication on the given matrices.
    /// If the input matrices are padded, the output matrix is computed
    /// without padding.
    fn mul_iter<'b>(left: &SliceMatrix<'b, T>, right: &SliceMatrix<'b, T>) -> PsetRes<Matrix<T>> {
        if left.num_cols() != right.num_rows() {
            return Err(PsetErr::Static("Matrix dims don't support right multiply"));
        }
        let mut res: Vec<Vec<T>> = Vec::with_capacity(left.num_rows());
        for left_row_off in 0..left.num_rows() {
            let mut row: Vec<T> = Vec::with_capacity(right.num_rows());
            for right_col_off in 0..right.num_cols() {
                let mut sum = T::default();
                for offset in 0..left.num_cols() {
                    let left_val = left
                        .get(left_row_off, offset)
                        .expect("Padded slice should be in bounds");
                    let right_val = right
                        .get(offset, right_col_off)
                        .expect("Padded slice should be in bounds");
                    sum += left_val * right_val;
                }
                row.push(sum);
            }
            res.push(row);
        }
        Ok(Matrix {
            inner: res,
            default_val: T::default(),
            padded_rows: left.row_pad_sz,
            padded_cols: right.col_pad_sz,
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
                let left_val = left
                    .parent
                    .inner
                    .get(left.rows.start + row_offset)
                    .and_then(|row| row.get(left.cols.start + col_offset))
                    .unwrap_or_else(|| &left.parent.default_val);
                let right_val = right
                    .parent
                    .inner
                    .get(right.rows.start + row_offset)
                    .and_then(|row| row.get(right.cols.start + col_offset))
                    .unwrap_or_else(|| &right.parent.default_val);
                row.push(op(left_val, right_val));
            }
            res.push(row);
        }
        Ok(Matrix {
            inner: res,
            default_val: T::default(),
            padded_rows: left.row_pad_sz.min(right.row_pad_sz),
            padded_cols: left.col_pad_sz.min(right.col_pad_sz),
        })
    }

    /// Performs O(n^3) recursive multiplication. Used as a stepping stone before
    /// Strassen's algorithm.
    fn mul_naive_rec<'b>(
        left: &SliceMatrix<'b, T>,
        right: &SliceMatrix<'b, T>,
        base_sz: usize,
    ) -> PsetRes<Matrix<T>> {
        if left.num_cols() <= base_sz || left.num_rows() <= base_sz {
            return Self::mul_iter(&left, &right);
        }

        // SliceMatrix is just pointers and counters, so cloning is cheap.
        let mut left_quad = left.try_split_quad()?;
        let mut right_quad = right.try_split_quad()?;

        for quad in [&mut left_quad, &mut right_quad] {
            // expands into padding all eight parts in question into
            // even squares.
            for part in [
                &mut quad.top_left,
                &mut quad.top_right,
                &mut quad.bottom_left,
                &mut quad.bottom_right,
            ] {
                if let Some(sl) = part {
                    sl.pad_even_square();
                }
            }
        }

        let mut tl1 = None;
        let mut tl2 = None;

        let mut bl1 = None;
        let mut bl2 = None;

        let mut tr1 = None;
        let mut tr2 = None;

        let mut br1 = None;
        let mut br2 = None;

        if let (Some(ltl), Some(rtl)) = (&left_quad.top_left, &right_quad.top_left) {
            tl1 = Some(Self::mul_naive_rec(ltl, rtl, base_sz)?);
        };
        if let (Some(ltr), Some(rbl)) = (&left_quad.top_right, &right_quad.bottom_left) {
            tl2 = Some(Self::mul_naive_rec(ltr, rbl, base_sz)?);
        }

        if let (Some(lbl), Some(rtl)) = (&left_quad.bottom_left, &right_quad.top_left) {
            bl1 = Some(Self::mul_naive_rec(lbl, rtl, base_sz)?);
        };
        if let (Some(lbr), Some(rbl)) = (&left_quad.bottom_right, &right_quad.bottom_left) {
            bl2 = Some(Self::mul_naive_rec(lbr, rbl, base_sz)?);
        };

        if let (Some(ltl), Some(rtr)) = (&left_quad.top_left, &right_quad.top_right) {
            tr1 = Some(Self::mul_naive_rec(ltl, rtr, base_sz)?);
        };
        if let (Some(ltr), Some(rbr)) = (&left_quad.top_right, &right_quad.bottom_right) {
            tr2 = Some(Self::mul_naive_rec(ltr, rbr, base_sz)?);
        };

        if let (Some(lbl), Some(rtr)) = (&left_quad.bottom_left, &right_quad.top_right) {
            br1 = Some(Self::mul_naive_rec(lbl, rtr, base_sz)?);
        };
        if let (Some(lbr), Some(rbr)) = (&left_quad.bottom_right, &right_quad.bottom_right) {
            br2 = Some(Self::mul_naive_rec(lbr, rbr, base_sz)?);
        };

        if left.rows.clone().eq(0..3)
            && left.cols.clone().eq(3..5)
            && right.rows.clone().eq(3..5)
            && right.cols.clone().eq(0..3)
        {
            println!("left: \n{left}");
            println!("right: \n{right}");

            if let Some(mtx) = &tr1 {
                println!("tr1: \n{mtx}");
            }
            if let Some(mtx) = &tr2 {
                println!("tr2: \n{mtx}");
            }
        }

        let parts = [(tl1, tl2), (bl1, bl2), (tr1, tr2), (br1, br2)]
            .into_iter()
            .map(|pair| match pair {
                (Some(mut m1), Some(m2)) => {
                    m1.add_in_place(&m2)?.trim_padding();
                    Ok(Some(m1))
                }
                (Some(mut mtx), None) | (None, Some(mut mtx)) => {
                    mtx.trim_padding();
                    Ok(Some(mtx))
                }
                (None, None) => Ok(None),
            })
            .collect::<PsetRes<Vec<Option<Matrix<T>>>>>()?;
        let mut parts_iter = parts.into_iter();
        let top_left = parts_iter.next().flatten();
        let bottom_left = parts_iter.next().flatten();
        let top_right = parts_iter.next().flatten();
        let bottom_right = parts_iter.next().flatten();

        let Some(mut mtx) = top_left else {
            return Err(PsetErr::Static(
                "Matrix prod with empty top left is invalid",
            ));
        };
        mtx.fill_neighbors(bottom_left, top_right, bottom_right)?;
        Ok(mtx)
    }

    /*
        fn mul_strassen<'b>(
            left: &SliceMatrix<'b, T>,
            right: &SliceMatrix<'b, T>,
            base_sz: usize,
        ) -> PsetRes<Matrix<T>> {
            if left.num_cols() <= base_sz || left.num_rows() <= base_sz {
                return Self::mul_iter(&left, &right);
            }

            // SliceMatrix is just pointers and counters, so cloning is cheap.
            let mut left_quad = left.try_split_quad()?;
            let mut right_quad = right.try_split_quad()?;

            for quad in [&mut left_quad, &mut right_quad] {
                quad.top_left.pad_even_square();
                quad.top_right.pad_even_square();
                quad.bottom_left.pad_even_square();
                quad.bottom_right.pad_even_square();
            }

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
            final_top_left.fill_neighbors(final_bottom_left, final_top_right, final_bottom_right)?;
            Ok(final_top_left)
            /*
            Matrix::from_slices(
                &(&final_top_left).into(),
                &(&final_bottom_left).into(),
                &(&final_top_right).into(),
                &(&final_bottom_right).into(),
            )
            */
        }
    */
}

impl<T> From<Vec<Vec<T>>> for Matrix<T>
where
    T: Default,
{
    fn from(item: Vec<Vec<T>>) -> Self {
        Matrix {
            inner: item,
            default_val: T::default(),
            padded_rows: 0,
            padded_cols: 0,
        }
    }
}

impl<'a, T> From<&'a Matrix<T>> for SliceMatrix<'a, T> {
    fn from(parent: &'a Matrix<T>) -> Self {
        SliceMatrix {
            parent,
            rows: 0..parent.inner.len(),
            cols: 0..parent.inner.get(0).map(|row| row.len()).unwrap_or(0),
            row_pad_sz: 0,
            col_pad_sz: 0,
        }
    }
}

impl<T> Display for SliceMatrix<'_, T>
where
    T: AddAssign + Default + Copy + Debug + 'static,
    for<'b> &'b T: Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "rows: {:?}, cols: {:?}, row_pad: {:?}, col_pad: {:?}",
            self.rows, self.cols, self.row_pad_sz, self.col_pad_sz
        )?;
        for row_ind in self.rows.clone() {
            let mut row = self.parent.inner[row_ind][self.cols.clone()].to_vec();
            for _ in 0..self.col_pad_sz {
                row.push(T::default());
            }
            writeln!(f, "{:?}", row)?;
        }
        for _ in 0..self.row_pad_sz {
            writeln!(f, "{:?}", vec![T::default(); self.num_rows()])?;
        }
        Ok(())
    }
}

impl<T> Display for Matrix<T>
where
    T: AddAssign + Default + Copy + Debug + 'static,
    for<'b> &'b T: Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let as_sl: SliceMatrix<T> = self.into();
        Display::fmt(&as_sl, f)?;
        Ok(())
    }
}

#[cfg(test)]
mod matrix_tests {
    use crate::{
        error::PsetRes,
        matrix::{Matrix, SliceMatrix},
        test_data::{
            get_asymm_test_matrices, get_square_test_matrices, get_test_4x4, get_test_6x5,
        },
    };

    /// Tests matrix addition and subtraction
    #[test]
    fn add_sub_square() -> PsetRes<()> {
        for matrices in get_square_test_matrices()? {
            let mut left: Matrix<i64> = matrices.left.into();
            let right: Matrix<i64> = matrices.right.into();
            assert_eq!(
                Matrix::add(&left, &right)?.take(),
                matrices.sum,
                "Sum should be equal"
            );
            assert_eq!(
                Matrix::sub(&left, &right)?.take(),
                matrices.diff,
                "Difference should be equal"
            );
            left.add_in_place(&right)?;
            assert_eq!(
                left.inner, matrices.sum,
                "Sum should be equal after add in place"
            );
            left.sub_in_place(&right)?.sub_in_place(&right)?;
            // ^ first undoes the addition, second performs sub
            assert_eq!(
                left.inner, matrices.diff,
                "Difference should be equal in place"
            );
        }
        Ok(())
    }

    /// Tests iterative matrix multiplication on square matrices
    #[test]
    fn mul_square() -> PsetRes<()> {
        for matrices in get_square_test_matrices()? {
            let left: Matrix<i64> = matrices.left.into();
            let right: Matrix<i64> = matrices.right.into();
            assert_eq!(
                Matrix::mul_iter(&left, &right)?.inner,
                matrices.prod,
                "Iter product should be equal"
            );
            assert_eq!(
                Matrix::mul_naive_rec(&left, &right, 3)?.inner,
                matrices.prod,
                "Naive recursive product should be equal"
            );
            /*
            assert_eq!(
                Matrix::mul_strassen(&left, &right, 3)?.inner,
                matrices.prod,
                "Strassen product should be equal"
            );
            */
        }
        Ok(())
    }

    /// Verifies the correctness of multiplication under padding. (Just checks
    /// iterative).
    #[test]
    fn mul_padded() -> PsetRes<()> {
        for matrices in get_asymm_test_matrices()? {
            let left: Matrix<i64> = matrices.left.into();
            let right: Matrix<i64> = matrices.right.into();

            // Padding should not affect matrix multiplication. It only
            // affects matrix segmentation.
            let mut left_sq: SliceMatrix<'_, i64> = (&left).into();
            left_sq.pad_even_square();
            let mut right_sq: SliceMatrix<'_, i64> = (&right).into();
            right_sq.pad_even_square();
            if left_sq.num_rows() != right_sq.num_rows() {
                continue;
            }

            let mut with_padding = SliceMatrix::mul_iter(&left_sq, &right_sq)?;
            with_padding.trim_dims(matrices.prod.len(), matrices.prod[0].len());
            assert_eq!(
                with_padding.take(),
                matrices.prod,
                "Padded iter product should also be equal"
            );
        }
        Ok(())
    }

    /// Tests iterative matrix multiplication on non-square matrices
    #[test]
    fn mul_asymm() -> PsetRes<()> {
        for matrices in get_asymm_test_matrices()? {
            let left: Matrix<i64> = matrices.left.into();
            let right: Matrix<i64> = matrices.right.into();
            assert_eq!(
                Matrix::mul_iter(&left, &right)?.take(),
                matrices.prod,
                "Iter product should be equal"
            );

            for base_cutoff in [3, 10, 16] {
                assert_eq!(
                    Matrix::mul_naive_rec(&left, &right, base_cutoff)?.take(),
                    matrices.prod,
                    "Naive recursive product should be equal"
                );
                /*
                assert_eq!(
                    Matrix::mul_strassen(&left, &right, base_cutoff)?.inner,
                    matrices.prod,
                    "Strassen product should be equal"
                );
                */
            }

            // Padding should not affect matrix multiplication. It only
            // affects matrix segmentation.
            let mut left_sq: SliceMatrix<'_, i64> = (&left).into();
            left_sq.pad_even_square();
            let mut right_sq: SliceMatrix<'_, i64> = (&right).into();
            right_sq.pad_even_square();
            if left_sq.num_rows() != right_sq.num_rows() {
                continue;
            }
            let mut padded = SliceMatrix::mul_iter(&left_sq, &right_sq)?;
            padded.trim_dims(matrices.prod.len(), matrices.prod[0].len());
            assert_eq!(
                padded.take(),
                matrices.prod,
                "Padded iter product should also be equal"
            );
        }
        Ok(())
    }

    #[test]
    fn split_merge_quad() -> PsetRes<()> {
        let iden: Matrix<i64> = Matrix::identity(3, 1);

        let joined = {
            // construct the 3x3 identity matrix from pieces
            let mut top_left: Matrix<i64> = vec![vec![1, 0], vec![0, 1]].into();
            let bottom_left: Matrix<i64> = vec![vec![0, 0]].into();
            let top_right: Matrix<i64> = vec![vec![0], vec![0]].into();
            let bottom_right: Matrix<i64> = vec![vec![1]].into();
            top_left.fill_neighbors(Some(bottom_left), Some(top_right), Some(bottom_right))?;
            top_left
        };

        assert_eq!(
            iden.inner, joined.inner,
            "Matrix should be returned to identity form"
        );
        Ok(())
    }

    #[test]
    fn static_4x4_mul() -> PsetRes<()> {
        let inputs = get_test_4x4()?;
        let left_mtx: Matrix<i64> = inputs.left.into();
        let right_mtx: Matrix<i64> = inputs.right.into();

        let iter_prod = Matrix::mul_iter(&left_mtx, &right_mtx)?;
        let naive_rec_prod = Matrix::mul_naive_rec(&left_mtx, &right_mtx, 3)?;
        // let strassen_prod = Matrix::mul_strassen(&left_mtx, &right_mtx, 3)?;

        assert_eq!(iter_prod.inner, inputs.prod, "Iter should equal input prod");
        assert_eq!(
            naive_rec_prod.inner, inputs.prod,
            "Naive rec should equal input prod"
        );
        /*
        assert_eq!(
            strassen_prod.inner, inputs.prod,
            "Strassen should equal input prod"
        );
        */

        Ok(())
    }

    #[test]
    fn static_6x5_mul() -> PsetRes<()> {
        let inputs = get_test_6x5()?;
        let left_mtx: Matrix<i64> = inputs.left.into();
        let right_mtx: Matrix<i64> = inputs.right.into();

        let iter_prod = Matrix::mul_iter(&left_mtx, &right_mtx)?;
        let naive_rec_prod = Matrix::mul_naive_rec(&left_mtx, &right_mtx, 3)?;
        // let strassen_prod = Matrix::mul_strassen(&left_mtx, &right_mtx, 3)?;

        assert_eq!(iter_prod.inner, inputs.prod, "Iter should equal input prod");
        println!("target: {iter_prod}");
        println!("rec prod: {naive_rec_prod}");
        assert_eq!(
            naive_rec_prod.inner, inputs.prod,
            "Naive rec should equal input prod"
        );
        /*
        assert_eq!(
            strassen_prod.inner, inputs.prod,
            "Strassen should equal input prod"
        );
        */

        Ok(())
    }
}
