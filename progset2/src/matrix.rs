use crate::error::{PsetErr, PsetRes};
use std::{
    fmt::{self, Debug, Display},
    ops::{Add, AddAssign, Mul, Range, Sub},
};

/// A matrix type that owns its own data.
#[derive(Debug, Clone)]
pub struct Matrix<T> {
    inner: Vec<Vec<T>>,
}

impl<T> Matrix<T>
where
    T: AddAssign + Default + Copy + Debug + 'static,
    for<'a> &'a T: Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    /// Returns a matrix representing the operation left + right
    pub fn add(left: &Matrix<T>, right: &Matrix<T>) -> PsetRes<Matrix<T>> {
        SliceMatrix::add(&left.into(), &right.into())
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
    ///
    /// Mutable references are required to apply padding before recursion. However,
    /// it is a guarantee that padding is removed and left and right have the same
    /// initialized values upon exit as enter (although their capacities likely change).
    pub fn mul_naive_rec(
        left: &mut Matrix<T>,
        right: &mut Matrix<T>,
        base_cutoff: usize,
    ) -> PsetRes<Matrix<T>> {
        if base_cutoff < 3 {
            return Err(PsetErr::Static("mul_naive_rec base_cutoff must exceed 2"));
        }

        let (l_init_rows, l_init_cols) = (left.num_rows(), left.num_cols());
        let (r_init_rows, r_init_cols) = (right.num_rows(), right.num_cols());
        left.pad_pow2();
        right.pad_pow2();

        let left_sl: SliceMatrix<T> = (&*left).into();
        let right_sl: SliceMatrix<T> = (&*right).into();
        let mut res = SliceMatrix::mul_naive_rec(&left_sl, &right_sl, base_cutoff)?;

        left.trim_dims(l_init_rows, l_init_cols);
        right.trim_dims(r_init_rows, r_init_cols);
        res.trim_dims(l_init_rows, r_init_cols);

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

    /// Returns the number of rows in this matrix
    fn num_rows(&self) -> usize {
        self.inner.len()
    }

    /// Returns the number of columns in this matrix
    fn num_cols(&self) -> usize {
        self.inner.get(0).map(|arr| arr.len()).unwrap_or(0)
    }

    /// Performs actions for every parallel entry between the two matrices, storing
    /// the results of operations back in self.
    fn op_one_to_one_in_place(&mut self, other: &Matrix<T>, op: fn(&T, &T) -> T) -> PsetRes<()> {
        if self.num_rows() != other.num_rows() || self.num_cols() != other.num_cols() {
            return Err(PsetErr::Static(
                "Cannot perform a one to one operation on matrices of different dimensions",
            ));
        }
        for row in 0..self.inner.len() {
            for col in 0..self.inner[0].len() {
                // indexing won't panic because dimensions are checked above
                self.inner[row][col] = op(&self.inner[row][col], &other.inner[row][col]);
            }
        }
        Ok(())
    }

    /// Pads self with zeroes to the bottom and right so that its dimensions
    /// are a power of 2.
    fn pad_pow2(&mut self) {
        fn round_up_nearest_pow2(num: usize) -> u32 {
            (num as f64).log2().ceil() as u32
        }
        let max_dim = self.num_rows().max(self.num_cols());
        let add_rows = 2usize.pow(round_up_nearest_pow2(max_dim)) - self.num_rows();
        let add_cols = 2usize.pow(round_up_nearest_pow2(max_dim)) - self.num_cols();
        for curr_row in &mut self.inner {
            curr_row.extend(vec![T::default(); add_cols]);
            debug_assert!((curr_row.len() as f64).log2().fract() == 0.);
        }
        for _ in 0..add_rows {
            self.inner
                .push(vec![T::default(); self.num_cols() + add_cols]);
        }
        debug_assert!((self.inner.len() as f64).log2().fract() == 0.);
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

    /// Assumes self is the top left in a four-part matrix. Adds neighbors into self
    /// in their named positions.
    /// Fails unless neighbors have the row and column similarities to produce a
    /// complete rectangle from the constituent parts.
    fn merge_neighbors(
        &mut self,
        bottom_left: Matrix<T>,
        top_right: Matrix<T>,
        bottom_right: Matrix<T>,
    ) -> PsetRes<()> {
        if self.num_rows() != top_right.num_rows()
            || bottom_left.num_rows() != bottom_right.num_rows()
            || self.num_cols() != bottom_left.num_cols()
            || top_right.num_cols() != bottom_right.num_cols()
        {
            return Err(PsetErr::Static(
                "fill_neighbors: given matrix dimensions do not support fill",
            ));
        }
        self.inner.extend(bottom_left.inner);
        for (left_row, right_row) in self.inner.iter_mut().zip(
            top_right
                .inner
                .into_iter()
                .chain(bottom_right.inner.into_iter()),
        ) {
            left_row.extend(right_row);
        }
        Ok(())
    }
}

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
        self.rows.end - self.rows.start
    }

    /// Returns the number of columns in this matrix WITH PADDING
    fn num_cols(&self) -> usize {
        debug_assert!(
            self.cols.end <= self.parent.num_cols(),
            "Slice matrix col range should always be in bounds"
        );
        self.cols.end - self.cols.start
    }

    /// Retrieves an entry from a row, col in the slice.
    /// COORDS ARE RELATIVE TO THE SLICE.
    /// Ex, the top left coord in the slice matrix is 0, 0 regardless
    /// of where it falls on the parent matrix.
    fn get(&self, req_row: usize, req_col: usize) -> Option<&T> {
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

    fn index(&self, req_row: usize, req_col: usize) -> &T {
        self.get(req_row, req_col)
            .expect("Accessed out of bounds slice index")
    }

    /// If successful, returns the slice matrix partitioned into four equal
    /// quadrants.
    /// Fails unless the given matrix is an even-dimension square. The method
    /// `pad_even_square` can be helpful getting there.
    fn try_split_quad<'b>(&'b self) -> PsetRes<SplitQuad<'b, T>> {
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

        /*
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
        */

        tl1.add_in_place(&tl2)?;
        bl1.add_in_place(&bl2)?;
        tr1.add_in_place(&tr2)?;
        br1.add_in_place(&br2)?;
        tl1.merge_neighbors(bl1, tr1, br1)?;

        Ok(tl1)
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
        Matrix { inner: item }
    }
}

impl<'a, T> From<&'a Matrix<T>> for SliceMatrix<'a, T> {
    fn from(parent: &'a Matrix<T>) -> Self {
        SliceMatrix {
            parent,
            rows: 0..parent.inner.len(),
            cols: 0..parent.inner.get(0).map(|row| row.len()).unwrap_or(0),
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
            let row = self.parent.inner[row_ind][self.cols.clone()].to_vec();
            writeln!(f, "{:?}", row)?;
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
        matrix::Matrix,
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
                Matrix::add(&left, &right)?.inner,
                matrices.sum,
                "Sum should be equal"
            );
            assert_eq!(
                Matrix::sub(&left, &right)?.inner,
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
            let mut left: Matrix<i64> = matrices.left.into();
            let mut right: Matrix<i64> = matrices.right.into();

            let iter_prod = Matrix::mul_iter(&left, &right)?;
            let naive_rec_prod = Matrix::mul_naive_rec(&mut left, &mut right, 3)?;

            assert_eq!(
                iter_prod.inner, matrices.prod,
                "Iter product should be equal"
            );
            assert_eq!(
                naive_rec_prod.inner, matrices.prod,
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

    /// Tests iterative matrix multiplication on non-square matrices
    #[test]
    fn mul_asymm() -> PsetRes<()> {
        for matrices in get_asymm_test_matrices()? {
            let mut left: Matrix<i64> = matrices.left.into();
            let mut right: Matrix<i64> = matrices.right.into();
            assert_eq!(
                Matrix::mul_iter(&left, &right)?.inner,
                matrices.prod,
                "Iter product should be equal"
            );

            for base_cutoff in [3, 10, 16] {
                assert_eq!(
                    Matrix::mul_naive_rec(&mut left, &mut right, base_cutoff)?.inner,
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
            top_left.merge_neighbors(bottom_left, top_right, bottom_right)?;
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
        let mut left_mtx: Matrix<i64> = inputs.left.into();
        let mut right_mtx: Matrix<i64> = inputs.right.into();

        let iter_prod = Matrix::mul_iter(&left_mtx, &right_mtx)?;
        let naive_rec_prod = Matrix::mul_naive_rec(&mut left_mtx, &mut right_mtx, 3)?;
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
        let mut left_mtx: Matrix<i64> = inputs.left.into();
        let mut right_mtx: Matrix<i64> = inputs.right.into();

        let iter_prod = Matrix::mul_iter(&left_mtx, &right_mtx)?;
        let naive_rec_prod = Matrix::mul_naive_rec(&mut left_mtx, &mut right_mtx, 3)?;
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

    /// Matrix padding should yield a square with height and width
    /// dimensions as a power of 2.
    #[test]
    fn matrix_padding() -> PsetRes<()> {
        let mut m4x5 = Matrix::identity(5, 1u8);
        m4x5.inner.pop();
        // Now it's a 4 x 5 matrix
        m4x5.pad_pow2();
        assert_eq!(
            m4x5.inner.len(),
            8,
            "Matrix of dim 5 should now have 2^3 rows"
        );
        assert_eq!(
            m4x5.inner[0].len(),
            8,
            "Matrix of dim 5 should now have 2^3 cols"
        );
        Ok(())
    }
}
