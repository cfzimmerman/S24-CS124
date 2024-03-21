use crate::error::{PsetErr, PsetRes};
use std::{
    fmt::{self, Debug, Display},
    ops::{Add, AddAssign, Mul, Range, Sub},
};

/// A matrix type that owns its own data.
#[derive(Debug)]
pub struct Matrix<T> {
    inner: Vec<Vec<T>>,
}

impl<T> Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Default + Copy + Debug + 'static,
    for<'a> &'a T: Add<Output = T> + Sub<Output = T>,
{
    /// Attempts to build a new slice matrix from a specific subset of rows and
    /// columns of the parent matrix.
    pub fn slice_range(&self, rows: Range<usize>, cols: Range<usize>) -> PsetRes<SliceMatrix<T>> {
        if self.num_rows() <= rows.start
            || self.num_rows() < rows.end
            || self.num_cols() <= cols.start
            || self.num_cols() < cols.end
        {
            return Err(PsetErr::Static("Slice range is invalid"));
        }
        Ok(SliceMatrix {
            parent: self,
            rows,
            cols,
            row_pad_sz: 0,
            col_pad_sz: 0,
        })
    }

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

        SliceMatrix::mul_naive_rec(&left_sl, &right_sl, base_cutoff)
    }

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
    pub fn from_parts(
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

    /// Returns the number of rows in this matrix
    fn num_rows(&self) -> usize {
        self.inner.len()
    }

    /// Returns the number of columns in this matrix
    fn num_cols(&self) -> usize {
        self.inner.get(0).map(|arr| arr.len()).unwrap_or(0)
    }
}

/// A flexible window over data held in some other Matrix.
/// Rows and columns index into the parent matrix, and padding
/// implies rows and columns of zero on the bottom and right
/// respectively.
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
    top_left: SliceMatrix<'a, T>,
    bottom_left: SliceMatrix<'a, T>,
    top_right: SliceMatrix<'a, T>,
    bottom_right: SliceMatrix<'a, T>,
}

impl<'a, T> SplitQuad<'a, T>
where
    T: Mul<Output = T> + AddAssign + Default + Copy + Debug + 'static,
    for<'b> &'b T: Add<Output = T> + Sub<Output = T>,
{
    /// Fails unless given an even-length square matrix (padding included).
    /// Returns that matrix partitioned into four equal quadrants.
    fn build(mtx: &SliceMatrix<'a, T>) -> PsetRes<Self> {
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
        let top_left_width = mtx.num_cols() / 2;
        let top_left_height = mtx.num_rows() / 2;

        let top_left = mtx.parent.slice_range(
            mtx.rows.start..(mtx.rows.start + top_left_height),
            mtx.cols.start..(mtx.cols.start + top_left_width),
        )?;
        let mut bottom_right = mtx.parent.slice_range(
            top_left.rows.end..mtx.rows.end,
            top_left.cols.end..mtx.cols.end,
        )?;
        let mut bottom_left = mtx
            .parent
            .slice_range(bottom_right.rows.clone(), top_left.cols.clone())?;
        let mut top_right = mtx
            .parent
            .slice_range(top_left.rows.clone(), bottom_right.cols.clone())?;

        // inherit parent padding
        bottom_right.row_pad_sz = mtx.row_pad_sz;
        bottom_right.col_pad_sz = mtx.col_pad_sz;
        bottom_left.row_pad_sz = mtx.row_pad_sz;
        top_right.col_pad_sz = mtx.col_pad_sz;

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
    T: Mul<Output = T> + AddAssign + Default + Copy + Debug + 'static,
    for<'b> &'b T: Add<Output = T> + Sub<Output = T>,
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
    fn try_split_quad(&self) -> PsetRes<SplitQuad<'a, T>> {
        SplitQuad::build(self)
    }

    /*
    /// Returns a range such that iterating its indices covers every row
    /// index in the slice matrix INCLUDING IMAGINARY PADDING
    fn row_range_padded(&self) -> Range<usize> {
        self.rows.start..(self.rows.end + self.row_pad_sz)
    }

    /// Returns a range such that iterating its indices covers every column
    /// index in the slice matrix INCLUDING IMAGINARY PADDING
    fn col_range_padded(&self) -> Range<usize> {
        self.cols.start..(self.cols.end + self.col_pad_sz)
    }
    */

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
        for left_row in left.rows.clone() {
            let mut row: Vec<T> = Vec::with_capacity(right.num_rows());
            for right_col in right.cols.clone() {
                let mut sum = T::default();
                for offset in left.cols.clone() {
                    sum +=
                        left.parent.inner[left_row][offset] * right.parent.inner[offset][right_col];
                }
                row.push(sum);
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
        let mut left_quad = left.try_split_quad()?;
        let mut right_quad = right.try_split_quad()?;

        for quad in [&mut left_quad, &mut right_quad] {
            quad.top_left.pad_even_square();
            quad.top_right.pad_even_square();
            quad.bottom_left.pad_even_square();
            quad.bottom_right.pad_even_square();
        }

        let tl1 = Self::mul_naive_rec(&left_quad.top_left, &right_quad.top_left, base_sz)?;
        let tl2 = Self::mul_naive_rec(&left_quad.top_right, &right_quad.bottom_left, base_sz)?;

        let tr1 = Self::mul_naive_rec(&left_quad.top_left, &right_quad.top_right, base_sz)?;
        let tr2 = Self::mul_naive_rec(&left_quad.top_right, &right_quad.bottom_right, base_sz)?;

        let bl1 = Self::mul_naive_rec(&left_quad.bottom_left, &right_quad.top_left, base_sz)?;
        let bl2 = Self::mul_naive_rec(&left_quad.bottom_right, &right_quad.bottom_left, base_sz)?;

        let br1 = Self::mul_naive_rec(&left_quad.bottom_left, &right_quad.top_right, base_sz)?;
        let br2 = Self::mul_naive_rec(&left_quad.bottom_right, &right_quad.bottom_right, base_sz)?;

        let top_left = Self::add(&(&tl1).into(), &(&tl2).into())?;
        let top_right = Self::add(&(&tr1).into(), &(&tr2).into())?;
        let bottom_left = Self::add(&(&bl1).into(), &(&bl2).into())?;
        let bottom_right = Self::add(&(&br1).into(), &(&br2).into())?;

        Matrix::from_parts(
            &(&top_left).into(),
            &(&bottom_left).into(),
            &(&top_right).into(),
            &(&bottom_right).into(),
        )
    }

    /// Performs an operation on parallel entries in the given matrices, writing the output
    /// to a new matrix. Useful for addition and subtraction.
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
        for (l_row, r_row) in left.rows.clone().zip(right.rows.clone()) {
            let mut row = Vec::with_capacity(left.num_cols());
            for (l_col, r_col) in left.cols.clone().zip(right.cols.clone()) {
                row.push(op(
                    &left.parent.inner[l_row][l_col],
                    &right.parent.inner[r_row][r_col],
                ));
            }
            res.push(row);
        }
        Ok(res.into())
    }
}

impl<T> From<Vec<Vec<T>>> for Matrix<T> {
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
            row_pad_sz: 0,
            col_pad_sz: 0,
        }
    }
}

impl<T> Display for SliceMatrix<'_, T>
where
    T: Mul<Output = T> + AddAssign + Default + Copy + Debug + 'static,
    for<'b> &'b T: Add<Output = T> + Sub<Output = T>,
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

#[cfg(test)]
mod matrix_tests {
    use crate::{
        error::PsetRes,
        matrix::{Matrix, SliceMatrix},
        test_data::{get_asymm_test_matrices, get_square_test_matrices},
    };

    /// Tests matrix addition and subtraction
    #[test]
    fn add_sub_square() -> PsetRes<()> {
        for matrices in get_square_test_matrices()? {
            let left: Matrix<i64> = matrices.left.into();
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
                Matrix::mul_iter(&left, &right)?.take(),
                matrices.prod,
                "Iter product should be equal"
            );
            assert_eq!(
                Matrix::mul_naive_rec(&left, &right, 3)?.take(),
                matrices.prod,
                "Naive recursive product should be equal"
            );
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
            assert_eq!(
                SliceMatrix::mul_iter(&left_sq, &right_sq)?.take(),
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
            assert_eq!(
                SliceMatrix::mul_iter(&left_sq, &right_sq)?.take(),
                matrices.prod,
                "Padded iter product should also be equal"
            );
        }
        Ok(())
    }

    #[test]
    fn split_merge_quad() -> PsetRes<()> {
        let iden: Matrix<i64> = Matrix::identity(3, 1);
        let mut sliced: SliceMatrix<i64> = (&iden).into();
        sliced.pad_even_square();
        let quad = sliced.try_split_quad()?;
        let joined = Matrix::from_parts(
            &quad.top_left,
            &quad.bottom_left,
            &quad.top_right,
            &quad.bottom_right,
        )?;
        assert_eq!(
            iden.inner, joined.inner,
            "Matrix should be returned to identity form"
        );
        Ok(())
    }
}
