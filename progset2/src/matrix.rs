use crate::error::{PsetErr, PsetRes};
use std::ops::{Add, AddAssign, Mul, Range, Sub};

#[derive(Debug)]
pub struct Matrix<T> {
    inner: Vec<Vec<T>>,
}

impl<T> Matrix<T>
where
    T: Mul<Output = T> + AddAssign + Default + Copy,
    for<'a> &'a T: Add<Output = T> + Sub<Output = T>,
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

    /// Returns a matrix representing the operation left - right
    pub fn sub(left: &Matrix<T>, right: &Matrix<T>) -> PsetRes<Matrix<T>> {
        SliceMatrix::sub(&left.into(), &right.into())
    }

    /// Performs O(n^3) iterative multiplication on the given matrices.
    pub fn mul_iter(left: &Matrix<T>, right: &Matrix<T>) -> PsetRes<Matrix<T>> {
        SliceMatrix::mul_iter(&left.into(), &right.into())
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

#[derive(Debug)]
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

impl<'a, T> SplitQuad<'a, T> {
    fn build(mtx: SliceMatrix<'a, T>) -> Self {
        todo!("Given a SliceMatrix, split it into four square submatrices of equal size");
    }
}

impl<'a, T> From<SplitQuad<'a, T>> for Matrix<T> {
    fn from(value: SplitQuad<'a, T>) -> Self {
        todo!("Merge a split quad back into a matrix");
    }
}

impl<'a, T> SliceMatrix<'a, T>
where
    T: Mul<Output = T> + AddAssign + Default + Copy,
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

    /// Returns a matrix representing the operation left + right
    fn add(left: &'a SliceMatrix<T>, right: &'a SliceMatrix<T>) -> PsetRes<Matrix<T>> {
        Self::op_one_to_one(left, right, |l, r| l + r)
    }

    /// Returns a matrix representing the operation left - right
    fn sub(left: &'a SliceMatrix<T>, right: &'a SliceMatrix<T>) -> PsetRes<Matrix<T>> {
        Self::op_one_to_one(left, right, |l, r| l - r)
    }

    /// If the value is in the normal slice range, returns it. If the value is only in
    /// the padded range, returns the default value for T. (with numbers, 0)
    /// Else, returns None, index out of bounds.
    fn get_padded(&self, row: usize, col: usize) -> Option<T> {
        if self.rows.contains(&row) && self.cols.contains(&col) {
            return self
                .parent
                .inner
                .get(row)
                .and_then(|sl| sl.get(col))
                .copied();
        }
        if self.row_range_padded().contains(&row) && self.col_range_padded().contains(&col) {
            return Some(T::default());
        }
        None
    }

    /// Performs O(n^3) iterative multiplication on the given matrices.
    pub fn mul_iter(left: &'a SliceMatrix<T>, right: &'a SliceMatrix<T>) -> PsetRes<Matrix<T>> {
        if left.num_cols() != right.num_rows() {
            return Err(PsetErr::Static("Matrix dims don't support right multiply"));
        }
        let mut res: Vec<Vec<T>> = Vec::with_capacity(left.num_rows());
        for left_row in left.row_range_padded() {
            let mut row: Vec<T> = Vec::with_capacity(right.num_rows());
            for right_col in right.col_range_padded() {
                let mut sum = T::default();
                for offset in left.col_range_padded() {
                    sum += match (
                        left.get_padded(left_row, offset),
                        right.get_padded(offset, right_col),
                    ) {
                        (Some(l), Some(r)) => l * r,
                        _ => {
                            return Err(PsetErr::Static(
                                "mul_iter tried to access index out of bounds",
                            ))
                        }
                    }
                }
                row.push(sum);
            }
            res.push(row);
        }
        Ok(res.into())
    }

    /// Performs an operation on parallel entries in the given matrices, writing the output
    /// to a new matrix. Useful for addition and subtraction.
    fn op_one_to_one(
        left: &'a SliceMatrix<T>,
        right: &'a SliceMatrix<T>,
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

#[cfg(test)]
mod matrix_tests {
    use crate::{
        error::PsetRes,
        matrix::Matrix,
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
    fn mul_iter_square() -> PsetRes<()> {
        for matrices in get_square_test_matrices()? {
            let left: Matrix<i64> = matrices.left.into();
            let right: Matrix<i64> = matrices.right.into();
            assert_eq!(
                Matrix::mul_iter(&left, &right)?.take(),
                matrices.prod,
                "Iter product should be equal"
            );
        }
        Ok(())
    }

    /// Tests iterative matrix multiplication on non-square matrices
    #[test]
    fn mul_iter_asymm() -> PsetRes<()> {
        for matrices in get_asymm_test_matrices()? {
            assert_eq!(
                Matrix::mul_iter(&matrices.left.into(), &matrices.right.into())?.take(),
                matrices.prod,
                "Iter product should be equal"
            );
        }
        Ok(())
    }
}
