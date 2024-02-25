use crate::error::{PsetErr, PsetRes};
use std::ops::{Add, AddAssign, Mul, Range, Sub};

#[derive(PartialEq, Eq)]
pub struct Matrix<T> {
    inner: Vec<Vec<T>>,
}

pub struct SliceMatrix<'a, T> {
    parent: &'a Matrix<T>,
    rows: Range<usize>,
    cols: Range<usize>,
}

impl<'a, T> SliceMatrix<'a, T>
where
    T: Mul<Output = T> + AddAssign + Default + Copy,
    for<'b> &'b T: Add<Output = T> + Sub<Output = T>,
{
    /// Returns the number of rows in this matrix
    fn num_rows(&self) -> usize {
        debug_assert!(
            self.rows.end <= self.parent.num_rows(),
            "Slice matrix row range should always be in bounds"
        );
        self.rows.start - self.rows.end
    }

    /// Returns the number of columns in this matrix
    fn num_cols(&self) -> usize {
        debug_assert!(
            self.cols.end <= self.parent.num_cols(),
            "Slice matrix col range should always be in bounds"
        );
        self.cols.start - self.cols.end
    }

    /// Returns a matrix representing the operation left + right
    pub fn add(left: &'a SliceMatrix<T>, right: &'a SliceMatrix<T>) -> PsetRes<Matrix<T>> {
        Self::op_one_to_one(left, right, |l, r| l + r)
    }

    /// Returns a matrix representing the operation left - right
    pub fn sub(left: &'a SliceMatrix<T>, right: &'a SliceMatrix<T>) -> PsetRes<Matrix<T>> {
        Self::op_one_to_one(left, right, |l, r| l - r)
    }

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
        Self::op_one_to_one(left, right, |(l, r)| l + r)
    }

    /// Returns a matrix representing the operation left - right
    pub fn sub(left: &Matrix<T>, right: &Matrix<T>) -> PsetRes<Matrix<T>> {
        Self::op_one_to_one(left, right, |(l, r)| l - r)
    }

    /// Performs O(n^3) iterative multiplication on the given matrices.
    pub fn mul_iter(left: &Matrix<T>, right: &Matrix<T>) -> PsetRes<Matrix<T>> {
        if left.num_cols() != right.num_rows() {
            return Err(PsetErr::Static("Matrix dims don't support right multiply"));
        }
        let mut res: Vec<Vec<T>> = Vec::with_capacity(left.num_rows());
        for left_row in 0..left.num_rows() {
            let mut row: Vec<T> = Vec::with_capacity(right.num_rows());
            for right_col in 0..right.num_cols() {
                let mut sum = T::default();
                for offset in 0..left.num_cols() {
                    sum += left.inner[left_row][offset] * right.inner[offset][right_col];
                }
                row.push(sum);
            }
            res.push(row);
        }
        Ok(res.into())
    }

    /// Returns the number of rows in this matrix
    fn num_rows(&self) -> usize {
        self.inner.len()
    }

    /// Returns the number of columns in this matrix
    fn num_cols(&self) -> usize {
        self.inner.get(0).map(|arr| arr.len()).unwrap_or(0)
    }

    /// Performs an operation mapping point i, j in left and right to output. Utility
    /// method for add and subtract.
    fn op_one_to_one(
        left: &Matrix<T>,
        right: &Matrix<T>,
        op: fn((&T, &T)) -> T,
    ) -> PsetRes<Matrix<T>> {
        todo!("Have the main matrix convert into a slice matrix and then call this. We only want one full impl of these operations.");
        if left.num_rows() != right.num_rows() || left.num_cols() != right.num_cols() {
            return Err(PsetErr::Static(
                "Cannot perform a one to one operation on matrices of different dimensions",
            ));
        }
        let res: Vec<Vec<T>> = left
            .inner
            .iter()
            .zip(right.inner.iter())
            .map(|(l_row, r_row)| l_row.iter().zip(r_row).map(op).collect())
            .collect();
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
        }
    }
}

#[cfg(test)]
mod matrix_tests {
    use crate::{error::PsetRes, matrix::Matrix, test_data::get_test_matrices};

    /// Tests iterative matrix multiplication
    #[test]
    fn mul_iter() -> PsetRes<()> {
        for matrices in get_test_matrices()? {
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

    /// Tests matrix addition and subtraction
    #[test]
    fn add_sub() -> PsetRes<()> {
        for matrices in get_test_matrices()? {
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
}
