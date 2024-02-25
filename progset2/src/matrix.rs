use crate::error::{PsetErr, PsetRes};
use std::ops::{Add, AddAssign, Mul, Sub};

#[derive(PartialEq, Eq)]
pub struct Matrix<T> {
    inner: Vec<Vec<T>>,
}

impl<T> Matrix<T>
where
    T: Add<Output = T> + Sub + Mul + AddAssign + Default + Copy,
{
    /// Returns the number of rows in this matrix
    fn num_rows(&self) -> usize {
        self.inner.len()
    }

    /// Returns the number of columns in this matrix
    fn num_cols(&self) -> usize {
        self.inner.get(0).map(|arr| arr.len()).unwrap_or(0)
    }

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
                    sum += left.inner[left_row][offset] + right.inner[offset][right_col];
                }
                row.push(sum);
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
