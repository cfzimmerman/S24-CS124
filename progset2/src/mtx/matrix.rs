use super::{padding::PadPow2, slice_matrix::SliceMatrix};
use crate::error::{PsetErr, PsetRes};
use std::{
    fmt::{self, Debug, Display},
    ops::{Add, AddAssign, Mul, Sub},
};

/// A matrix type that owns its own data.
#[derive(Debug, Clone)]
pub struct Matrix<T> {
    pub inner: Vec<Vec<T>>,
}

impl<T> Matrix<T> {
    /// Returns the number of rows in this matrix
    pub fn num_rows(&self) -> usize {
        self.inner.len()
    }

    /// Returns the number of columns in this matrix
    pub fn num_cols(&self) -> usize {
        self.inner.first().map(|arr| arr.len()).unwrap_or(0)
    }
}

/// Expresses the signature of a recursive matrix multiplication
/// algorithm as used in this implementation. Allows some code
/// reuse between naive and Strassen recursive multiplication.
type RecMatrixMulAlgo<T> = fn(&SliceMatrix<T>, &SliceMatrix<T>, usize) -> PsetRes<Matrix<T>>;

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
        Self::mul_rec_inner(left, right, base_cutoff, SliceMatrix::mul_naive_rec)
    }

    /// Computes multiplication of left * right using Strassen's algorithm, switching to
    /// iterative multiplication for sub-matrices below the base_cutoff.
    pub fn mul_strassen(
        left: &mut Matrix<T>,
        right: &mut Matrix<T>,
        base_cutoff: usize,
    ) -> PsetRes<Matrix<T>> {
        Self::mul_rec_inner(left, right, base_cutoff, SliceMatrix::mul_strassen)
    }

    /// Builds an identity matrix
    /// For a usize matrix, diagon_val is 1, the value on the diagonal.
    pub fn identity(dim: usize, diagon_val: T) -> Matrix<T> {
        (0..dim)
            .map(|row_ind| {
                let mut cols = vec![T::default(); dim];
                cols[row_ind] = diagon_val;
                cols
            })
            .collect::<Vec<Vec<T>>>()
            .into()
    }

    /// Adds `other` to `self`. Returns reference for method chaining.
    pub fn add_in_place(&mut self, other: &Matrix<T>) -> PsetRes<&mut Self> {
        self.op_one_to_one_in_place(other, |l, r| l + r)?;
        Ok(self)
    }

    /// Subtracts `other` from `self`. Performs
    /// (self entry) - (other entry) at every index.
    /// Returns reference for method chaining.
    pub fn sub_in_place(&mut self, other: &Matrix<T>) -> PsetRes<&mut Self> {
        self.op_one_to_one_in_place(other, |l, r| l - r)?;
        Ok(self)
    }

    /// Assumes self is the top left in a four-part matrix. Adds neighbors into self
    /// in their named positions.
    /// Fails unless neighbors have the row and column similarities to produce a
    /// complete rectangle from the constituent parts.
    pub fn merge_neighbors(
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

    /// Performs a recursive matrix multiplication algorithm. Takes care of
    /// both error checking and padding.
    fn mul_rec_inner(
        left: &mut Matrix<T>,
        right: &mut Matrix<T>,
        base_cutoff: usize,
        algo: RecMatrixMulAlgo<T>,
    ) -> PsetRes<Matrix<T>> {
        if base_cutoff < 3 {
            return Err(PsetErr::Static("rec_mul base_cutoff must exceed 2"));
        }
        let left_pad = PadPow2::new(left);
        let right_pad = PadPow2::new(right);

        let left_sl: SliceMatrix<T> = (&*left).into();
        let right_sl: SliceMatrix<T> = (&*right).into();
        let mut res = algo(&left_sl, &right_sl, base_cutoff)?;

        left_pad.undo(left);
        right_pad.undo(right);
        PadPow2::trim_dims(&mut res, left_pad.rows_init_ct, right_pad.cols_init_ct);
        Ok(res)
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
}

impl<T> From<Vec<Vec<T>>> for Matrix<T>
where
    T: Default,
{
    fn from(item: Vec<Vec<T>>) -> Self {
        Matrix { inner: item }
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
