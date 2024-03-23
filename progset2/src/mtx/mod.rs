pub mod cli;
pub mod experiments;
pub mod matrix;
mod padding;
mod slice_matrix;

#[cfg(test)]
mod matrix_tests {
    use crate::{
        error::PsetRes,
        mtx::{matrix::Matrix, padding::PadPow2},
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
            assert_eq!(
                Matrix::mul_strassen(&mut left, &mut right, 3)?.inner,
                matrices.prod,
                "Strassen product should be equal"
            );
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
                assert_eq!(
                    Matrix::mul_strassen(&mut left, &mut right, base_cutoff)?.inner,
                    matrices.prod,
                    "Strassen product should be equal"
                );
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
        let strassen_prod = Matrix::mul_strassen(&mut left_mtx, &mut right_mtx, 3)?;

        assert_eq!(iter_prod.inner, inputs.prod, "Iter should equal input prod");
        assert_eq!(
            naive_rec_prod.inner, inputs.prod,
            "Naive rec should equal input prod"
        );
        assert_eq!(
            strassen_prod.inner, inputs.prod,
            "Strassen should equal input prod"
        );

        Ok(())
    }

    #[test]
    fn static_6x5_mul() -> PsetRes<()> {
        let inputs = get_test_6x5()?;
        let mut left_mtx: Matrix<i64> = inputs.left.into();
        let mut right_mtx: Matrix<i64> = inputs.right.into();

        let iter_prod = Matrix::mul_iter(&left_mtx, &right_mtx)?;
        let naive_rec_prod = Matrix::mul_naive_rec(&mut left_mtx, &mut right_mtx, 3)?;
        let strassen_prod = Matrix::mul_strassen(&mut left_mtx, &mut right_mtx, 3)?;

        assert_eq!(iter_prod.inner, inputs.prod, "Iter should equal input prod");
        assert_eq!(
            naive_rec_prod.inner, inputs.prod,
            "Naive rec should equal input prod"
        );
        assert_eq!(
            strassen_prod.inner, inputs.prod,
            "Strassen should equal input prod"
        );

        Ok(())
    }

    /// Matrix padding should yield a square with height and width
    /// dimensions as a power of 2.
    #[test]
    fn matrix_padding() -> PsetRes<()> {
        let mut m4x5 = Matrix::identity(5, 1u8);
        m4x5.inner.pop();
        // Now it's a 4 x 5 matrix
        PadPow2::pad_pow2(&mut m4x5);
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
