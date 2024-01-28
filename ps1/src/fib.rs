use nalgebra::{Matrix2, Matrix2x1};

use crate::matrix::ModMtx2;

pub type FibSz = u64;

/// Computes the nth Fibonacci number using a recursive algorithm
pub fn fib_rec(num: FibSz) -> FibSz {
    match num {
        0 => 0,
        1 => 1,
        _ => fib_rec(num - 1) + fib_rec(num - 2),
    }
}

/// Computes the nth Fibonacci number using an iterative algorithm
pub fn fib_iter(num: FibSz) -> FibSz {
    let mut two_back: FibSz = 0;
    let mut one_back: FibSz = 1;
    match num {
        0 => return two_back,
        1 => return one_back,
        _ => (),
    };

    for _ in 2..=num {
        let next = one_back + two_back;
        two_back = one_back;
        one_back = next;
    }
    one_back
}

/// Computes the nth Fibonacci number using a matrix exponentiation algorithm
pub fn fib_mtx(num: FibSz) -> FibSz {
    let pow: u32 = (num - 1)
        .try_into()
        .expect("Failed to downcast fib_mtx num to u32.");
    let mut mtx = Matrix2::new(1, 1, 1, 0);
    mtx.pow_mut(pow);
    let res = mtx * Matrix2x1::new(1, 0);
    res[(0, 0)]
}

/// Computes the nth Fibonacci number ussing matrix exponentation mod a given constant
pub fn fib_mtx_mod(num: FibSz, modc: usize) -> FibSz {
    let pow: usize = (num - 1)
        .try_into()
        .expect("Failed to cast FibSz to usize.");
    let mtx = ModMtx2::new([[1, 1], [1, 0]], modc);
    let res = mtx.mod_pow(pow);
    res.get()[0][0]
}

pub fn fib_mtx_mod_2_16(num: FibSz) -> FibSz {
    fib_mtx_mod(num, 2usize.pow(16))
}

#[cfg(test)]
mod tests {
    use crate::fib::{fib_iter, fib_mtx, fib_mtx_mod, fib_mtx_mod_2_16, fib_rec, FibSz};

    const TEST_UPTO: FibSz = 20;

    #[test]
    fn compare_methods() {
        let modc = 2usize.pow(32);
        for fib in 1..=TEST_UPTO {
            let rec = fib_rec(fib);
            let iter = fib_iter(fib);
            let mtx = fib_mtx(fib);
            let mod_mtx = fib_mtx_mod(fib, modc);
            let mod_mtx_16 = fib_mtx_mod_2_16(fib);
            assert_eq!(rec, iter);
            assert_eq!(rec, mtx);
            assert_eq!(rec, mod_mtx);
            assert_eq!(rec, mod_mtx_16);
        }
    }
}
