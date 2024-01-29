use crate::{benchmarks::Timer, matrix::ModMtx2};
use nalgebra::{Matrix2, Matrix2x1};

pub type FibSz = u64;

/// Computes the nth Fibonacci number using a recursive algorithm
pub fn fib_rec(num: FibSz, modc: u64, timer: &mut Option<Timer>) -> Option<FibSz> {
    if let Some(tm) = timer {
        if tm.expired() {
            return None;
        }
    }
    match num {
        0 => return Some(0),
        1 => return Some(1),
        _ => (),
    };
    match (fib_rec(num - 1, modc, timer), fib_rec(num - 2, modc, timer)) {
        (Some(o), Some(t)) => Some((o + t) % modc),
        _ => None,
    }
}

/// Computes the nth Fibonacci number using an iterative algorithm
pub fn fib_iter(num: FibSz, modc: u64, timer: &mut Option<Timer>) -> Option<FibSz> {
    let mut two_back: FibSz = 0;
    let mut one_back: FibSz = 1;
    match num {
        0 => return Some(two_back),
        1 => return Some(one_back),
        _ => (),
    };

    for _ in 2..=num {
        if let Some(tm) = timer {
            if tm.expired() {
                return None;
            }
        }
        if u64::MAX - one_back < two_back {
            println!("Exited fib_iter early to prevent overflow: one_back: {one_back}, two_back: {two_back}");
            return Some(one_back);
        }
        let next = one_back + two_back;
        two_back = one_back % modc;
        one_back = next % modc;
    }
    Some(one_back)
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
pub fn fib_mtx_mod(num: FibSz, modc: u64, timer: &mut Option<Timer>) -> Option<FibSz> {
    let pow: u64 = (num - 1).try_into().expect("Failed to cast FibSz to u64.");
    let mtx = ModMtx2::new([[1, 1], [1, 0]], modc);
    mtx.mod_pow(pow, timer).map(|res| res.get()[0][0])
}

#[cfg(test)]
mod tests {
    use crate::fib::{fib_iter, fib_mtx_mod, fib_rec};

    #[test]
    fn compare_methods() {
        let test_upto = 20;
        let modc = u64::MAX;
        for fib in 1..=test_upto {
            let rec = fib_rec(fib, modc, &mut None);
            let iter = fib_iter(fib, modc, &mut None);
            let mod_mtx = fib_mtx_mod(fib, modc, &mut None);
            assert_eq!(rec, iter);
            assert_eq!(rec, mod_mtx);
        }
    }

    #[test]
    fn compare_faster_methods() {
        let test_upto = 2u64.pow(12);
        let modc = 2u64.pow(16);
        for fib in 1..=test_upto {
            let iter = fib_iter(fib, modc, &mut None);
            let mod_mtx = fib_mtx_mod(fib, modc, &mut None);
            assert_eq!(iter, mod_mtx);
        }
    }
}
