use std::time::{Duration, Instant};

use nalgebra::{Matrix2, Matrix2x1};

pub type FibSz = u64;

/// Computes the nth Fibonacci number using a recursive algorithm
fn fib_rec(num: FibSz) -> FibSz {
    match num {
        0 => 0,
        1 => 1,
        _ => fib_rec(num - 1) + fib_rec(num - 2),
    }
}

/// Computes the nth Fibonacci number using an iterative algorithm
fn fib_iter(num: FibSz) -> FibSz {
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
fn fib_mtx(num: FibSz) -> FibSz {
    let mut mtx = Matrix2::new(1, 1, 1, 0);
    let pow: u32 = (num - 1)
        .try_into()
        .expect("Failed to downcast fib_mtx num to u32.");
    mtx.pow_mut(pow);
    let res = mtx * Matrix2x1::new(1, 0);
    res[(0, 0)]
}

struct Testable {
    pub name: &'static str,
    pub algo: fn(FibSz) -> FibSz,
}

const ALGOS: [Testable; 3] = [
    Testable {
        name: "Recursive",
        algo: fib_rec,
    },
    Testable {
        name: "Iterative",
        algo: fib_iter,
    },
    Testable {
        name: "Matrix",
        algo: fib_mtx,
    },
];

const TIME_LIMIT: Duration = Duration::from_secs(16);

fn main() {
    for algorithm in ALGOS {
        let mut fib_num: u64 = 1;
        let timer = Instant::now();
        loop {
            (algorithm.algo)(fib_num);
            if TIME_LIMIT <= timer.elapsed() {
                break;
            }
            fib_num += 1;
        }
        println!(
            "{} reached Fibonacci number {} in less than {:?}",
            algorithm.name, fib_num, TIME_LIMIT
        );
    }
}

/*

Three runs in release mode, v1.75.0 on an M1 Macbook Pro

Recursive reached Fibonacci number 47 in less than 16s
Iterative reached Fibonacci number 225292 in less than 16s
Matrix reached Fibonacci number 196707435 in less than 16s

Recursive reached Fibonacci number 47 in less than 16s
Iterative reached Fibonacci number 225173 in less than 16s
Matrix reached Fibonacci number 196817479 in less than 16s

Recursive reached Fibonacci number 47 in less than 16s
Iterative reached Fibonacci number 225279 in less than 16s
Matrix reached Fibonacci number 196781876 in less than 16s

*/

#[cfg(test)]
mod tests {
    use crate::{fib_iter, fib_mtx, fib_rec, FibSz};

    const TEST_UPTO: FibSz = 20;

    #[test]
    fn compare_methods() {
        for fib in 1..=TEST_UPTO {
            let rec = fib_rec(fib);
            let iter = fib_iter(fib);
            let mtx = fib_mtx(fib);
            assert_eq!(rec, iter);
            assert_eq!(rec, mtx);
        }
    }
}
