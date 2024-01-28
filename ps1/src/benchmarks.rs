use std::time::{Duration, Instant};

use crate::fib::{fib_iter, fib_mtx, fib_mtx_mod_2_16, fib_rec, FibSz};

struct Testable {
    pub name: &'static str,
    pub algo: fn(FibSz) -> FibSz,
}

const ALGOS: [Testable; 4] = [
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
    Testable {
        name: "Mod Matrix",
        algo: fib_mtx_mod_2_16,
    },
];

/// For questions 2a and 2c
pub fn race_test(time_limit: &Duration) {
    for algorithm in ALGOS {
        let mut fib_num: u64 = 1;
        let timer = Instant::now();
        loop {
            (algorithm.algo)(fib_num);
            if time_limit <= &timer.elapsed() {
                break;
            }
            fib_num += 1;
        }
        println!(
            "{} reached Fibonacci number {} in less than {:?}",
            algorithm.name, fib_num, time_limit
        );
    }
}

pub fn get_times(fib_num: u64) {
    for algorithm in ALGOS {
        let timer = Instant::now();
        (algorithm.algo)(fib_num);
        println!("{} in {:?}", algorithm.name, timer.elapsed());
    }
}
