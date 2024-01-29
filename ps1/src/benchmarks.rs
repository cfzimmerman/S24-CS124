use crate::fib::{fib_iter, fib_mtx_mod, fib_rec, FibSz};
use std::time::{Duration, Instant};

pub type FibAlgo = fn(FibSz, u64, &mut Option<Timer>) -> Option<FibSz>;

struct Testable {
    pub name: &'static str,
    pub algo: FibAlgo,
    /// The maximum value a test should ever attempt to compute.
    pub testable_max: FibSz,
}

const ALGOS: [Testable; 3] = [
    Testable {
        name: "Recursive",
        algo: fib_rec,
        testable_max: 2048,
    },
    Testable {
        name: "Iterative",
        algo: fib_iter,
        testable_max: u64::MAX - 1,
    },
    Testable {
        name: "Mod Matrix",
        algo: fib_mtx_mod,
        testable_max: u64::MAX - 1,
    },
];

/// For questions 2a and 2c
pub fn race_test(time_limit: &Duration, mod_c: u64) {
    for algorithm in ALGOS {
        let mut fib_num: u64 = 1;
        let timer = Instant::now();
        loop {
            (algorithm.algo)(fib_num, mod_c, &mut None);
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

pub fn get_time(fib_num: u64, algo: FibAlgo, mod_c: u64) -> (Option<FibSz>, Duration) {
    let timer = Instant::now();
    let res = algo(fib_num, mod_c, &mut None);
    (res, timer.elapsed())
}

/// Binary searches for the closest value between lower_bound and upper_bound
/// for which func takes less than time_limit.
pub fn search_time(
    func: FibAlgo,
    lower_bound: FibSz,
    upper_bound: FibSz,
    mod_c: u64,
    time_limit: Duration,
) -> FibSz {
    let mut lower = lower_bound;
    let mut upper = upper_bound;
    while lower <= upper {
        let try_val = (lower + upper) / 2;
        if try_val < lower {
            println!(
                "stopped because of wrapping overflow: lower: {lower}, upper: {try_val}, try_val: {try_val}"
            );
            return lower;
        }
        println!("lower_bound: {}, upper_bound: {}", lower, upper);
        match func(try_val, mod_c, &mut Some(Timer::new(time_limit))) {
            Some(_) => {
                lower = try_val + 1;
            }
            None => {
                upper = try_val - 1;
            }
        }
    }
    lower - 1
}

/// Runs search_time on all algorithms
pub fn search_all_times(mod_c: u64, time_limit: Duration) {
    for algo in ALGOS {
        let limit = search_time(algo.algo, 1, algo.testable_max, mod_c, time_limit);
        println!("🎉 {} reached {} in {:?}\n", algo.name, limit, time_limit);
    }
}

pub struct Timer {
    length: Duration,
    running: Instant,
    expired: bool,
}

impl Timer {
    pub fn new(length: Duration) -> Self {
        Timer {
            length,
            running: Instant::now(),
            expired: false,
        }
    }

    pub fn expired(&mut self) -> bool {
        if self.running.elapsed() >= self.length {
            self.expired = true;
        }
        self.expired
    }
}
