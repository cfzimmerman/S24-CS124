use ps1::{benchmarks::get_time, fib::fib_mtx_mod};

fn main() {
    let mod_c = 2u64.pow(16);
    // search_all_times(mod_c, Duration::from_secs(60));
    dbg!(get_time(u64::MAX, fib_mtx_mod, mod_c));
}
