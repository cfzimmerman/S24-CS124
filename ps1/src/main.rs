pub type FibSz = u64;

fn fib_rec(num: FibSz) -> FibSz {
    match num {
        0 => 0,
        1 => 1,
        _ => fib_rec(num - 1) + fib_rec(num - 2),
    }
}

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

fn main() {}

#[cfg(test)]
mod tests {
    use crate::{fib_iter, fib_rec, FibSz};

    const TEST_UPTO: FibSz = 20;

    #[test]
    fn yield_identical() {
        for fib in 1..=TEST_UPTO {
            let rec = fib_rec(fib);
            let iter = fib_iter(fib);
            assert_eq!(rec, iter);
        }
    }
}
