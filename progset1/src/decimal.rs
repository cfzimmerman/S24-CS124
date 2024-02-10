use std::ops::{Add, Mul};

// The number of digits of float precision to preserve.
const PREC_POW10: i32 = 12;

/// Holds values of float precision in integer form. This makes
/// Decimal compatible with types requiring Eq and Hash.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy)]
pub struct Decimal(i64);

impl Decimal {
    pub fn new(num: f64) -> Self {
        Decimal((num * 10f64.powi(PREC_POW10)) as i64)
    }

    pub fn get(&self) -> f64 {
        self.0 as f64 / 10f64.powi(PREC_POW10)
    }
}

impl Add for &Decimal {
    type Output = Decimal;

    fn add(self, rhs: Self) -> Self::Output {
        Decimal::new(self.get() + rhs.get())
    }
}

impl Mul for &Decimal {
    type Output = Decimal;

    fn mul(self, rhs: Self) -> Self::Output {
        Decimal::new(self.get() * rhs.get())
    }
}

#[cfg(test)]
mod decimal_tests {
    use super::Decimal;

    /// Verifies the correctness of arithmetic involving Decimals
    #[test]
    fn decimal_ops() {
        let no_trunc: f64 = 0.1234567891;
        let nt1 = Decimal::new(no_trunc);
        let nt2 = Decimal::new(no_trunc);

        let nt_sum = &nt1 + &nt2;
        let nt_prod = &nt1 * &Decimal::new(2.0);
        assert_eq!(no_trunc + no_trunc, nt_sum.get(), "addition");
        assert_eq!(nt_sum, nt_prod, "multiplication");
    }
}
