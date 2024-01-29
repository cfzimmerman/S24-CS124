use crate::{benchmarks::Timer, fib::FibSz};

pub type RawMtx = [[FibSz; 2]; 2];

#[derive(Debug, PartialEq)]
pub struct ModMtx2 {
    mtx: RawMtx,
    modc: u64,
}

impl ModMtx2 {
    /// Constructs a new 2 by 2 ModMtx. Each inner array in RawMtx is a row.
    pub fn new(mut mtx: RawMtx, modc: u64) -> Self {
        for row in 0..2 {
            for col in 0..2 {
                mtx[row][col] = mtx[row][col] % modc as FibSz;
            }
        }
        ModMtx2 { mtx, modc }
    }

    pub fn get(&self) -> &RawMtx {
        &self.mtx
    }

    /// Constructs the identity matrix
    pub fn identity(modc: u64) -> ModMtx2 {
        ModMtx2 {
            mtx: [[1, 0], [0, 1]],
            modc,
        }
    }

    /// Computes the dot product of self's row and factor's col mod self.modc
    fn mod_dot_prod(&self, factor: &ModMtx2, row: usize, col: usize) -> FibSz {
        let mut prod = 0;
        for ind in 0..2 {
            prod += self.mtx[row][ind] * factor.mtx[ind][col];
        }
        prod % self.modc as FibSz
    }

    /// Returns the result of a 2x2 matrix multiplication
    pub fn mod_mult(&self, factor: &ModMtx2) -> Self {
        let mut res = Self::identity(self.modc);
        for res_row in 0..2 {
            for res_col in 0..2 {
                res.mtx[res_row][res_col] = self.mod_dot_prod(factor, res_row, res_col);
            }
        }
        res
    }

    /// Computes matrix power mod modc. If a timer is provided, calculation
    /// will quit and return None if the time limit is exceeded.
    pub fn mod_pow(mut self, pow: u64, timer: &mut Option<Timer>) -> Option<Self> {
        let mut pow = pow;
        let mut odd_facs = Self::identity(self.modc);
        if pow == 0 {
            return Some(odd_facs);
        }

        while pow > 1 {
            if let Some(tm) = timer {
                if tm.expired() {
                    return None;
                }
            }
            if pow % 2 == 1 {
                odd_facs = self.mod_mult(&odd_facs);
                pow -= 1;
            }
            self.mtx = self.mod_mult(&self).mtx;
            pow /= 2;
        }
        Some(self.mod_mult(&odd_facs))
    }
}

#[cfg(test)]
mod tests {
    use super::ModMtx2;

    #[test]
    fn unmodded_mult() {
        let mod_c = 512;
        let base = ModMtx2::new([[14, 23], [9, 1]], mod_c);
        let mult = ModMtx2::new([[7, 2], [5, 11]], mod_c);
        let expected = ModMtx2::new([[213, 281], [68, 29]], mod_c);

        assert_eq!(base.mod_mult(&mult), expected);
        assert_eq!(base.mod_mult(&ModMtx2::identity(mod_c)), base);
    }

    #[test]
    fn modded_mult() {
        let mod_c = 64;
        let base = ModMtx2::new([[14, 23], [9, 1]], mod_c);
        let mult = ModMtx2::new([[7, 2], [5, 11]], mod_c);
        let expected = ModMtx2::new([[213, 281], [68, 29]], mod_c);

        assert_eq!(base.mod_mult(&mult), expected);
    }

    #[test]
    fn unmodded_pow() {
        let mod_c = 2048;
        let pow = 5;
        let base = ModMtx2::new([[2, 1], [2, 3]], mod_c);
        let expected = ModMtx2::new([[342, 341], [682, 683]], mod_c);
        assert_eq!(base.mod_pow(pow, &mut None), Some(expected));
    }

    #[test]
    fn modded_pow() {
        let mod_c = 64;
        let pow = 10;
        let base = ModMtx2::new([[1, 1], [1, 0]], mod_c);
        let expected = ModMtx2::new([[89, 55], [55, 34]], mod_c);
        assert_eq!(base.mod_pow(pow, &mut None), Some(expected));
    }
}
