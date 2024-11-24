// We want the "suboptimal" (less accurate) case here because it is faster.
#![allow(clippy::suboptimal_flops)]

use std::ops::{Add, Mul};

/// Computes (a * b) + c, leveraging FMA if available
#[inline]
#[must_use]
pub fn multiply_add(a: f32, b: f32, c: f32) -> f32 {
    if cfg!(target_feature = "fma") {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

pub trait FastMulAdd: Sized + Mul<Self, Output = Self> + Add<Self, Output = Self> {
    /// Computes (self * a) + b, leveraging FMA if available.
    ///
    /// If FMA is not available, the implementation should prefer computation
    /// speed over accuracy (i.e. compute (self * a) + b without the benefits
    /// of just one rounding error).
    fn fast_mul_add(self, a: Self, b: Self) -> Self;
}

impl FastMulAdd for f32 {
    fn fast_mul_add(self, a: Self, b: Self) -> Self {
        if cfg!(target_feature = "fma") {
            self.mul_add(a, b)
        } else {
            self * a + b
        }
    }
}

impl FastMulAdd for f64 {
    fn fast_mul_add(self, a: Self, b: Self) -> Self {
        if cfg!(target_feature = "fma") {
            self.mul_add(a, b)
        } else {
            self * a + b
        }
    }
}
