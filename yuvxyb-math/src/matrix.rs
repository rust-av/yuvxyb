// These types are intended for floats, which are not `Eq`
#![allow(clippy::derive_partial_eq_without_eq)]

use std::ops::{Div, Mul, Neg};

use crate::mul_add::FastMulAdd;

#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct RowVector<T>(T, T, T);

impl<T: Copy> RowVector<T> {
    pub const fn new(x: T, y: T, z: T) -> Self {
        Self(x, y, z)
    }

    #[must_use]
    pub const fn x(&self) -> T {
        self.0
    }
    #[must_use]
    pub const fn y(&self) -> T {
        self.1
    }
    #[must_use]
    pub const fn z(&self) -> T {
        self.2
    }

    pub const fn values(self) -> [T; 3] {
        let Self(x, y, z) = self;
        [x, y, z]
    }
}

impl<T> RowVector<T>
where
    T: Copy + FastMulAdd + Mul<T, Output = T> + Div<T, Output = T> + Neg<Output = T>,
{
    pub fn cross(&self, other: &Self) -> Self {
        let Self(sx, sy, sz) = *self;
        let Self(ox, oy, oz) = *other;

        Self::new(
            sy.fast_mul_add(oz, -(sz * oy)),
            sz.fast_mul_add(ox, -(sx * oz)),
            sx.fast_mul_add(oy, -(sy * ox)),
        )
    }

    #[must_use]
    pub fn dot(&self, other: &Self) -> T {
        self.0
            .fast_mul_add(other.0, self.1.fast_mul_add(other.1, self.2 * other.2))
    }

    pub fn scalar_div(&self, x: T) -> Self {
        Self(self.0 / x, self.1 / x, self.2 / x)
    }

    pub fn component_mul(&self, other: &Self) -> Self {
        Self(self.0 * other.0, self.1 * other.1, self.2 * other.2)
    }
}

impl<T: Copy> From<[T; 3]> for RowVector<T> {
    fn from(value: [T; 3]) -> Self {
        let [x, y, z] = value;
        Self::new(x, y, z)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct ColVector<T>(T, T, T);

impl<T: Copy> ColVector<T> {
    pub const fn new(r: T, g: T, b: T) -> Self {
        Self(r, g, b)
    }

    #[must_use]
    pub const fn r(&self) -> T {
        self.0
    }
    #[must_use]
    pub const fn g(&self) -> T {
        self.1
    }
    #[must_use]
    pub const fn b(&self) -> T {
        self.2
    }

    pub const fn transpose(self) -> RowVector<T> {
        RowVector::new(self.0, self.1, self.2)
    }

    pub const fn values(self) -> [T; 3] {
        let Self(r, g, b) = self;
        [r, g, b]
    }
}

impl<T: Copy> From<[T; 3]> for ColVector<T> {
    fn from(value: [T; 3]) -> Self {
        let [r, g, b] = value;
        Self::new(r, g, b)
    }
}

#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct Matrix<T>(RowVector<T>, RowVector<T>, RowVector<T>);

impl<T: Copy> Matrix<T> {
    pub const fn new(r1: RowVector<T>, r2: RowVector<T>, r3: RowVector<T>) -> Self {
        Self(r1, r2, r3)
    }

    pub const fn r1(&self) -> &RowVector<T> {
        &self.0
    }
    pub const fn r2(&self) -> &RowVector<T> {
        &self.1
    }
    pub const fn r3(&self) -> &RowVector<T> {
        &self.2
    }

    pub const fn transpose(self) -> Self {
        let Self(r1, r2, r3) = self;

        let RowVector(s11, s12, s13) = r1;
        let RowVector(s21, s22, s23) = r2;
        let RowVector(s31, s32, s33) = r3;

        Self::new(
            RowVector::new(s11, s21, s31),
            RowVector::new(s12, s22, s32),
            RowVector::new(s13, s23, s33),
        )
    }
}

impl Matrix<f32> {
    pub const fn identity() -> Self {
        Self::new(
            RowVector::new(1.0, 0.0, 0.0),
            RowVector::new(0.0, 1.0, 0.0),
            RowVector::new(0.0, 0.0, 1.0),
        )
    }
}

impl Matrix<f64> {
    pub const fn identity() -> Self {
        Self::new(
            RowVector::new(1.0, 0.0, 0.0),
            RowVector::new(0.0, 1.0, 0.0),
            RowVector::new(0.0, 0.0, 1.0),
        )
    }
}

impl<T> Matrix<T>
where
    T: Copy + FastMulAdd + Mul<T, Output = T> + Div<T, Output = T> + Neg<Output = T>,
{
    pub fn scalar_div(&self, x: T) -> Self {
        Self(
            self.0.scalar_div(x),
            self.1.scalar_div(x),
            self.2.scalar_div(x),
        )
    }

    /// Will panic if the matrix is not invertible
    pub fn invert(&self) -> Self {
        // Cramer's rule
        let Self(ref r1, ref r2, ref r3) = *self;

        let RowVector(s11, s12, s13) = *r1;
        let RowVector(s21, s22, s23) = *r2;
        let RowVector(s31, s32, s33) = *r3;

        let minor_11 = s22.fast_mul_add(s33, -(s32 * s23));
        let minor_12 = s21.fast_mul_add(s33, -(s31 * s23));
        let minor_13 = s21.fast_mul_add(s32, -(s31 * s22));

        let minor_21 = s12.fast_mul_add(s33, -(s32 * s13));
        let minor_22 = s11.fast_mul_add(s33, -(s31 * s13));
        let minor_23 = s11.fast_mul_add(s32, -(s31 * s12));

        let minor_31 = s12.fast_mul_add(s23, -(s22 * s13));
        let minor_32 = s11.fast_mul_add(s23, -(s21 * s13));
        let minor_33 = s11.fast_mul_add(s22, -(s21 * s12));

        let determinant =
            s11.fast_mul_add(minor_11, -s12.fast_mul_add(minor_12, -(s13 * minor_13)));

        Self::new(
            RowVector::new(minor_11, -minor_12, minor_13),
            RowVector::new(-minor_21, minor_22, -minor_23),
            RowVector::new(minor_31, -minor_32, minor_33),
        )
        .transpose()
        .scalar_div(determinant)
    }

    pub fn mul_vec(&self, rhs: &ColVector<T>) -> ColVector<T> {
        let Self(ref r1, ref r2, ref r3) = *self;

        ColVector::new(
            r1.0.fast_mul_add(rhs.0, r1.1.fast_mul_add(rhs.1, r1.2 * rhs.2)),
            r2.0.fast_mul_add(rhs.0, r2.1.fast_mul_add(rhs.1, r2.2 * rhs.2)),
            r3.0.fast_mul_add(rhs.0, r3.1.fast_mul_add(rhs.1, r3.2 * rhs.2)),
        )
    }

    pub fn mul_mat(&self, rhs: Self) -> Self {
        let Self(ref r1, ref r2, ref r3) = *self;
        let Self(o1, o2, o3) = rhs;

        Self::new(
            RowVector::new(
                r1.0.fast_mul_add(o1.0, r1.1.fast_mul_add(o2.0, r1.2 * o3.0)),
                r1.0.fast_mul_add(o1.1, r1.1.fast_mul_add(o2.1, r1.2 * o3.1)),
                r1.0.fast_mul_add(o1.2, r1.1.fast_mul_add(o2.2, r1.2 * o3.2)),
            ),
            RowVector::new(
                r2.0.fast_mul_add(o1.0, r2.1.fast_mul_add(o2.0, r2.2 * o3.0)),
                r2.0.fast_mul_add(o1.1, r2.1.fast_mul_add(o2.1, r2.2 * o3.1)),
                r2.0.fast_mul_add(o1.2, r2.1.fast_mul_add(o2.2, r2.2 * o3.2)),
            ),
            RowVector::new(
                r3.0.fast_mul_add(o1.0, r3.1.fast_mul_add(o2.0, r3.2 * o3.0)),
                r3.0.fast_mul_add(o1.1, r3.1.fast_mul_add(o2.1, r3.2 * o3.1)),
                r3.0.fast_mul_add(o1.2, r3.1.fast_mul_add(o2.2, r3.2 * o3.2)),
            ),
        )
    }

    #[must_use]
    pub fn mul_arr(&self, rhs: [T; 3]) -> [T; 3] {
        let Self(ref r1, ref r2, ref r3) = *self;

        [
            r1.0.fast_mul_add(rhs[0], r1.1.fast_mul_add(rhs[1], r1.2 * rhs[2])),
            r2.0.fast_mul_add(rhs[0], r2.1.fast_mul_add(rhs[1], r2.2 * rhs[2])),
            r3.0.fast_mul_add(rhs[0], r3.1.fast_mul_add(rhs[1], r3.2 * rhs[2])),
        ]
    }

    pub const fn values(self) -> [[T; 3]; 3] {
        [self.0.values(), self.1.values(), self.2.values()]
    }
}
