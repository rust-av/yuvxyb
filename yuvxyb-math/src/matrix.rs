use crate::multiply_add;

#[derive(Debug, Clone)]
#[must_use]
pub struct RowVector(f32, f32, f32);

impl RowVector {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self(x, y, z)
    }

    #[must_use]
    pub const fn x(&self) -> f32 {
        self.0
    }
    #[must_use]
    pub const fn y(&self) -> f32 {
        self.1
    }
    #[must_use]
    pub const fn z(&self) -> f32 {
        self.2
    }

    pub fn cross(&self, other: &Self) -> Self {
        let Self(sx, sy, sz) = *self;
        let Self(ox, oy, oz) = *other;

        Self::new(
            multiply_add(sy, oz, -(sz * oy)),
            multiply_add(sz, ox, -(sx * oz)),
            multiply_add(sx, oy, -(sy * ox)),
        )
    }

    #[must_use]
    pub fn dot(&self, other: &Self) -> f32 {
        multiply_add(
            self.0,
            other.0,
            multiply_add(self.1, other.1, self.2 * other.2),
        )
    }

    pub fn scalar_div(&self, x: f32) -> Self {
        Self(self.0 / x, self.1 / x, self.2 / x)
    }

    pub fn component_mul(&self, other: &Self) -> Self {
        Self(self.0 * other.0, self.1 * other.1, self.2 * other.2)
    }

    pub fn values(self) -> [f32; 3] {
        let Self(x, y, z) = self;
        [x, y, z]
    }
}

impl From<[f32; 3]> for RowVector {
    fn from(value: [f32; 3]) -> Self {
        Self::new(value[0], value[1], value[2])
    }
}

#[derive(Debug, Clone, PartialEq)]
#[must_use]
pub struct ColVector(f32, f32, f32);

impl ColVector {
    pub const fn new(r: f32, g: f32, b: f32) -> Self {
        Self(r, g, b)
    }

    #[must_use]
    pub const fn r(&self) -> f32 {
        self.0
    }
    #[must_use]
    pub const fn g(&self) -> f32 {
        self.1
    }
    #[must_use]
    pub const fn b(&self) -> f32 {
        self.2
    }

    pub const fn transpose(self) -> RowVector {
        RowVector::new(self.0, self.1, self.2)
    }

    pub fn values(self) -> [f32; 3] {
        let Self(r, g, b) = self;
        [r, g, b]
    }
}

impl From<[f32; 3]> for ColVector {
    fn from(value: [f32; 3]) -> Self {
        Self::new(value[0], value[1], value[2])
    }
}

#[derive(Debug, Clone)]
#[must_use]
pub struct Matrix(RowVector, RowVector, RowVector);

impl Matrix {
    pub const fn new(r1: RowVector, r2: RowVector, r3: RowVector) -> Self {
        Self(r1, r2, r3)
    }

    pub const fn r1(&self) -> &RowVector {
        &self.0
    }
    pub const fn r2(&self) -> &RowVector {
        &self.1
    }
    pub const fn r3(&self) -> &RowVector {
        &self.2
    }

    pub const fn identity() -> Self {
        Self::new(
            RowVector::new(1.0, 0.0, 0.0),
            RowVector::new(0.0, 1.0, 0.0),
            RowVector::new(0.0, 0.0, 1.0),
        )
    }

    pub fn scalar_div(&self, x: f32) -> Self {
        Self(
            self.0.scalar_div(x),
            self.1.scalar_div(x),
            self.2.scalar_div(x),
        )
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

    /// Will panic if the matrix is not invertible
    pub fn invert(&self) -> Self {
        // Cramer's rule
        let Self(ref r1, ref r2, ref r3) = *self;

        let RowVector(s11, s12, s13) = *r1;
        let RowVector(s21, s22, s23) = *r2;
        let RowVector(s31, s32, s33) = *r3;

        let minor_11 = multiply_add(s22, s33, -(s32 * s23));
        let minor_12 = multiply_add(s21, s33, -(s31 * s23));
        let minor_13 = multiply_add(s21, s32, -(s31 * s22));

        let minor_21 = multiply_add(s12, s33, -(s32 * s13));
        let minor_22 = multiply_add(s11, s33, -(s31 * s13));
        let minor_23 = multiply_add(s11, s32, -(s31 * s12));

        let minor_31 = multiply_add(s12, s23, -(s22 * s13));
        let minor_32 = multiply_add(s11, s23, -(s21 * s13));
        let minor_33 = multiply_add(s11, s22, -(s21 * s12));

        let determinant = multiply_add(
            s11,
            minor_11,
            -multiply_add(s12, minor_12, -(s13 * minor_13)),
        );

        Self::new(
            RowVector::new(minor_11, -minor_12, minor_13),
            RowVector::new(-minor_21, minor_22, -minor_23),
            RowVector::new(minor_31, -minor_32, minor_33),
        )
        .transpose()
        .scalar_div(determinant)
    }

    pub fn mul_vec(&self, rhs: &ColVector) -> ColVector {
        let Self(ref r1, ref r2, ref r3) = *self;

        ColVector::new(
            multiply_add(r1.0, rhs.0, multiply_add(r1.1, rhs.1, r1.2 * rhs.2)),
            multiply_add(r2.0, rhs.0, multiply_add(r2.1, rhs.1, r2.2 * rhs.2)),
            multiply_add(r3.0, rhs.0, multiply_add(r3.1, rhs.1, r3.2 * rhs.2)),
        )
    }

    pub fn mul_mat(&self, rhs: Self) -> Self {
        let Self(ref r1, ref r2, ref r3) = *self;
        let Self(o1, o2, o3) = rhs;

        Self::new(
            RowVector::new(
                multiply_add(r1.0, o1.0, multiply_add(r1.1, o2.0, r1.2 * o3.0)),
                multiply_add(r1.0, o1.1, multiply_add(r1.1, o2.1, r1.2 * o3.1)),
                multiply_add(r1.0, o1.2, multiply_add(r1.1, o2.2, r1.2 * o3.2)),
            ),
            RowVector::new(
                multiply_add(r2.0, o1.0, multiply_add(r2.1, o2.0, r2.2 * o3.0)),
                multiply_add(r2.0, o1.1, multiply_add(r2.1, o2.1, r2.2 * o3.1)),
                multiply_add(r2.0, o1.2, multiply_add(r2.1, o2.2, r2.2 * o3.2)),
            ),
            RowVector::new(
                multiply_add(r3.0, o1.0, multiply_add(r3.1, o2.0, r3.2 * o3.0)),
                multiply_add(r3.0, o1.1, multiply_add(r3.1, o2.1, r3.2 * o3.1)),
                multiply_add(r3.0, o1.2, multiply_add(r3.1, o2.2, r3.2 * o3.2)),
            ),
        )
    }

    #[must_use]
    pub fn mul_arr(&self, rhs: [f32; 3]) -> [f32; 3] {
        let Self(ref r1, ref r2, ref r3) = *self;

        [
            multiply_add(r1.0, rhs[0], multiply_add(r1.1, rhs[1], r1.2 * rhs[2])),
            multiply_add(r2.0, rhs[0], multiply_add(r2.1, rhs[1], r2.2 * rhs[2])),
            multiply_add(r3.0, rhs[0], multiply_add(r3.1, rhs[1], r3.2 * rhs[2])),
        ]
    }

    pub fn values(self) -> [[f32; 3]; 3] {
        [
            self.0.values(),
            self.1.values(),
            self.2.values(),
        ]
    }
}
