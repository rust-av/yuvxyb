//! SIMD-optimized cube root implementations using the `wide` crate.
//!
//! This module provides vectorized cube root functions that process
//! 4, 8, or 16 f32 values at once using SSE/AVX/AVX-512/NEON instructions.

use wide::{f32x4, f32x8, f32x16, f64x2};

/// Vectorized cube root for 4 f32 values.
///
/// Uses the same Newton-Raphson algorithm as the scalar cbrtf_fast,
/// but processes 4 values in parallel.
#[inline]
#[must_use]
pub fn cbrtf_x4(x: f32x4) -> f32x4 {
    // Convert to arrays for bit manipulation
    let x_arr: [f32; 4] = x.into();

    // Process each element: get initial approximation via bit manipulation
    let t_arr: [f32; 4] = [
        initial_approx(x_arr[0]),
        initial_approx(x_arr[1]),
        initial_approx(x_arr[2]),
        initial_approx(x_arr[3]),
    ];

    // Convert to f64 for Newton iterations (for accuracy)
    // Process in two halves using f64x2
    let x_lo = f64x2::new([x_arr[0] as f64, x_arr[1] as f64]);
    let x_hi = f64x2::new([x_arr[2] as f64, x_arr[3] as f64]);
    let t_lo = f64x2::new([t_arr[0] as f64, t_arr[1] as f64]);
    let t_hi = f64x2::new([t_arr[2] as f64, t_arr[3] as f64]);

    // First Newton iteration: t = t * (x + x + r) / (x + r + r) where r = t^3
    let r_lo = t_lo * t_lo * t_lo;
    let r_hi = t_hi * t_hi * t_hi;
    let x2_lo = x_lo + x_lo;
    let x2_hi = x_hi + x_hi;
    let t_lo = t_lo * (x2_lo + r_lo) / (x_lo + r_lo + r_lo);
    let t_hi = t_hi * (x2_hi + r_hi) / (x_hi + r_hi + r_hi);

    // Second Newton iteration
    let r_lo = t_lo * t_lo * t_lo;
    let r_hi = t_hi * t_hi * t_hi;
    let t_lo = t_lo * (x2_lo + r_lo) / (x_lo + r_lo + r_lo);
    let t_hi = t_hi * (x2_hi + r_hi) / (x_hi + r_hi + r_hi);

    // Convert back to f32
    let t_lo_arr: [f64; 2] = t_lo.into();
    let t_hi_arr: [f64; 2] = t_hi.into();
    f32x4::new([
        t_lo_arr[0] as f32,
        t_lo_arr[1] as f32,
        t_hi_arr[0] as f32,
        t_hi_arr[1] as f32,
    ])
}

/// Vectorized cube root for 8 f32 values (AVX).
///
/// Uses the same Newton-Raphson algorithm as the scalar cbrtf_fast,
/// but processes 8 values in parallel.
#[inline]
#[must_use]
pub fn cbrtf_x8(x: f32x8) -> f32x8 {
    let x_arr: [f32; 8] = x.into();

    // Get initial approximations
    let t_arr: [f32; 8] = [
        initial_approx(x_arr[0]),
        initial_approx(x_arr[1]),
        initial_approx(x_arr[2]),
        initial_approx(x_arr[3]),
        initial_approx(x_arr[4]),
        initial_approx(x_arr[5]),
        initial_approx(x_arr[6]),
        initial_approx(x_arr[7]),
    ];

    // Process in four f64x2 chunks for precision
    let x0 = f64x2::new([x_arr[0] as f64, x_arr[1] as f64]);
    let x1 = f64x2::new([x_arr[2] as f64, x_arr[3] as f64]);
    let x2 = f64x2::new([x_arr[4] as f64, x_arr[5] as f64]);
    let x3 = f64x2::new([x_arr[6] as f64, x_arr[7] as f64]);

    let mut t0 = f64x2::new([t_arr[0] as f64, t_arr[1] as f64]);
    let mut t1 = f64x2::new([t_arr[2] as f64, t_arr[3] as f64]);
    let mut t2 = f64x2::new([t_arr[4] as f64, t_arr[5] as f64]);
    let mut t3 = f64x2::new([t_arr[6] as f64, t_arr[7] as f64]);

    let x2_0 = x0 + x0;
    let x2_1 = x1 + x1;
    let x2_2 = x2 + x2;
    let x2_3 = x3 + x3;

    // First Newton iteration
    let r0 = t0 * t0 * t0;
    let r1 = t1 * t1 * t1;
    let r2 = t2 * t2 * t2;
    let r3 = t3 * t3 * t3;
    t0 = t0 * (x2_0 + r0) / (x0 + r0 + r0);
    t1 = t1 * (x2_1 + r1) / (x1 + r1 + r1);
    t2 = t2 * (x2_2 + r2) / (x2 + r2 + r2);
    t3 = t3 * (x2_3 + r3) / (x3 + r3 + r3);

    // Second Newton iteration
    let r0 = t0 * t0 * t0;
    let r1 = t1 * t1 * t1;
    let r2 = t2 * t2 * t2;
    let r3 = t3 * t3 * t3;
    t0 = t0 * (x2_0 + r0) / (x0 + r0 + r0);
    t1 = t1 * (x2_1 + r1) / (x1 + r1 + r1);
    t2 = t2 * (x2_2 + r2) / (x2 + r2 + r2);
    t3 = t3 * (x2_3 + r3) / (x3 + r3 + r3);

    // Convert back to f32
    let t0_arr: [f64; 2] = t0.into();
    let t1_arr: [f64; 2] = t1.into();
    let t2_arr: [f64; 2] = t2.into();
    let t3_arr: [f64; 2] = t3.into();
    f32x8::new([
        t0_arr[0] as f32,
        t0_arr[1] as f32,
        t1_arr[0] as f32,
        t1_arr[1] as f32,
        t2_arr[0] as f32,
        t2_arr[1] as f32,
        t3_arr[0] as f32,
        t3_arr[1] as f32,
    ])
}

/// Vectorized cube root for 16 f32 values (AVX-512).
///
/// Uses the same Newton-Raphson algorithm as the scalar cbrtf_fast,
/// but processes 16 values in parallel.
#[inline]
#[must_use]
pub fn cbrtf_x16(x: f32x16) -> f32x16 {
    let x_arr: [f32; 16] = x.into();

    // Get initial approximations for all 16 elements
    let t_arr: [f32; 16] = [
        initial_approx(x_arr[0]),
        initial_approx(x_arr[1]),
        initial_approx(x_arr[2]),
        initial_approx(x_arr[3]),
        initial_approx(x_arr[4]),
        initial_approx(x_arr[5]),
        initial_approx(x_arr[6]),
        initial_approx(x_arr[7]),
        initial_approx(x_arr[8]),
        initial_approx(x_arr[9]),
        initial_approx(x_arr[10]),
        initial_approx(x_arr[11]),
        initial_approx(x_arr[12]),
        initial_approx(x_arr[13]),
        initial_approx(x_arr[14]),
        initial_approx(x_arr[15]),
    ];

    // Process in eight f64x2 chunks for f64 precision
    let x0 = f64x2::new([x_arr[0] as f64, x_arr[1] as f64]);
    let x1 = f64x2::new([x_arr[2] as f64, x_arr[3] as f64]);
    let x2 = f64x2::new([x_arr[4] as f64, x_arr[5] as f64]);
    let x3 = f64x2::new([x_arr[6] as f64, x_arr[7] as f64]);
    let x4 = f64x2::new([x_arr[8] as f64, x_arr[9] as f64]);
    let x5 = f64x2::new([x_arr[10] as f64, x_arr[11] as f64]);
    let x6 = f64x2::new([x_arr[12] as f64, x_arr[13] as f64]);
    let x7 = f64x2::new([x_arr[14] as f64, x_arr[15] as f64]);

    let mut t0 = f64x2::new([t_arr[0] as f64, t_arr[1] as f64]);
    let mut t1 = f64x2::new([t_arr[2] as f64, t_arr[3] as f64]);
    let mut t2 = f64x2::new([t_arr[4] as f64, t_arr[5] as f64]);
    let mut t3 = f64x2::new([t_arr[6] as f64, t_arr[7] as f64]);
    let mut t4 = f64x2::new([t_arr[8] as f64, t_arr[9] as f64]);
    let mut t5 = f64x2::new([t_arr[10] as f64, t_arr[11] as f64]);
    let mut t6 = f64x2::new([t_arr[12] as f64, t_arr[13] as f64]);
    let mut t7 = f64x2::new([t_arr[14] as f64, t_arr[15] as f64]);

    let x2_0 = x0 + x0;
    let x2_1 = x1 + x1;
    let x2_2 = x2 + x2;
    let x2_3 = x3 + x3;
    let x2_4 = x4 + x4;
    let x2_5 = x5 + x5;
    let x2_6 = x6 + x6;
    let x2_7 = x7 + x7;

    // First Newton iteration
    let r0 = t0 * t0 * t0;
    let r1 = t1 * t1 * t1;
    let r2 = t2 * t2 * t2;
    let r3 = t3 * t3 * t3;
    let r4 = t4 * t4 * t4;
    let r5 = t5 * t5 * t5;
    let r6 = t6 * t6 * t6;
    let r7 = t7 * t7 * t7;
    t0 = t0 * (x2_0 + r0) / (x0 + r0 + r0);
    t1 = t1 * (x2_1 + r1) / (x1 + r1 + r1);
    t2 = t2 * (x2_2 + r2) / (x2 + r2 + r2);
    t3 = t3 * (x2_3 + r3) / (x3 + r3 + r3);
    t4 = t4 * (x2_4 + r4) / (x4 + r4 + r4);
    t5 = t5 * (x2_5 + r5) / (x5 + r5 + r5);
    t6 = t6 * (x2_6 + r6) / (x6 + r6 + r6);
    t7 = t7 * (x2_7 + r7) / (x7 + r7 + r7);

    // Second Newton iteration
    let r0 = t0 * t0 * t0;
    let r1 = t1 * t1 * t1;
    let r2 = t2 * t2 * t2;
    let r3 = t3 * t3 * t3;
    let r4 = t4 * t4 * t4;
    let r5 = t5 * t5 * t5;
    let r6 = t6 * t6 * t6;
    let r7 = t7 * t7 * t7;
    t0 = t0 * (x2_0 + r0) / (x0 + r0 + r0);
    t1 = t1 * (x2_1 + r1) / (x1 + r1 + r1);
    t2 = t2 * (x2_2 + r2) / (x2 + r2 + r2);
    t3 = t3 * (x2_3 + r3) / (x3 + r3 + r3);
    t4 = t4 * (x2_4 + r4) / (x4 + r4 + r4);
    t5 = t5 * (x2_5 + r5) / (x5 + r5 + r5);
    t6 = t6 * (x2_6 + r6) / (x6 + r6 + r6);
    t7 = t7 * (x2_7 + r7) / (x7 + r7 + r7);

    // Convert back to f32
    let t0_arr: [f64; 2] = t0.into();
    let t1_arr: [f64; 2] = t1.into();
    let t2_arr: [f64; 2] = t2.into();
    let t3_arr: [f64; 2] = t3.into();
    let t4_arr: [f64; 2] = t4.into();
    let t5_arr: [f64; 2] = t5.into();
    let t6_arr: [f64; 2] = t6.into();
    let t7_arr: [f64; 2] = t7.into();
    f32x16::new([
        t0_arr[0] as f32,
        t0_arr[1] as f32,
        t1_arr[0] as f32,
        t1_arr[1] as f32,
        t2_arr[0] as f32,
        t2_arr[1] as f32,
        t3_arr[0] as f32,
        t3_arr[1] as f32,
        t4_arr[0] as f32,
        t4_arr[1] as f32,
        t5_arr[0] as f32,
        t5_arr[1] as f32,
        t6_arr[0] as f32,
        t6_arr[1] as f32,
        t7_arr[0] as f32,
        t7_arr[1] as f32,
    ])
}

/// Compute initial approximation for cube root using bit manipulation.
/// This is the scalar version used to seed the Newton iterations.
#[inline]
fn initial_approx(x: f32) -> f32 {
    // B1 = (127-127.0/3-0.03306235651)*2**23
    const B1: u32 = 709_958_130;

    let ui: u32 = x.to_bits();
    let sign = ui & 0x8000_0000;
    let hx = ui & 0x7FFF_FFFF;
    let approx = hx / 3 + B1;
    f32::from_bits(sign | approx)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference scalar implementation - the FreeBSD algorithm with 2 Newton iterations.
    // This is identical to cbrtf_fast in cbrtf.rs. SIMD versions must match this exactly.
    fn scalar_cbrtf_fast(x: f32) -> f32 {
        const B1: u32 = 709_958_130; // (127 - 127.0/3 - 0.03306235651) * 2^23

        let mut ui: u32 = x.to_bits();
        let mut hx: u32 = ui & 0x7FFF_FFFF;
        hx = hx / 3 + B1;
        ui &= 0x8000_0000;
        ui |= hx;

        let mut t: f64 = f64::from(f32::from_bits(ui));
        let xf64 = f64::from(x);

        // Newton iteration 1
        let mut r = t * t * t;
        t = t * (xf64 + xf64 + r) / (xf64 + r + r);

        // Newton iteration 2
        r = t * t * t;
        t = t * (xf64 + xf64 + r) / (xf64 + r + r);

        t as f32
    }

    /// SIMD must produce bit-identical results to scalar_cbrtf_fast.
    /// Tests ~16M values (every 128th bit pattern) across all f32 exponents.
    #[test]
    fn test_simd_matches_scalar_cbrtf_fast() {
        const STEP: u32 = 128;

        // Positive normals: 0x00800000 to 0x7F7FFFFF
        let mut bits = 0x0080_0000u32;
        while bits < 0x7F80_0000 {
            let x = f32::from_bits(bits);
            let scalar = scalar_cbrtf_fast(x);

            let r4: [f32; 4] = cbrtf_x4(f32x4::splat(x)).into();
            let r8: [f32; 8] = cbrtf_x8(f32x8::splat(x)).into();
            let r16: [f32; 16] = cbrtf_x16(f32x16::splat(x)).into();

            assert_eq!(
                r4[0].to_bits(),
                scalar.to_bits(),
                "cbrtf_x4 mismatch at x={x:e}: got {}, expected {}",
                r4[0],
                scalar
            );
            assert_eq!(
                r8[0].to_bits(),
                scalar.to_bits(),
                "cbrtf_x8 mismatch at x={x:e}: got {}, expected {}",
                r8[0],
                scalar
            );
            assert_eq!(
                r16[0].to_bits(),
                scalar.to_bits(),
                "cbrtf_x16 mismatch at x={x:e}: got {}, expected {}",
                r16[0],
                scalar
            );

            bits = bits.saturating_add(STEP);
        }

        // Negative values
        let mut bits = 0x8080_0000u32;
        while bits < 0xFF80_0000 {
            let x = f32::from_bits(bits);
            let scalar = scalar_cbrtf_fast(x);

            let r4: [f32; 4] = cbrtf_x4(f32x4::splat(x)).into();
            assert_eq!(
                r4[0].to_bits(),
                scalar.to_bits(),
                "cbrtf_x4 mismatch at x={x:e}: got {}, expected {}",
                r4[0],
                scalar
            );

            bits = bits.saturating_add(STEP);
        }
    }

    /// Verify scalar_cbrtf_fast accuracy vs std::cbrt (the "gold standard").
    /// This documents the algorithm's accuracy, not SIMD-specific behavior.
    #[test]
    fn test_scalar_cbrtf_fast_accuracy_vs_std() {
        const STEP: u32 = 128;
        let mut max_ulp = 0i32;
        let mut worst_x = 0.0f32;

        let mut bits = 0x0080_0000u32;
        while bits < 0x7F80_0000 {
            let x = f32::from_bits(bits);
            let fast = scalar_cbrtf_fast(x);
            let std = x.cbrt();

            let ulp = (fast.to_bits() as i32 - std.to_bits() as i32).abs();
            if ulp > max_ulp {
                max_ulp = ulp;
                worst_x = x;
            }

            bits = bits.saturating_add(STEP);
        }

        // Document actual accuracy. On glibc/Linux this is typically 0 ULP
        // because glibc uses the same FreeBSD algorithm.
        assert!(
            max_ulp <= 1,
            "scalar_cbrtf_fast vs std::cbrt: max {max_ulp} ULP at x={worst_x:e}"
        );
    }
}
