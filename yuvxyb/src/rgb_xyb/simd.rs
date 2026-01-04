//! SIMD-optimized RGB <-> XYB conversions.
//!
//! This module provides vectorized versions of the color space conversion
//! functions that process multiple pixels at once for improved performance.

use wide::{f32x16, f32x8};
use yuvxyb_math::{cbrtf_x16, cbrtf_x8};

use super::{
    INVERSE_OPSIN_ABSORBANCE_MATRIX, NEG_OPSIN_ABSORBANCE_BIAS, OPSIN_ABSORBANCE_BIAS,
    OPSIN_ABSORBANCE_MATRIX,
};

/// Converts 32-bit floating point linear RGB to XYB using SIMD, in place.
///
/// This processes the input in batches of 16 pixels for maximum performance,
/// falling back to scalar processing for remainders.
#[inline]
pub fn linear_rgb_to_xyb_simd(input: &mut [[f32; 3]]) {
    // Precompute the absorbance bias (negated cube root)
    let absorbance_bias: [f32; 3] = [
        -yuvxyb_math::cbrtf(OPSIN_ABSORBANCE_BIAS[0]),
        -yuvxyb_math::cbrtf(OPSIN_ABSORBANCE_BIAS[1]),
        -yuvxyb_math::cbrtf(OPSIN_ABSORBANCE_BIAS[2]),
    ];

    // Process 16 pixels at a time
    let chunks_16 = input.len() / 16;

    for chunk_idx in 0..chunks_16 {
        let base = chunk_idx * 16;

        // Load 16 pixels and transpose to SoA
        let mut r_arr = [0.0f32; 16];
        let mut g_arr = [0.0f32; 16];
        let mut b_arr = [0.0f32; 16];

        for i in 0..16 {
            let p = input[base + i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }

        let r = f32x16::new(r_arr);
        let g = f32x16::new(g_arr);
        let b = f32x16::new(b_arr);

        // Matrix multiply: mixed = M * rgb + bias
        let m00 = f32x16::splat(OPSIN_ABSORBANCE_MATRIX[0]);
        let m01 = f32x16::splat(OPSIN_ABSORBANCE_MATRIX[1]);
        let m02 = f32x16::splat(OPSIN_ABSORBANCE_MATRIX[2]);
        let m10 = f32x16::splat(OPSIN_ABSORBANCE_MATRIX[3]);
        let m11 = f32x16::splat(OPSIN_ABSORBANCE_MATRIX[4]);
        let m12 = f32x16::splat(OPSIN_ABSORBANCE_MATRIX[5]);
        let m20 = f32x16::splat(OPSIN_ABSORBANCE_MATRIX[6]);
        let m21 = f32x16::splat(OPSIN_ABSORBANCE_MATRIX[7]);
        let m22 = f32x16::splat(OPSIN_ABSORBANCE_MATRIX[8]);

        let bias0 = f32x16::splat(OPSIN_ABSORBANCE_BIAS[0]);
        let bias1 = f32x16::splat(OPSIN_ABSORBANCE_BIAS[1]);
        let bias2 = f32x16::splat(OPSIN_ABSORBANCE_BIAS[2]);

        let mut mixed0 = m00 * r + m01 * g + m02 * b + bias0;
        let mut mixed1 = m10 * r + m11 * g + m12 * b + bias1;
        let mut mixed2 = m20 * r + m21 * g + m22 * b + bias2;

        // Clamp negative values to zero
        let zero = f32x16::splat(0.0);
        mixed0 = mixed0.max(zero);
        mixed1 = mixed1.max(zero);
        mixed2 = mixed2.max(zero);

        // Apply cube root + bias offset
        let absorb0 = f32x16::splat(absorbance_bias[0]);
        let absorb1 = f32x16::splat(absorbance_bias[1]);
        let absorb2 = f32x16::splat(absorbance_bias[2]);

        mixed0 = cbrtf_x16(mixed0) + absorb0;
        mixed1 = cbrtf_x16(mixed1) + absorb1;
        mixed2 = cbrtf_x16(mixed2) + absorb2;

        // Convert mixed to XYB
        let half = f32x16::splat(0.5);
        let x = half * (mixed0 - mixed1);
        let y = half * (mixed0 + mixed1);
        let b_out = mixed2;

        // Transpose back to AoS and store
        let x_arr: [f32; 16] = x.into();
        let y_arr: [f32; 16] = y.into();
        let b_arr: [f32; 16] = b_out.into();

        for i in 0..16 {
            input[base + i] = [x_arr[i], y_arr[i], b_arr[i]];
        }
    }

    // Process remaining pixels with scalar code
    let scalar_start = chunks_16 * 16;
    for pix in &mut input[scalar_start..] {
        let mut mixed = opsin_absorbance_scalar(pix);
        for (m, absorb) in mixed.iter_mut().zip(absorbance_bias.iter()) {
            if *m < 0.0 {
                *m = 0.0;
            }
            *m = yuvxyb_math::cbrtf(*m) + *absorb;
        }
        *pix = mixed_to_xyb_scalar(&mixed);
    }
}

/// Converts 32-bit floating point XYB to Linear RGB using SIMD, in place.
///
/// This processes the input in batches of 16 pixels for maximum performance,
/// falling back to scalar processing for remainders.
#[inline]
pub fn xyb_to_linear_rgb_simd(input: &mut [[f32; 3]]) {
    // Precompute biases
    let biases_cbrt: [f32; 3] = [
        yuvxyb_math::cbrtf(NEG_OPSIN_ABSORBANCE_BIAS[0]),
        yuvxyb_math::cbrtf(NEG_OPSIN_ABSORBANCE_BIAS[1]),
        yuvxyb_math::cbrtf(NEG_OPSIN_ABSORBANCE_BIAS[2]),
    ];

    // Process 16 pixels at a time
    let chunks_16 = input.len() / 16;

    for chunk_idx in 0..chunks_16 {
        let base = chunk_idx * 16;

        // Load 16 pixels and transpose to SoA
        let mut x_arr = [0.0f32; 16];
        let mut y_arr = [0.0f32; 16];
        let mut b_arr = [0.0f32; 16];

        for i in 0..16 {
            let p = input[base + i];
            x_arr[i] = p[0];
            y_arr[i] = p[1];
            b_arr[i] = p[2];
        }

        let x = f32x16::new(x_arr);
        let y = f32x16::new(y_arr);
        let b = f32x16::new(b_arr);

        // XYB -> mixed (gamma RGB space)
        let mut gamma_r = y + x;
        let mut gamma_g = y - x;
        let mut gamma_b = b;

        // Subtract bias and cube
        let bias0 = f32x16::splat(biases_cbrt[0]);
        let bias1 = f32x16::splat(biases_cbrt[1]);
        let bias2 = f32x16::splat(biases_cbrt[2]);

        gamma_r -= bias0;
        gamma_g -= bias1;
        gamma_b -= bias2;

        // Cube: x^3 = x * x * x, then subtract neg_bias
        let neg_bias0 = f32x16::splat(NEG_OPSIN_ABSORBANCE_BIAS[0]);
        let neg_bias1 = f32x16::splat(NEG_OPSIN_ABSORBANCE_BIAS[1]);
        let neg_bias2 = f32x16::splat(NEG_OPSIN_ABSORBANCE_BIAS[2]);

        let mixed_r = gamma_r * gamma_r * gamma_r + neg_bias0;
        let mixed_g = gamma_g * gamma_g * gamma_g + neg_bias1;
        let mixed_b = gamma_b * gamma_b * gamma_b + neg_bias2;

        // Matrix multiply by inverse matrix
        let im00 = f32x16::splat(INVERSE_OPSIN_ABSORBANCE_MATRIX[0]);
        let im01 = f32x16::splat(INVERSE_OPSIN_ABSORBANCE_MATRIX[1]);
        let im02 = f32x16::splat(INVERSE_OPSIN_ABSORBANCE_MATRIX[2]);
        let im10 = f32x16::splat(INVERSE_OPSIN_ABSORBANCE_MATRIX[3]);
        let im11 = f32x16::splat(INVERSE_OPSIN_ABSORBANCE_MATRIX[4]);
        let im12 = f32x16::splat(INVERSE_OPSIN_ABSORBANCE_MATRIX[5]);
        let im20 = f32x16::splat(INVERSE_OPSIN_ABSORBANCE_MATRIX[6]);
        let im21 = f32x16::splat(INVERSE_OPSIN_ABSORBANCE_MATRIX[7]);
        let im22 = f32x16::splat(INVERSE_OPSIN_ABSORBANCE_MATRIX[8]);

        let out_r = im00 * mixed_r + im01 * mixed_g + im02 * mixed_b;
        let out_g = im10 * mixed_r + im11 * mixed_g + im12 * mixed_b;
        let out_b = im20 * mixed_r + im21 * mixed_g + im22 * mixed_b;

        // Transpose back to AoS and store
        let r_arr: [f32; 16] = out_r.into();
        let g_arr: [f32; 16] = out_g.into();
        let b_arr: [f32; 16] = out_b.into();

        for i in 0..16 {
            input[base + i] = [r_arr[i], g_arr[i], b_arr[i]];
        }
    }

    // Process remaining pixels with scalar code
    let start = chunks_16 * 16;
    for pix in &mut input[start..] {
        let mut gamma_rgb = [pix[1] + pix[0], pix[1] - pix[0], pix[2]];
        for ((rgb, bias_cbrt), neg_bias) in gamma_rgb
            .iter_mut()
            .zip(biases_cbrt.iter())
            .zip(NEG_OPSIN_ABSORBANCE_BIAS.iter())
        {
            *rgb -= *bias_cbrt;
            let tmp = (*rgb) * (*rgb);
            *rgb = tmp * (*rgb) + *neg_bias;
        }

        pix[0] = INVERSE_OPSIN_ABSORBANCE_MATRIX[0] * gamma_rgb[0]
            + INVERSE_OPSIN_ABSORBANCE_MATRIX[1] * gamma_rgb[1]
            + INVERSE_OPSIN_ABSORBANCE_MATRIX[2] * gamma_rgb[2];
        pix[1] = INVERSE_OPSIN_ABSORBANCE_MATRIX[3] * gamma_rgb[0]
            + INVERSE_OPSIN_ABSORBANCE_MATRIX[4] * gamma_rgb[1]
            + INVERSE_OPSIN_ABSORBANCE_MATRIX[5] * gamma_rgb[2];
        pix[2] = INVERSE_OPSIN_ABSORBANCE_MATRIX[6] * gamma_rgb[0]
            + INVERSE_OPSIN_ABSORBANCE_MATRIX[7] * gamma_rgb[1]
            + INVERSE_OPSIN_ABSORBANCE_MATRIX[8] * gamma_rgb[2];
    }
}

/// Converts 32-bit floating point linear RGB to XYB using f32x8 SIMD, in place.
///
/// Processes 8 pixels at a time - optimal for AVX/AVX2 hardware.
#[inline]
pub fn linear_rgb_to_xyb_simd_x8(input: &mut [[f32; 3]]) {
    let absorbance_bias: [f32; 3] = [
        -yuvxyb_math::cbrtf(OPSIN_ABSORBANCE_BIAS[0]),
        -yuvxyb_math::cbrtf(OPSIN_ABSORBANCE_BIAS[1]),
        -yuvxyb_math::cbrtf(OPSIN_ABSORBANCE_BIAS[2]),
    ];

    let chunks_8 = input.len() / 8;

    for chunk_idx in 0..chunks_8 {
        let base = chunk_idx * 8;

        // Load 8 pixels and transpose to SoA
        let mut r_arr = [0.0f32; 8];
        let mut g_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];

        for i in 0..8 {
            let p = input[base + i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }

        let r = f32x8::new(r_arr);
        let g = f32x8::new(g_arr);
        let b = f32x8::new(b_arr);

        // Matrix multiply: mixed = M * rgb + bias
        let m00 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[0]);
        let m01 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[1]);
        let m02 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[2]);
        let m10 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[3]);
        let m11 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[4]);
        let m12 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[5]);
        let m20 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[6]);
        let m21 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[7]);
        let m22 = f32x8::splat(OPSIN_ABSORBANCE_MATRIX[8]);

        let bias0 = f32x8::splat(OPSIN_ABSORBANCE_BIAS[0]);
        let bias1 = f32x8::splat(OPSIN_ABSORBANCE_BIAS[1]);
        let bias2 = f32x8::splat(OPSIN_ABSORBANCE_BIAS[2]);

        let mut mixed0 = m00 * r + m01 * g + m02 * b + bias0;
        let mut mixed1 = m10 * r + m11 * g + m12 * b + bias1;
        let mut mixed2 = m20 * r + m21 * g + m22 * b + bias2;

        // Clamp negative values to zero
        let zero = f32x8::splat(0.0);
        mixed0 = mixed0.max(zero);
        mixed1 = mixed1.max(zero);
        mixed2 = mixed2.max(zero);

        // Apply cube root + bias offset
        let absorb0 = f32x8::splat(absorbance_bias[0]);
        let absorb1 = f32x8::splat(absorbance_bias[1]);
        let absorb2 = f32x8::splat(absorbance_bias[2]);

        mixed0 = cbrtf_x8(mixed0) + absorb0;
        mixed1 = cbrtf_x8(mixed1) + absorb1;
        mixed2 = cbrtf_x8(mixed2) + absorb2;

        // Convert mixed to XYB
        let half = f32x8::splat(0.5);
        let x = half * (mixed0 - mixed1);
        let y = half * (mixed0 + mixed1);
        let b_out = mixed2;

        // Transpose back to AoS and store
        let x_arr: [f32; 8] = x.into();
        let y_arr: [f32; 8] = y.into();
        let b_arr: [f32; 8] = b_out.into();

        for i in 0..8 {
            input[base + i] = [x_arr[i], y_arr[i], b_arr[i]];
        }
    }

    // Process remaining pixels with scalar code
    let scalar_start = chunks_8 * 8;
    for pix in &mut input[scalar_start..] {
        let mut mixed = opsin_absorbance_scalar(pix);
        for (m, absorb) in mixed.iter_mut().zip(absorbance_bias.iter()) {
            if *m < 0.0 {
                *m = 0.0;
            }
            *m = yuvxyb_math::cbrtf(*m) + *absorb;
        }
        *pix = mixed_to_xyb_scalar(&mixed);
    }
}

/// Converts 32-bit floating point XYB to Linear RGB using f32x8 SIMD, in place.
///
/// Processes 8 pixels at a time - optimal for AVX/AVX2 hardware.
#[inline]
pub fn xyb_to_linear_rgb_simd_x8(input: &mut [[f32; 3]]) {
    let biases_cbrt: [f32; 3] = [
        yuvxyb_math::cbrtf(NEG_OPSIN_ABSORBANCE_BIAS[0]),
        yuvxyb_math::cbrtf(NEG_OPSIN_ABSORBANCE_BIAS[1]),
        yuvxyb_math::cbrtf(NEG_OPSIN_ABSORBANCE_BIAS[2]),
    ];

    let chunks_8 = input.len() / 8;

    for chunk_idx in 0..chunks_8 {
        let base = chunk_idx * 8;

        // Load 8 pixels and transpose to SoA
        let mut x_arr = [0.0f32; 8];
        let mut y_arr = [0.0f32; 8];
        let mut b_arr = [0.0f32; 8];

        for i in 0..8 {
            let p = input[base + i];
            x_arr[i] = p[0];
            y_arr[i] = p[1];
            b_arr[i] = p[2];
        }

        let x = f32x8::new(x_arr);
        let y = f32x8::new(y_arr);
        let b = f32x8::new(b_arr);

        // XYB -> mixed (gamma RGB space)
        let mut gamma_r = y + x;
        let mut gamma_g = y - x;
        let mut gamma_b = b;

        // Subtract bias and cube
        let bias0 = f32x8::splat(biases_cbrt[0]);
        let bias1 = f32x8::splat(biases_cbrt[1]);
        let bias2 = f32x8::splat(biases_cbrt[2]);

        gamma_r -= bias0;
        gamma_g -= bias1;
        gamma_b -= bias2;

        // Cube: x^3 = x * x * x, then subtract neg_bias
        let neg_bias0 = f32x8::splat(NEG_OPSIN_ABSORBANCE_BIAS[0]);
        let neg_bias1 = f32x8::splat(NEG_OPSIN_ABSORBANCE_BIAS[1]);
        let neg_bias2 = f32x8::splat(NEG_OPSIN_ABSORBANCE_BIAS[2]);

        let mixed_r = gamma_r * gamma_r * gamma_r + neg_bias0;
        let mixed_g = gamma_g * gamma_g * gamma_g + neg_bias1;
        let mixed_b = gamma_b * gamma_b * gamma_b + neg_bias2;

        // Matrix multiply by inverse matrix
        let im00 = f32x8::splat(INVERSE_OPSIN_ABSORBANCE_MATRIX[0]);
        let im01 = f32x8::splat(INVERSE_OPSIN_ABSORBANCE_MATRIX[1]);
        let im02 = f32x8::splat(INVERSE_OPSIN_ABSORBANCE_MATRIX[2]);
        let im10 = f32x8::splat(INVERSE_OPSIN_ABSORBANCE_MATRIX[3]);
        let im11 = f32x8::splat(INVERSE_OPSIN_ABSORBANCE_MATRIX[4]);
        let im12 = f32x8::splat(INVERSE_OPSIN_ABSORBANCE_MATRIX[5]);
        let im20 = f32x8::splat(INVERSE_OPSIN_ABSORBANCE_MATRIX[6]);
        let im21 = f32x8::splat(INVERSE_OPSIN_ABSORBANCE_MATRIX[7]);
        let im22 = f32x8::splat(INVERSE_OPSIN_ABSORBANCE_MATRIX[8]);

        let out_r = im00 * mixed_r + im01 * mixed_g + im02 * mixed_b;
        let out_g = im10 * mixed_r + im11 * mixed_g + im12 * mixed_b;
        let out_b = im20 * mixed_r + im21 * mixed_g + im22 * mixed_b;

        // Transpose back to AoS and store
        let r_arr: [f32; 8] = out_r.into();
        let g_arr: [f32; 8] = out_g.into();
        let b_arr: [f32; 8] = out_b.into();

        for i in 0..8 {
            input[base + i] = [r_arr[i], g_arr[i], b_arr[i]];
        }
    }

    // Process remaining pixels with scalar code
    let start = chunks_8 * 8;
    for pix in &mut input[start..] {
        let mut gamma_rgb = [pix[1] + pix[0], pix[1] - pix[0], pix[2]];
        for ((rgb, bias_cbrt), neg_bias) in gamma_rgb
            .iter_mut()
            .zip(biases_cbrt.iter())
            .zip(NEG_OPSIN_ABSORBANCE_BIAS.iter())
        {
            *rgb -= *bias_cbrt;
            let tmp = (*rgb) * (*rgb);
            *rgb = tmp * (*rgb) + *neg_bias;
        }

        pix[0] = INVERSE_OPSIN_ABSORBANCE_MATRIX[0] * gamma_rgb[0]
            + INVERSE_OPSIN_ABSORBANCE_MATRIX[1] * gamma_rgb[1]
            + INVERSE_OPSIN_ABSORBANCE_MATRIX[2] * gamma_rgb[2];
        pix[1] = INVERSE_OPSIN_ABSORBANCE_MATRIX[3] * gamma_rgb[0]
            + INVERSE_OPSIN_ABSORBANCE_MATRIX[4] * gamma_rgb[1]
            + INVERSE_OPSIN_ABSORBANCE_MATRIX[5] * gamma_rgb[2];
        pix[2] = INVERSE_OPSIN_ABSORBANCE_MATRIX[6] * gamma_rgb[0]
            + INVERSE_OPSIN_ABSORBANCE_MATRIX[7] * gamma_rgb[1]
            + INVERSE_OPSIN_ABSORBANCE_MATRIX[8] * gamma_rgb[2];
    }
}

// Scalar helper functions for remainder processing
#[inline]
fn opsin_absorbance_scalar(rgb: &[f32; 3]) -> [f32; 3] {
    [
        OPSIN_ABSORBANCE_MATRIX[0] * rgb[0]
            + OPSIN_ABSORBANCE_MATRIX[1] * rgb[1]
            + OPSIN_ABSORBANCE_MATRIX[2] * rgb[2]
            + OPSIN_ABSORBANCE_BIAS[0],
        OPSIN_ABSORBANCE_MATRIX[3] * rgb[0]
            + OPSIN_ABSORBANCE_MATRIX[4] * rgb[1]
            + OPSIN_ABSORBANCE_MATRIX[5] * rgb[2]
            + OPSIN_ABSORBANCE_BIAS[1],
        OPSIN_ABSORBANCE_MATRIX[6] * rgb[0]
            + OPSIN_ABSORBANCE_MATRIX[7] * rgb[1]
            + OPSIN_ABSORBANCE_MATRIX[8] * rgb[2]
            + OPSIN_ABSORBANCE_BIAS[2],
    ]
}

#[inline]
fn mixed_to_xyb_scalar(mixed: &[f32; 3]) -> [f32; 3] {
    [
        0.5 * (mixed[0] - mixed[1]),
        0.5 * (mixed[0] + mixed[1]),
        mixed[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rgb_xyb::{linear_rgb_to_xyb, xyb_to_linear_rgb};

    /// Generate test data with various patterns to exercise edge cases
    fn make_test_data(count: usize) -> Vec<[f32; 3]> {
        let mut data = Vec::with_capacity(count);
        for i in 0..count {
            let t = i as f32 / count as f32;
            // Mix of patterns: gradient, primaries, grays, random-ish
            let pixel = match i % 8 {
                0 => [t, t, t],                   // gray gradient
                1 => [t, 0.0, 0.0],               // red gradient
                2 => [0.0, t, 0.0],               // green gradient
                3 => [0.0, 0.0, t],               // blue gradient
                4 => [1.0 - t, t, 0.5],           // mixed
                5 => [t * 0.5, t * 0.3, t * 0.8], // scaled
                6 => [
                    (i as f32 * 0.123) % 1.0, // pseudo-random
                    (i as f32 * 0.456) % 1.0,
                    (i as f32 * 0.789) % 1.0,
                ],
                _ => [0.0, 0.0, 0.0], // black
            };
            data.push(pixel);
        }
        // Add edge cases
        if count > 4 {
            data[0] = [0.0, 0.0, 0.0]; // black
            data[1] = [1.0, 1.0, 1.0]; // white
            data[2] = [1.0, 0.0, 0.0]; // pure red
            data[3] = [0.0, 1.0, 0.0]; // pure green
        }
        data
    }

    fn assert_matches_scalar(scalar: &[[f32; 3]], simd: &[[f32; 3]], tolerance: f32, name: &str) {
        assert_eq!(scalar.len(), simd.len(), "{}: length mismatch", name);
        for (i, (s, v)) in scalar.iter().zip(simd.iter()).enumerate() {
            for j in 0..3 {
                let diff = (s[j] - v[j]).abs();
                assert!(
                    diff < tolerance,
                    "{}: mismatch at pixel {}, channel {}: scalar={}, simd={}, diff={}",
                    name,
                    i,
                    j,
                    s[j],
                    v[j],
                    diff
                );
            }
        }
    }

    // Test x16 with 35 pixels (2 full x16 chunks + 3 remainder via scalar)
    #[test]
    fn test_linear_rgb_to_xyb_x16_matches_scalar() {
        let test_data = make_test_data(35);
        let scalar_result = linear_rgb_to_xyb(test_data.clone());
        let mut simd_data = test_data;
        linear_rgb_to_xyb_simd(&mut simd_data);
        assert_matches_scalar(&scalar_result, &simd_data, 1e-5, "rgb_to_xyb x16");
    }

    #[test]
    fn test_xyb_to_linear_rgb_x16_matches_scalar() {
        let rgb_data = make_test_data(35);
        let xyb_data = linear_rgb_to_xyb(rgb_data);
        let scalar_result = xyb_to_linear_rgb(xyb_data.clone());
        let mut simd_data = xyb_data;
        xyb_to_linear_rgb_simd(&mut simd_data);
        assert_matches_scalar(&scalar_result, &simd_data, 1e-5, "xyb_to_rgb x16");
    }

    // Test x8 with 19 pixels (2 full x8 chunks + 3 remainder via scalar)
    #[test]
    fn test_linear_rgb_to_xyb_x8_matches_scalar() {
        let test_data = make_test_data(19);
        let scalar_result = linear_rgb_to_xyb(test_data.clone());
        let mut simd_data = test_data;
        linear_rgb_to_xyb_simd_x8(&mut simd_data);
        assert_matches_scalar(&scalar_result, &simd_data, 1e-5, "rgb_to_xyb x8");
    }

    #[test]
    fn test_xyb_to_linear_rgb_x8_matches_scalar() {
        let rgb_data = make_test_data(19);
        let xyb_data = linear_rgb_to_xyb(rgb_data);
        let scalar_result = xyb_to_linear_rgb(xyb_data.clone());
        let mut simd_data = xyb_data;
        xyb_to_linear_rgb_simd_x8(&mut simd_data);
        assert_matches_scalar(&scalar_result, &simd_data, 1e-5, "xyb_to_rgb x8");
    }

    // Test with larger data to ensure bulk processing works
    #[test]
    fn test_simd_large_data() {
        let test_data = make_test_data(1000);

        // Test x16
        let scalar_result = linear_rgb_to_xyb(test_data.clone());
        let mut simd_x16 = test_data.clone();
        linear_rgb_to_xyb_simd(&mut simd_x16);
        assert_matches_scalar(&scalar_result, &simd_x16, 1e-5, "large rgb_to_xyb x16");

        // Test x8
        let mut simd_x8 = test_data;
        linear_rgb_to_xyb_simd_x8(&mut simd_x8);
        assert_matches_scalar(&scalar_result, &simd_x8, 1e-5, "large rgb_to_xyb x8");
    }

    #[test]
    fn test_simd_roundtrip_x16() {
        let original = make_test_data(35);
        let mut data = original.clone();
        linear_rgb_to_xyb_simd(&mut data);
        xyb_to_linear_rgb_simd(&mut data);
        assert_matches_scalar(&original, &data, 1e-4, "roundtrip x16");
    }

    #[test]
    fn test_simd_roundtrip_x8() {
        let original = make_test_data(19);
        let mut data = original.clone();
        linear_rgb_to_xyb_simd_x8(&mut data);
        xyb_to_linear_rgb_simd_x8(&mut data);
        assert_matches_scalar(&original, &data, 1e-4, "roundtrip x8");
    }
}
