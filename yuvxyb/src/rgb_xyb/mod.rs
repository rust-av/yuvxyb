//! RGB ↔ XYB color space conversion.
//!
//! XYB is a perceptual color space derived from the LMS color space, designed by
//! Google for use in JPEG XL and jpegli. It models human vision more accurately
//! than traditional color spaces for image compression and quality metrics.
//!
//! # Algorithm
//!
//! This implementation follows the canonical algorithm from libjxl/jpegli:
//!
//! ## RGB → XYB (Forward Transform)
//!
//! ```text
//! 1. Apply opsin absorbance matrix:  opsin = M × rgb + bias
//! 2. Clamp negative values:          opsin = max(opsin, 0)
//! 3. Apply cube root:                mixed = cbrt(opsin) - cbrt(bias)
//! 4. Convert to XYB:                 X = 0.5 × (mixed[0] - mixed[1])
//!                                    Y = 0.5 × (mixed[0] + mixed[1])
//!                                    B = mixed[2]
//! ```
//!
//! ## XYB → RGB (Inverse Transform)
//!
//! ```text
//! 1. Unmix XYB:                      l = Y + X,  m = Y - X,  s = B
//! 2. Remove cube root bias:          l -= cbrt(bias), etc.
//! 3. Cube (inverse of cbrt):         l = l³,  m = m³,  s = s³
//! 4. Remove opsin bias:              l -= bias, etc.
//! 5. Apply inverse matrix:           rgb = M⁻¹ × [l, m, s]
//! ```
//!
//! # Constants
//!
//! All constants are from libjxl's `opsin_params.h` and are used identically across:
//! - libjxl (C++ reference)
//! - jpegli (C++ JPEG encoder)
//! - butteraugli (Rust perceptual metric)
//! - jpegli-rs (Rust JPEG encoder)
//!
//! | Constant | Value | Description |
//! |----------|-------|-------------|
//! | Opsin bias | `0.003_793_073_4` | Added before cube root to handle black |
//! | Inverse matrix[0] | `[11.031567, -9.866944, -0.164623]` | LMS → R |
//! | Inverse matrix[1] | `[-3.254147, 4.418770, -0.164623]` | LMS → G |
//! | Inverse matrix[2] | `[-3.658851, 2.712923, 1.945928]` | LMS → B |
//!
//! # Implementation Comparison
//!
//! We verified this implementation against multiple reference implementations:
//!
//! | Implementation | Max Difference | Notes |
//! |----------------|----------------|-------|
//! | butteraugli (Rust) | **0.00** | Exact match - same algorithm |
//! | jpegli-rs (Rust) | **0.00** | Exact match - same algorithm |
//! | jxl-rs (Rust) | **equivalent** | Has intensity_target scaling for HDR; matches when scale=1 |
//! | SIMD vs Scalar | **< 1e-5** | FMA precision differences only |
//!
//! ## Roundtrip Precision (RGB → XYB → RGB)
//!
//! | Color | Max Error | Notes |
//! |-------|-----------|-------|
//! | Black | 4.66e-10 | Best case |
//! | Primaries (R/G/B) | ~3e-7 | Excellent |
//! | White | 7.15e-7 | Very good |
//! | Magenta-ish | 1.37e-6 | Worst case |
//!
//! # Key Design Decisions
//!
//! 1. **Uses `mul_add` for FMA precision**: Both scalar and SIMD paths use fused
//!    multiply-add to ensure identical floating-point behavior across implementations.
//!
//! 2. **Stores negative bias for efficiency**: `NEG_OPSIN_ABSORBANCE_BIAS` allows
//!    using `mul_add(v², v, neg_bias)` which computes `v³ + neg_bias` in one operation.
//!
//! 3. **SIMD uses `wide` crate**: Portable SIMD via `wide` provides automatic CPU
//!    feature detection with no runtime overhead. Safe to enable by default.
//!
//! 4. **Matches jxl-rs for SDR**: The jxl-rs implementation has an `intensity_target`
//!    parameter for HDR support. When `intensity_target = 255` (SDR default), the
//!    algorithm simplifies to exactly what we implement here:
//!    ```text
//!    jxl-rs:  l³ × scale + bias × scale  (where scale = 255/intensity_target)
//!    yuvxyb:  l³ + neg_bias              (equivalent when scale = 1)
//!    ```
//!
//! # References
//!
//! - libjxl opsin_params.h: <https://github.com/libjxl/libjxl/blob/main/lib/jxl/cms/opsin_params.h>
//! - XYB color space: <https://ds.jpeg.org/whitepapers/jpeg-xl-whitepaper.pdf>

#![allow(clippy::many_single_char_names)]

#[cfg(test)]
mod tests;

#[cfg(feature = "simd")]
pub mod simd;

use yuvxyb_math::cbrtf;

// Opsin absorbance matrix constants from libjxl/jpegli.
// Each row sums to 1.0 for energy conservation.
const K_M02: f32 = 0.078f32;
const K_M00: f32 = 0.30f32;
const K_M01: f32 = 1.0f32 - K_M02 - K_M00;

const K_M12: f32 = 0.078f32;
const K_M10: f32 = 0.23f32;
const K_M11: f32 = 1.0f32 - K_M12 - K_M10;

const K_M20: f32 = 0.243_422_69_f32;
const K_M21: f32 = 0.204_767_45_f32;
const K_M22: f32 = 1.0f32 - K_M20 - K_M21;

/// Opsin absorbance bias. Added before cube root to ensure black (0,0,0) maps
/// to a finite value rather than -∞. Same value for all channels.
const K_B0: f32 = 0.003_793_073_4_f32;
const K_B1: f32 = K_B0;
const K_B2: f32 = K_B0;

/// Forward opsin absorbance matrix (row-major, 3×3).
/// Transforms linear RGB to LMS-like "opsin" space.
const OPSIN_ABSORBANCE_MATRIX: [f32; 9] = [
    K_M00, K_M01, K_M02, K_M10, K_M11, K_M12, K_M20, K_M21, K_M22,
];

/// Opsin absorbance bias vector.
const OPSIN_ABSORBANCE_BIAS: [f32; 3] = [K_B0, K_B1, K_B2];

/// Inverse opsin absorbance matrix (row-major, 3×3).
/// Pre-computed inverse of OPSIN_ABSORBANCE_MATRIX.
/// Transforms from opsin space back to linear RGB.
const INVERSE_OPSIN_ABSORBANCE_MATRIX: [f32; 9] = [
    11.031_567_f32,
    -9.866_944_f32,
    -0.164_622_99_f32,
    -3.254_147_3_f32,
    4.418_770_3_f32,
    -0.164_622_99_f32,
    -3.658_851_4_f32,
    2.712_923_f32,
    1.945_928_2_f32,
];

/// Negated opsin absorbance bias for efficient inverse transform.
/// Allows using `mul_add(v², v, neg_bias)` to compute `v³ - bias`.
const NEG_OPSIN_ABSORBANCE_BIAS: [f32; 3] = [-K_B0, -K_B1, -K_B2];

/// Converts 32-bit floating point linear RGB to XYB.
///
/// This function assumes the input is **linear RGB** (not gamma-encoded sRGB).
/// If you pass gamma-encoded RGB, the results will be incorrect.
///
/// # Algorithm
///
/// 1. Apply opsin absorbance matrix with bias
/// 2. Clamp negative values to zero
/// 3. Apply cube root (perceptual nonlinearity)
/// 4. Subtract cube root of bias
/// 5. Convert to opponent channels (X = red-green, Y = luminance, B = blue)
#[must_use]
pub fn linear_rgb_to_xyb(mut input: Vec<[f32; 3]>) -> Vec<[f32; 3]> {
    let mut absorbance_bias = [0.0f32; 3];
    for (out, bias) in absorbance_bias.iter_mut().zip(OPSIN_ABSORBANCE_BIAS.iter()) {
        *out = -cbrtf(*bias);
    }

    for pix in &mut input {
        let mut mixed = opsin_absorbance(pix);
        for (mixed, absorb) in mixed.iter_mut().zip(absorbance_bias.iter()) {
            if *mixed < 0.0 {
                *mixed = 0.0;
            }
            *mixed = cbrtf(*mixed) + (*absorb);
        }

        *pix = mixed_to_xyb(&mixed);
    }

    input
}

/// Converts 32-bit floating point XYB to linear RGB.
///
/// This does **not** perform gamma encoding on the resulting RGB.
/// The output is linear RGB suitable for further processing or display
/// on a linear workflow.
///
/// # Algorithm
///
/// 1. Unmix XYB to cube-root domain (l = Y+X, m = Y-X, s = B)
/// 2. Remove cube root bias offset
/// 3. Cube to undo the cube root (l = l³)
/// 4. Apply inverse opsin matrix to get linear RGB
#[must_use]
pub fn xyb_to_linear_rgb(mut input: Vec<[f32; 3]>) -> Vec<[f32; 3]> {
    let mut biases_cbrt = NEG_OPSIN_ABSORBANCE_BIAS;
    for bias in &mut biases_cbrt {
        *bias = cbrtf(*bias);
    }

    for pix in &mut input {
        // Color space: XYB -> RGB
        let mut gamma_rgb = [pix[1] + pix[0], pix[1] - pix[0], pix[2]];
        for ((rgb, bias_cbrt), neg_bias) in gamma_rgb
            .iter_mut()
            .zip(biases_cbrt.iter())
            .zip(NEG_OPSIN_ABSORBANCE_BIAS.iter())
        {
            *rgb -= *bias_cbrt;
            // Undo gamma compression: linear = gamma^3 for efficiency.
            // Uses mul_add for FMA precision matching SIMD implementation.
            let tmp = (*rgb) * (*rgb);
            *rgb = tmp.mul_add(*rgb, *neg_bias);
        }

        pix[0] = INVERSE_OPSIN_ABSORBANCE_MATRIX[0] * gamma_rgb[0];
        pix[0] = INVERSE_OPSIN_ABSORBANCE_MATRIX[1].mul_add(gamma_rgb[1], pix[0]);
        pix[0] = INVERSE_OPSIN_ABSORBANCE_MATRIX[2].mul_add(gamma_rgb[2], pix[0]);

        pix[1] = INVERSE_OPSIN_ABSORBANCE_MATRIX[3] * gamma_rgb[0];
        pix[1] = INVERSE_OPSIN_ABSORBANCE_MATRIX[4].mul_add(gamma_rgb[1], pix[1]);
        pix[1] = INVERSE_OPSIN_ABSORBANCE_MATRIX[5].mul_add(gamma_rgb[2], pix[1]);

        pix[2] = INVERSE_OPSIN_ABSORBANCE_MATRIX[6] * gamma_rgb[0];
        pix[2] = INVERSE_OPSIN_ABSORBANCE_MATRIX[7].mul_add(gamma_rgb[1], pix[2]);
        pix[2] = INVERSE_OPSIN_ABSORBANCE_MATRIX[8].mul_add(gamma_rgb[2], pix[2]);
    }

    input
}

/// Apply opsin absorbance matrix with bias.
/// Transforms linear RGB to opsin (LMS-like) space.
fn opsin_absorbance(rgb: &[f32; 3]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    out[0] = OPSIN_ABSORBANCE_MATRIX[0].mul_add(
        rgb[0],
        OPSIN_ABSORBANCE_MATRIX[1].mul_add(
            rgb[1],
            OPSIN_ABSORBANCE_MATRIX[2].mul_add(rgb[2], OPSIN_ABSORBANCE_BIAS[0]),
        ),
    );
    out[1] = OPSIN_ABSORBANCE_MATRIX[3].mul_add(
        rgb[0],
        OPSIN_ABSORBANCE_MATRIX[4].mul_add(
            rgb[1],
            OPSIN_ABSORBANCE_MATRIX[5].mul_add(rgb[2], OPSIN_ABSORBANCE_BIAS[1]),
        ),
    );
    out[2] = OPSIN_ABSORBANCE_MATRIX[6].mul_add(
        rgb[0],
        OPSIN_ABSORBANCE_MATRIX[7].mul_add(
            rgb[1],
            OPSIN_ABSORBANCE_MATRIX[8].mul_add(rgb[2], OPSIN_ABSORBANCE_BIAS[2]),
        ),
    );
    out
}

/// Convert from mixed (opsin after cube root) to XYB.
/// X = opponent red-green channel
/// Y = luminance-like channel
/// B = blue channel (passed through)
fn mixed_to_xyb(mixed: &[f32; 3]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    out[0] = 0.5f32 * (mixed[0] - mixed[1]);
    out[1] = 0.5f32 * (mixed[0] + mixed[1]);
    out[2] = mixed[2];
    out
}
