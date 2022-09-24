//! Conversion from YUV to XYB requires a few steps. The most direct path is
//! YUV -> sRGB -> Linear RGB -> XYZ -> LMS -> XYB. We can fuse the matrices for
//! XYZ, LMS, and XYB to decrease the number of steps to just YUV -> sRGB ->
//! Linear RGB -> XYB.

#![allow(clippy::many_single_char_names)]

use nalgebra::{Matrix3, Matrix3x1};
use once_cell::sync::OnceCell;

// These are each provided in row-major order
const LINEAR_SRGB_TO_XYZ_MATRIX: [f32; 9] = [
    0.4124, 0.3576, 0.1805, 0.2126, 0.7152, 0.0722, 0.0193, 0.1192, 0.9505,
];
const XYZ_TO_LMS_MATRIX: [f32; 9] = [
    0.240_576,
    0.855_098,
    -0.039_698_3,
    -0.417_076,
    1.177_26,
    0.0786_283,
    0.0,
    0.0,
    0.516_835,
];
const LMS_TO_XYB_MATRIX: [f32; 9] = [1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0];

static LINEAR_SRGB_TO_XYB_MATRIX: OnceCell<Matrix3<f32>> = OnceCell::new();
static XYB_TO_LINEAR_SRGB_MATRIX: OnceCell<Matrix3<f32>> = OnceCell::new();

fn linear_srgb_to_xyb_matrix() -> &'static Matrix3<f32> {
    LINEAR_SRGB_TO_XYB_MATRIX.get_or_init(|| {
        let srgb_to_xyz = Matrix3::from_row_slice(&LINEAR_SRGB_TO_XYZ_MATRIX);
        let xyz_to_lms = Matrix3::from_row_slice(&XYZ_TO_LMS_MATRIX);
        let lms_to_xyb = Matrix3::from_row_slice(&LMS_TO_XYB_MATRIX);

        srgb_to_xyz * xyz_to_lms * lms_to_xyb
    })
}

fn xyb_to_linear_srgb_matrix() -> &'static Matrix3<f32> {
    XYB_TO_LINEAR_SRGB_MATRIX.get_or_init(|| {
        linear_srgb_to_xyb_matrix()
            .try_inverse()
            .expect("This matrix has an inverse")
    })
}

/// Converts 32-bit floating point linear RGB to XYB. This function does assume
/// that the input is Linear RGB. If you pass it gamma-encoded RGB, the results
/// will be incorrect.
#[must_use]
pub fn linear_rgb_to_xyb(input: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let transform = linear_srgb_to_xyb_matrix();
    input
        .iter()
        .map(|pix| {
            let pix = Matrix3x1::from_column_slice(pix);
            let res = transform * pix;
            [res[0], res[1], res[2]]
        })
        .collect()
}

/// Converts 32-bit floating point XYB to Linear RGB. This does not perform
/// gamma encoding on the resulting RGB.
#[must_use]
pub fn xyb_to_linear_rgb(input: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let transform = xyb_to_linear_srgb_matrix();
    input
        .iter()
        .map(|pix| {
            let pix = Matrix3x1::from_column_slice(pix);
            let res = transform * pix;
            [res[0], res[1], res[2]]
        })
        .collect()
}
