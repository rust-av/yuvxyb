//! Conversion from YUV to XYB requires a few steps. The most direct path is
//! YUV -> sRGB -> Linear RGB -> XYZ -> LMS -> XYB.

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

fn transform_linear_rgb_to_xyb_pixel(pix: &[f32; 3]) -> [f32; 3] {
    let rgb = Matrix3x1::from_column_slice(pix);
    let xyz = linear_srgb_to_xyz_matrix() * rgb;
    let lms = xyz_to_lms_matrix() * xyz;
    let xyb = lms_to_xyb_matrix() * lms;
    [xyb[0], xyb[1], xyb[2]]
}

fn transform_xyb_to_linear_rgb_pixel(pix: &[f32; 3]) -> [f32; 3] {
    let xyb = Matrix3x1::from_column_slice(pix);
    let lms = xyb_to_lms_matrix() * xyb;
    let xyz = lms_to_xyz_matrix() * lms;
    let rgb = xyz_to_linear_srgb_matrix() * xyz;
    [rgb[0], rgb[1], rgb[2]]
}

#[cfg(test)]
mod tests {
    use std::{fs::File, path::PathBuf};

    use pxm::PFM;

    use super::*;

    #[test]
    fn linear_rgb_to_xyb_correct() {
        let source_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_data")
            .join("tank_linear_rgb.pfm");
        let expected_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_data")
            .join("tank_xyb.pfm");
        let source = PFM::read_from(&mut File::open(source_path).unwrap()).unwrap();
        let expected = PFM::read_from(&mut File::open(expected_path).unwrap()).unwrap();
        let source = source
            .data
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect::<Vec<_>>();
        let expected = expected
            .data
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect::<Vec<_>>();

        let result = linear_rgb_to_xyb(&source);
        for (exp, res) in expected.into_iter().zip(result.into_iter()) {
            assert!(
                (exp[0] - res[0]).abs() < 0.0005,
                "Difference in X channel: Expected {:.4}, got {:.4}",
                exp[0],
                res[0]
            );
            assert!(
                (exp[1] - res[1]).abs() < 0.0005,
                "Difference in Y channel: Expected {:.4}, got {:.4}",
                exp[1],
                res[1]
            );
            assert!(
                (exp[2] - res[2]).abs() < 0.0005,
                "Difference in B channel: Expected {:.4}, got {:.4}",
                exp[2],
                res[2]
            );
        }
    }

    #[test]
    fn xyb_to_linear_rgb_correct() {
        let source_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_data")
            .join("tank_xyb.pfm");
        let expected_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_data")
            .join("tank_linear_rgb.pfm");
        let source = PFM::read_from(&mut File::open(source_path).unwrap()).unwrap();
        let expected = PFM::read_from(&mut File::open(expected_path).unwrap()).unwrap();
        let source = source
            .data
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect::<Vec<_>>();
        let expected = expected
            .data
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect::<Vec<_>>();

        let result = xyb_to_linear_rgb(&source);
        for (exp, res) in expected.into_iter().zip(result.into_iter()) {
            assert!(
                (exp[0] - res[0]).abs() < 0.0005,
                "Difference in R channel: Expected {:.4}, got {:.4}",
                exp[0],
                res[0]
            );
            assert!(
                (exp[1] - res[1]).abs() < 0.0005,
                "Difference in G channel: Expected {:.4}, got {:.4}",
                exp[1],
                res[1]
            );
            assert!(
                (exp[2] - res[2]).abs() < 0.0005,
                "Difference in B channel: Expected {:.4}, got {:.4}",
                exp[2],
                res[2]
            );
        }
    }
}
