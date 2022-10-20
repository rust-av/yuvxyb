//! Conversion from YUV to XYB requires a few steps. The most direct path is
//! YUV -> sRGB -> Linear RGB -> XYZ -> LMS -> XYB.

#![allow(clippy::many_single_char_names)]

use nalgebra::{Matrix3, Matrix3x1};
use num_traits::clamp;

// These are each provided in row-major order
const XYZ_TO_LINEAR_SRGB_MATRIX: [f32; 9] = [
    0.7328, 0.4296, -0.1624, -0.7036, 1.6975, 0.0061, 0.0030, 0.0136, 0.9834,
];
const XYZ_TO_LMS_MATRIX: [f32; 9] = [
    0.4002, 0.7076, -0.0808, -0.2263, 1.1653, 0.0457, 0.0, 0.0, 0.9182,
];
const LMS_TO_XYB_MATRIX: [f32; 9] = [1.0, -1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0];

fn linear_srgb_to_xyz_matrix() -> Matrix3<f32> {
    xyz_to_linear_srgb_matrix()
        .try_inverse()
        .expect("has inverse")
}

fn xyz_to_lms_matrix() -> Matrix3<f32> {
    Matrix3::from_row_slice(&XYZ_TO_LMS_MATRIX)
}

fn lms_to_xyb_matrix() -> Matrix3<f32> {
    Matrix3::from_row_slice(&LMS_TO_XYB_MATRIX)
}

fn xyb_to_lms_matrix() -> Matrix3<f32> {
    lms_to_xyb_matrix().try_inverse().expect("has inverse")
}

fn lms_to_xyz_matrix() -> Matrix3<f32> {
    xyz_to_lms_matrix().try_inverse().expect("has inverse")
}

fn xyz_to_linear_srgb_matrix() -> Matrix3<f32> {
    Matrix3::from_row_slice(&XYZ_TO_LINEAR_SRGB_MATRIX)
}

const PLANE_X_MIN: f32 = -0.0979;
const PLANE_X_MAX: f32 = 0.1799;
const PLANE_Y_MIN: f32 = 0.0;
const PLANE_Y_MAX: f32 = 6.1848;
const PLANE_B_MIN: f32 = 0.0;
const PLANE_B_MAX: f32 = 6.1808;

/// Converts 32-bit floating point linear RGB to XYB. This function does assume
/// that the input is Linear RGB. If you pass it gamma-encoded RGB, the results
/// will be incorrect.
#[must_use]
pub fn linear_rgb_to_xyb(input: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let rgb_to_xyz = linear_srgb_to_xyz_matrix();
    let xyz_to_lms = xyz_to_lms_matrix();
    let lms_to_xyb = lms_to_xyb_matrix();
    let result: Vec<[f32; 3]> = input
        .iter()
        .map(|pix| {
            let rgb = Matrix3x1::from_column_slice(pix);
            let xyz = rgb_to_xyz * rgb;
            let lms = xyz_to_lms * xyz;
            let xyb = lms_to_xyb * lms;
            [
                clamp(xyb[0], PLANE_X_MIN, PLANE_X_MAX),
                clamp(xyb[1], PLANE_Y_MIN, PLANE_Y_MAX),
                clamp(xyb[2], PLANE_B_MIN, PLANE_B_MAX),
            ]
        })
        .collect();

    result
}

/// Converts 32-bit floating point XYB to Linear RGB. This does not perform
/// gamma encoding on the resulting RGB.
#[must_use]
pub fn xyb_to_linear_rgb(input: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let xyb_to_lms = xyb_to_lms_matrix();
    let lms_to_xyz = lms_to_xyz_matrix();
    let xyz_to_rgb = xyz_to_linear_srgb_matrix();
    let result: Vec<[f32; 3]> = input
        .iter()
        .map(|xyb| {
            let xyb = Matrix3x1::from_column_slice(xyb);
            let lms = xyb_to_lms * xyb;
            let xyz = lms_to_xyz * lms;
            let rgb = xyz_to_rgb * xyz;
            [rgb[0], rgb[1], rgb[2]]
        })
        .collect();

    result
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
        let source_data = source
            .data
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect::<Vec<_>>();
        let expected_data = expected
            .data
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect::<Vec<_>>();

        let result = linear_rgb_to_xyb(&source_data);
        for (exp, res) in expected_data.into_iter().zip(result.into_iter()) {
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
        let source_data = source
            .data
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect::<Vec<_>>();
        let expected_data = expected
            .data
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect::<Vec<_>>();

        let result = xyb_to_linear_rgb(&source_data);
        for (exp, res) in expected_data.into_iter().zip(result.into_iter()) {
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
