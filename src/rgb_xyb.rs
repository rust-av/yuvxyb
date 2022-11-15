#![allow(clippy::many_single_char_names)]

use crate::fastmath::cbrtf;

const K_M02: f32 = 0.078f32;
const K_M00: f32 = 0.30f32;
const K_M01: f32 = 1.0f32 - K_M02 - K_M00;

const K_M12: f32 = 0.078f32;
const K_M10: f32 = 0.23f32;
const K_M11: f32 = 1.0f32 - K_M12 - K_M10;

const K_M20: f32 = 0.243_422_69_f32;
const K_M21: f32 = 0.204_767_45_f32;
const K_M22: f32 = 1.0f32 - K_M20 - K_M21;

const K_B0: f32 = 0.003_793_073_4_f32;
const K_B1: f32 = K_B0;
const K_B2: f32 = K_B0;

const OPSIN_ABSORBANCE_MATRIX: [f32; 9] = [
    K_M00, K_M01, K_M02, K_M10, K_M11, K_M12, K_M20, K_M21, K_M22,
];
const OPSIN_ABSORBANCE_BIAS: [f32; 3] = [K_B0, K_B1, K_B2];
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
const NEG_OPSIN_ABSORBANCE_BIAS: [f32; 3] = [-K_B0, -K_B1, -K_B2];

/// Converts 32-bit floating point linear RGB to XYB. This function does assume
/// that the input is Linear RGB. If you pass it gamma-encoded RGB, the results
/// will be incorrect.
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

/// Converts 32-bit floating point XYB to Linear RGB. This does not perform
/// gamma encoding on the resulting RGB.
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

#[inline]
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

#[inline]
fn mixed_to_xyb(mixed: &[f32; 3]) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    out[0] = 0.5f32 * (mixed[0] - mixed[1]);
    out[1] = 0.5f32 * (mixed[0] + mixed[1]);
    out[2] = mixed[2];
    out
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::{Path, PathBuf},
    };

    use av_data::pixel::{ColorPrimaries, TransferCharacteristic};

    use crate::{Rgb, Xyb};

    fn parse_xyb_txt(path: &Path) -> Vec<[f32; 3]> {
        let input = fs::read_to_string(path).unwrap();
        input
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty())
            .map(|l| {
                let (x, l) = l.split_once(' ').unwrap();
                let (y, b) = l.split_once(' ').unwrap();
                [x.parse().unwrap(), y.parse().unwrap(), b.parse().unwrap()]
            })
            .collect()
    }

    #[test]
    fn rgb_to_xyb_correct() {
        let source_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_data")
            .join("tank_srgb.png");
        let expected_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_data")
            .join("tank_xyb.txt");
        let source = image::open(&source_path).unwrap();
        let source_data = source
            .to_rgb32f()
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect::<Vec<_>>();
        let source = Rgb::new(
            source_data,
            1448,
            1080,
            TransferCharacteristic::SRGB,
            ColorPrimaries::BT709,
        )
        .unwrap();
        let expected_data = parse_xyb_txt(&expected_path);

        let result = Xyb::try_from(source).unwrap();
        for (exp, res) in expected_data.into_iter().zip(result.data()) {
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
    fn xyb_to_rgb_correct() {
        let source_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_data")
            .join("tank_xyb.txt");
        let expected_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_data")
            .join("tank_srgb.png");
        let source_data = parse_xyb_txt(&source_path);
        let source = Xyb::new(source_data, 1448, 1080).unwrap();
        let expected = image::open(&expected_path).unwrap();
        let expected_data = expected
            .to_rgb32f()
            .chunks_exact(3)
            .map(|chunk| [chunk[0], chunk[1], chunk[2]])
            .collect::<Vec<_>>();

        let result =
            Rgb::try_from((source, TransferCharacteristic::SRGB, ColorPrimaries::BT709)).unwrap();
        for (exp, res) in expected_data.into_iter().zip(result.data()) {
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
