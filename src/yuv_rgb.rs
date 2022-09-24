//! Conversion from YUV to XYB requires a few steps. The most direct path is
//! YUV -> sRGB -> Linear RGB -> XYZ -> LMS -> XYB. We can fuse the matrices for
//! XYZ, LMS, and XYB to decrease the number of steps to just YUV -> sRGB ->
//! Linear RGB -> XYB.

#![allow(clippy::many_single_char_names)]

mod color;
mod transfer;

use std::mem::size_of;

use anyhow::Result;
use nalgebra::Matrix3x1;
use num_traits::clamp;

use self::transfer::TransferFunction;
use crate::{Yuv, YuvConfig, YuvPixel};

/// Converts 8..=16-bit YUV data to 32-bit floating point Linear RGB
/// in a range of 0.0..=1.0;
pub fn yuv_to_linear_rgb<T: YuvPixel>(input: &Yuv<T>) -> Result<Vec<[f32; 3]>> {
    let transform = get_yuv_to_rgb_matrix(input.config())?;
    let data = to_yuv444f32(input);
    let rgb = data
        .into_iter()
        .map(|pix| {
            let pix = Matrix3x1::from_column_slice(&pix);
            let res = transform * pix;
            [res[0], res[1], res[2]]
        })
        .collect::<Vec<_>>();
    input.config().transfer_characteristics.to_linear(&rgb)
}

/// Converts 32-bit floating point Linear RGB in a range of 0.0..=1.0
/// to 8..=16-bit YUV.
///
/// # Errors
/// - If the `YuvConfig` would produce an invalid image
pub fn linear_rgb_to_yuv<T: YuvPixel>(
    input: &[[f32; 3]],
    width: u32,
    height: u32,
    config: YuvConfig,
) -> Result<Yuv<T>> {
    let rgb = config.transfer_characteristics.to_gamma(input)?;
    let transform = get_rgb_to_yuv_matrix(config)?;
    let yuv = rgb
        .iter()
        .map(|pix| {
            let pix = Matrix3x1::from_column_slice(pix);
            let res = transform * pix;
            [res[0], res[1], res[2]]
        })
        .collect::<Vec<_>>();
    Ok(from_yuv444f32(
        &yuv,
        width as usize,
        height as usize,
        config,
    ))
}

#[must_use]
fn to_yuv444f32<T: YuvPixel>(input: &Yuv<T>) -> Vec<[f32; 3]> {
    #[inline(always)]
    fn to_f32<T: YuvPixel>(val: T, bd: u8, full_range: bool, is_chroma: bool) -> f32 {
        // Converts to a float value in the range 0.0..=1.0
        let val: f32 = val.as_();
        let max_val = f32::from(u16::MAX >> (16 - bd));
        clamp(
            (if full_range {
                val
            } else {
                (if is_chroma {
                    224.0 / 255.0
                } else {
                    219.0 / 255.0
                }) * val
            }) / max_val,
            0.0,
            1.0,
        )
    }

    let w = input.width() as usize;
    let h = input.height() as usize;
    let ss_x = input.config().subsampling_x;
    let ss_y = input.config().subsampling_y;
    let bd = input.config().bit_depth;
    let full_range = input.config().full_range;

    let data = input.data();
    let mut output = vec![[0.0, 0.0, 0.0]; w * h];
    for y in 0..h {
        for x in 0..w {
            let y_pos = y * w + x;
            let uv_pos = (y >> ss_y) * (w >> ss_x) + (x >> ss_x);
            // SAFETY: The bounds of the YUV data are validated when we construct it.
            unsafe {
                *output.get_unchecked_mut(y_pos) = [
                    to_f32(*data[0].get_unchecked(y_pos), bd, full_range, false),
                    to_f32(*data[1].get_unchecked(uv_pos), bd, full_range, true),
                    to_f32(*data[2].get_unchecked(uv_pos), bd, full_range, true),
                ]
            }
        }
    }
    output
}

fn from_yuv444f32<T: YuvPixel>(
    input: &[[f32; 3]],
    width: usize,
    height: usize,
    config: YuvConfig,
) -> Yuv<T> {
    #[inline(always)]
    fn from_f32<T: YuvPixel>(val: f32, bd: u8, full_range: bool, is_chroma: bool) -> T {
        // Converts from a float value in the range 0.0..=1.0 to a valid value in the
        // requested bit depth and range
        let max_val = f32::from(u16::MAX >> (16 - bd));
        let fval = (if full_range {
            val
        } else {
            (if is_chroma {
                224.0 / 255.0
            } else {
                219.0 / 255.0
            }) * val
        }) * max_val;
        if size_of::<T>() == 1 {
            T::from_u8(fval.round() as u8).expect("This is a u8")
        } else {
            T::from_u16(fval.round() as u16).expect("This is a u16")
        }
    }

    let ss_x = config.subsampling_x;
    let ss_y = config.subsampling_y;
    let bd = config.bit_depth;
    let full_range = config.full_range;
    let chroma_size = (width >> ss_x) * (height >> ss_y);

    let mut output = [
        vec![T::zero(); width * height],
        vec![T::zero(); chroma_size],
        vec![T::zero(); chroma_size],
    ];
    let mut last_uv_pos = usize::MAX;
    for y in 0..height {
        for x in 0..width {
            let y_pos = y * width + x;
            let uv_pos = (y >> ss_y) * (width >> ss_x) + (x >> ss_x);
            // SAFETY: The bounds of the YUV data are validated when we construct it.
            unsafe {
                let pix = input.get_unchecked(y_pos);
                *output[0].get_unchecked_mut(y_pos) = from_f32(pix[0], bd, full_range, false);
                if uv_pos != last_uv_pos {
                    // Small optimization to avoid doing unnecessary calculations and writes
                    *output[1].get_unchecked_mut(uv_pos) = from_f32(pix[1], bd, full_range, true);
                    *output[2].get_unchecked_mut(uv_pos) = from_f32(pix[2], bd, full_range, true);
                    last_uv_pos = uv_pos;
                }
            }
        }
    }
    Yuv {
        data: output,
        width: width as u32,
        height: height as u32,
        config,
    }
}

#[cfg(test)]
mod tests {
    use av_data::pixel::ColorPrimaries;

    use super::*;

    #[test]
    fn bt601_full_to_rgb_8b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u8, u8, u8)> =
            vec![(168, 152, 92), (71, 57, 230), (122, 122, 79), (133, 96, 39)];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: true,
            matrix_coefficients: MatrixCoefficients::ST170M,
            transfer_characteristics: TransferCharacteristic::BT1886,
            color_primaries: ColorPrimaries::ST170M,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = bt601_full_to_rgb(&input);
        dbg!(&rgb);
        dbg!(&rgb_pixels);
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u8> = rgb_to_bt601_full(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt601_limited_to_rgb_8b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u8, u8, u8)> = vec![
            (160, 149, 97),
            (77, 67, 215),
            (121, 123, 86),
            (130, 101, 52),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: false,
            matrix_coefficients: MatrixCoefficients::ST170M,
            transfer_characteristics: TransferCharacteristic::BT1886,
            color_primaries: ColorPrimaries::ST170M,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = bt601_limited_to_rgb(&input);
        dbg!(&rgb);
        dbg!(&rgb_pixels);
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u8> = rgb_to_bt601_limited(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt601_full_to_rgb_10b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u16, u16, u16)> = vec![
            (674, 609, 367),
            (286, 226, 920),
            (491, 487, 315),
            (532, 385, 155),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 10,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: true,
            matrix_coefficients: MatrixCoefficients::ST170M,
            transfer_characteristics: TransferCharacteristic::BT1886,
            color_primaries: ColorPrimaries::ST170M,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = bt601_full_to_rgb(&input);
        dbg!(&rgb);
        dbg!(&rgb_pixels);
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u16> = rgb_to_bt601_full(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt601_limited_to_rgb_10b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u16, u16, u16)> = vec![
            (641, 595, 388),
            (309, 267, 861),
            (484, 490, 344),
            (520, 403, 206),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 10,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: false,
            matrix_coefficients: MatrixCoefficients::ST170M,
            transfer_characteristics: TransferCharacteristic::BT1886,
            color_primaries: ColorPrimaries::ST170M,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = bt601_limited_to_rgb(&input);
        dbg!(&rgb);
        dbg!(&rgb_pixels);
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u16> = rgb_to_bt601_limited(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt709_full_to_rgb_8b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u8, u8, u8)> = vec![
            (174, 131, 91),
            (73, 109, 232),
            (114, 162, 109),
            (112, 205, 109),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: true,
            matrix_coefficients: MatrixCoefficients::BT709,
            transfer_characteristics: TransferCharacteristic::BT1886,
            color_primaries: ColorPrimaries::BT709,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = bt709_full_to_rgb(&input);
        dbg!(&rgb);
        dbg!(&rgb_pixels);
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u8> = rgb_to_bt709_full(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt709_limited_to_rgb_8b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u8, u8, u8)> = vec![
            (165, 131, 95),
            (79, 112, 220),
            (114, 158, 111),
            (112, 195, 111),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: false,
            matrix_coefficients: MatrixCoefficients::BT709,
            transfer_characteristics: TransferCharacteristic::BT1886,
            color_primaries: ColorPrimaries::BT709,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = bt709_limited_to_rgb(&input);
        dbg!(&rgb);
        dbg!(&rgb_pixels);
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u8> = rgb_to_bt709_limited(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt709_full_to_rgb_10b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u16, u16, u16)> = vec![
            (698, 525, 362),
            (295, 438, 931),
            (459, 650, 435),
            (448, 820, 437),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 10,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: true,
            matrix_coefficients: MatrixCoefficients::BT709,
            transfer_characteristics: TransferCharacteristic::BT1886,
            color_primaries: ColorPrimaries::BT709,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = bt709_full_to_rgb(&input);
        dbg!(&rgb);
        dbg!(&rgb_pixels);
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u16> = rgb_to_bt709_full(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt709_limited_to_rgb_10b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u16, u16, u16)> = vec![
            (662, 523, 380),
            (316, 447, 879),
            (457, 632, 444),
            (447, 782, 446),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 10,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: false,
            matrix_coefficients: MatrixCoefficients::BT709,
            transfer_characteristics: TransferCharacteristic::BT1886,
            color_primaries: ColorPrimaries::BT709,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = bt709_limited_to_rgb(&input);
        dbg!(&rgb);
        dbg!(&rgb_pixels);
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u16> = rgb_to_bt709_limited(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt2020_full_to_rgb_8b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u8, u8, u8)> = vec![
            (170, 133, 90),
            (84, 104, 233),
            (112, 163, 109),
            (108, 205, 110),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: true,
            matrix_coefficients: MatrixCoefficients::BT2020NonConstantLuminance,
            transfer_characteristics: TransferCharacteristic::BT2020Ten,
            color_primaries: ColorPrimaries::BT2020,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = bt2020_full_to_rgb(&input);
        dbg!(&rgb);
        dbg!(&rgb_pixels);
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u8> = rgb_to_bt2020_full(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt2020_limited_to_rgb_8b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u8, u8, u8)> = vec![
            (162, 132, 95),
            (88, 107, 220),
            (112, 159, 111),
            (109, 196, 112),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 8,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: false,
            matrix_coefficients: MatrixCoefficients::BT2020NonConstantLuminance,
            transfer_characteristics: TransferCharacteristic::BT2020Ten,
            color_primaries: ColorPrimaries::BT2020,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = bt2020_limited_to_rgb(&input);
        dbg!(&rgb);
        dbg!(&rgb_pixels);
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u8> = rgb_to_bt2020_limited(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt2020_full_to_rgb_10b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u16, u16, u16)> = vec![
            (684, 533, 361),
            (336, 416, 931),
            (449, 653, 436),
            (435, 822, 440),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 10,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: true,
            matrix_coefficients: MatrixCoefficients::BT2020NonConstantLuminance,
            transfer_characteristics: TransferCharacteristic::BT2020Ten,
            color_primaries: ColorPrimaries::BT2020,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = bt2020_full_to_rgb(&input);
        dbg!(&rgb);
        dbg!(&rgb_pixels);
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u16> = rgb_to_bt2020_full(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }

    #[test]
    fn bt2020_limited_to_rgb_10b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u16, u16, u16)> = vec![
            (649, 530, 380),
            (352, 428, 879),
            (449, 635, 445),
            (437, 784, 449),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (115.0 / 255.0, 191.0 / 255.0, 180.0 / 255.0),
            (238.0 / 255.0, 28.0 / 255.0, 39.0 / 255.0),
            (84.0 / 255.0, 117.0 / 255.0, 178.0 / 255.0),
            (82.0 / 255.0, 106.0 / 255.0, 254.0 / 255.0),
        ];
        let config = YuvConfig {
            bit_depth: 10,
            subsampling_x: 0,
            subsampling_y: 0,
            full_range: false,
            matrix_coefficients: MatrixCoefficients::BT2020NonConstantLuminance,
            transfer_characteristics: TransferCharacteristic::BT2020Ten,
            color_primaries: ColorPrimaries::BT2020,
        };
        let input = Yuv::new(
            [
                yuv_pixels.iter().map(|pix| pix.0).collect(),
                yuv_pixels.iter().map(|pix| pix.1).collect(),
                yuv_pixels.iter().map(|pix| pix.2).collect(),
            ],
            2,
            2,
            config,
        )
        .unwrap();
        let rgb = bt2020_limited_to_rgb(&input);
        dbg!(&rgb);
        dbg!(&rgb_pixels);
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.005,
                "{:.4} != expected {:.4}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u16> = rgb_to_bt2020_limited(&rgb, 2, 2, config).unwrap();
        for (i, expected) in yuv_pixels.iter().enumerate() {
            assert_eq!(yuv.data()[0][i], expected.0);
            assert_eq!(yuv.data()[1][i], expected.1);
            assert_eq!(yuv.data()[2][i], expected.2);
        }
    }
}
