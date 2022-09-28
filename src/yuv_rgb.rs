//! Conversion from YUV to XYB requires a few steps. The most direct path is
//! YUV -> sRGB -> Linear RGB -> XYZ -> LMS -> XYB. We can fuse the matrices for
//! XYZ, LMS, and XYB to decrease the number of steps to just YUV -> sRGB ->
//! Linear RGB -> XYB.

#![allow(clippy::many_single_char_names)]

mod color;
mod transfer;

use std::mem::size_of;

use anyhow::{bail, Result};
use num_traits::clamp;

use self::{
    color::{rgb_to_yuv, yuv_to_rgb},
    transfer::TransferFunction,
};
use crate::{Yuv, YuvConfig, YuvPixel};

/// Converts 8..=16-bit YUV data to 32-bit floating point Linear RGB
/// in a range of 0.0..=1.0;
pub fn yuv_to_linear_rgb<T: YuvPixel>(input: &Yuv<T>) -> Result<Vec<[f32; 3]>> {
    let rgb = yuv_to_rgb(input)?;
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
    rgb_to_yuv(&rgb, width, height, config)
}

fn ycbcr_to_ypbpr<T: YuvPixel>(input: &Yuv<T>) -> Result<Vec<[f32; 3]>> {
    let w = input.width() as usize;
    let h = input.height() as usize;
    let ss_x = input.config().subsampling_x;
    let ss_y = input.config().subsampling_y;
    let bd = input.config().bit_depth;
    let full_range = input.config().full_range;

    let to_luma: &dyn Fn(T) -> f32 = match (bd, full_range) {
        (8, false) => &to_f32_luma::<T, 8, false>,
        (10, false) => &to_f32_luma::<T, 10, false>,
        (12, false) => &to_f32_luma::<T, 12, false>,
        (8, true) => &to_f32_luma::<T, 8, true>,
        (9, true) => &to_f32_luma::<T, 9, true>,
        (10, true) => &to_f32_luma::<T, 10, true>,
        (11, true) => &to_f32_luma::<T, 11, true>,
        (12, true) => &to_f32_luma::<T, 12, true>,
        (13, true) => &to_f32_luma::<T, 13, true>,
        (14, true) => &to_f32_luma::<T, 14, true>,
        (15, true) => &to_f32_luma::<T, 15, true>,
        (16, true) => &to_f32_luma::<T, 16, true>,
        _ => {
            bail!(
                "Bit depths 8, 10, and 12 for limited range or 8-16 for full range are supported"
            );
        }
    };
    let to_chroma: &dyn Fn(T) -> f32 = match (bd, full_range) {
        (8, false) => &to_f32_chroma::<T, 8, false>,
        (10, false) => &to_f32_chroma::<T, 10, false>,
        (12, false) => &to_f32_chroma::<T, 12, false>,
        (8, true) => &to_f32_chroma::<T, 8, true>,
        (9, true) => &to_f32_chroma::<T, 9, true>,
        (10, true) => &to_f32_chroma::<T, 10, true>,
        (11, true) => &to_f32_chroma::<T, 11, true>,
        (12, true) => &to_f32_chroma::<T, 12, true>,
        (13, true) => &to_f32_chroma::<T, 13, true>,
        (14, true) => &to_f32_chroma::<T, 14, true>,
        (15, true) => &to_f32_chroma::<T, 15, true>,
        (16, true) => &to_f32_chroma::<T, 16, true>,
        _ => {
            bail!(
                "Bit depths 8, 10, and 12 for limited range or 8-16 for full range are supported"
            );
        }
    };

    let data = input.data();
    let mut output = vec![[0.0, 0.0, 0.0]; w * h];
    for y in 0..h {
        for x in 0..w {
            let y_pos = y * w + x;
            let uv_pos = (y >> ss_y) * (w >> ss_x) + (x >> ss_x);
            // SAFETY: The bounds of the YUV data are validated when we construct it.
            unsafe {
                *output.get_unchecked_mut(y_pos) = [
                    to_luma(*data[0].get_unchecked(y_pos)),
                    to_chroma(*data[1].get_unchecked(uv_pos)),
                    to_chroma(*data[2].get_unchecked(uv_pos)),
                ];
            }
        }
    }
    Ok(output)
}

fn ypbpr_to_ycbcr<T: YuvPixel>(
    input: &[[f32; 3]],
    width: usize,
    height: usize,
    config: YuvConfig,
) -> Result<Yuv<T>> {
    let ss_x = config.subsampling_x;
    let ss_y = config.subsampling_y;
    let bd = config.bit_depth;
    let full_range = config.full_range;
    let chroma_size = (width >> ss_x) * (height >> ss_y);

    let from_luma: &dyn Fn(f32) -> T = match (bd, full_range) {
        (8, false) => &from_f32_luma::<T, 8, false>,
        (10, false) => &from_f32_luma::<T, 10, false>,
        (12, false) => &from_f32_luma::<T, 12, false>,
        (8, true) => &from_f32_luma::<T, 8, true>,
        (9, true) => &from_f32_luma::<T, 9, true>,
        (10, true) => &from_f32_luma::<T, 10, true>,
        (11, true) => &from_f32_luma::<T, 11, true>,
        (12, true) => &from_f32_luma::<T, 12, true>,
        (13, true) => &from_f32_luma::<T, 13, true>,
        (14, true) => &from_f32_luma::<T, 14, true>,
        (15, true) => &from_f32_luma::<T, 15, true>,
        (16, true) => &from_f32_luma::<T, 16, true>,
        _ => {
            bail!(
                "Bit depths 8, 10, and 12 for limited range or 8-16 for full range are supported"
            );
        }
    };
    let from_chroma: &dyn Fn(f32) -> T = match (bd, full_range) {
        (8, false) => &from_f32_chroma::<T, 8, false>,
        (10, false) => &from_f32_chroma::<T, 10, false>,
        (12, false) => &from_f32_chroma::<T, 12, false>,
        (8, true) => &from_f32_chroma::<T, 8, true>,
        (9, true) => &from_f32_chroma::<T, 9, true>,
        (10, true) => &from_f32_chroma::<T, 10, true>,
        (11, true) => &from_f32_chroma::<T, 11, true>,
        (12, true) => &from_f32_chroma::<T, 12, true>,
        (13, true) => &from_f32_chroma::<T, 13, true>,
        (14, true) => &from_f32_chroma::<T, 14, true>,
        (15, true) => &from_f32_chroma::<T, 15, true>,
        (16, true) => &from_f32_chroma::<T, 16, true>,
        _ => {
            bail!(
                "Bit depths 8, 10, and 12 for limited range or 8-16 for full range are supported"
            );
        }
    };

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
                *output[0].get_unchecked_mut(y_pos) = from_luma(pix[0]);
                if uv_pos != last_uv_pos {
                    // Small optimization to avoid doing unnecessary calculations and writes
                    *output[1].get_unchecked_mut(uv_pos) = from_chroma(pix[1]);
                    *output[2].get_unchecked_mut(uv_pos) = from_chroma(pix[2]);
                    last_uv_pos = uv_pos;
                }
            }
        }
    }
    Ok(Yuv {
        data: output,
        width: width as u32,
        height: height as u32,
        config,
    })
}

#[inline(always)]
fn to_f32_luma<T: YuvPixel, const BD: u8, const FULL_RANGE: bool>(val: T) -> f32 {
    // Converts to a float value in the range 0.0..=1.0
    let val: f32 = val.as_();
    let max_val = f32::from(u16::MAX >> (16 - BD));
    clamp(
        (if FULL_RANGE {
            val
        } else {
            (match BD {
                8 => max_val / 219.0,
                10 => max_val / (940.0 - 64.0),
                12 => max_val / (3760.0 - 256.0),
                _ => unreachable!("Only bit depths 8, 10, and 12 are supported for limited range"),
            }) * (val - f32::from(16u16 << (BD - 8)))
        }) / max_val,
        0.0,
        1.0,
    )
}

#[inline(always)]
fn to_f32_chroma<T: YuvPixel, const BD: u8, const FULL_RANGE: bool>(val: T) -> f32 {
    // Converts to a float value in the range -0.5..=0.5
    let val: f32 = val.as_();
    let max_val = f32::from(u16::MAX >> (16 - BD));
    clamp(
        (if FULL_RANGE {
            val
        } else {
            (match BD {
                8 => max_val / 224.0,
                10 => max_val / (960.0 - 64.0),
                12 => max_val / (3840.0 - 256.0),
                _ => unreachable!("Only bit depths 8, 10, and 12 are supported for limited range"),
            }) * (val - f32::from(16u16 << (BD - 8)))
        }) / max_val
            - 0.5,
        -0.5,
        0.5,
    )
}

#[inline(always)]
fn from_f32_luma<T: YuvPixel, const BD: u8, const FULL_RANGE: bool>(val: f32) -> T {
    // Converts to a float value in the range 0.0..=1.0
    let max_val = f32::from(u16::MAX >> (16 - BD));
    let fval = clamp(
        if FULL_RANGE {
            val * max_val
        } else {
            ((match BD {
                8 => 219.0 / max_val,
                10 => (940.0 - 64.0) / max_val,
                12 => (3760.0 - 256.0) / max_val,
                _ => unreachable!("Only bit depths 8, 10, and 12 are supported for limited range"),
            }) * val)
                .mul_add(max_val, f32::from(16u16 << (BD - 8)))
        },
        0.0,
        max_val,
    );
    if size_of::<T>() == 1 {
        T::from_u8(fval.round() as u8).expect("This is a u8")
    } else {
        T::from_u16(fval.round() as u16).expect("This is a u16")
    }
}

#[inline(always)]
fn from_f32_chroma<T: YuvPixel, const BD: u8, const FULL_RANGE: bool>(val: f32) -> T {
    // Converts from a float value in the range -0.5..=0.5
    let val: f32 = val + 0.5;
    let max_val = f32::from(u16::MAX >> (16 - BD));
    let fval = clamp(
        if FULL_RANGE {
            val * max_val
        } else {
            ((match BD {
                8 => 224.0 / max_val,
                10 => (960.0 - 64.0) / max_val,
                12 => (3840.0 - 256.0) / max_val,
                _ => unreachable!("Only bit depths 8, 10, and 12 are supported for limited range"),
            }) * val)
                .mul_add(max_val, f32::from(16u16 << (BD - 8)))
        },
        0.0,
        max_val,
    );
    if size_of::<T>() == 1 {
        T::from_u8(fval.round() as u8).expect("This is a u8")
    } else {
        T::from_u16(fval.round() as u16).expect("This is a u16")
    }
}

#[cfg(test)]
mod tests {
    use num_traits::clamp;

    use super::{from_f32_chroma, from_f32_luma, to_f32_chroma, to_f32_luma};

    #[test]
    fn to_f32_luma_full() {
        let inputs: &[u8] = &[
            0, 12, 16, 25, 55, 120, 128, 140, 180, 215, 235, 240, 250, 255,
        ];
        let outputs: &[f32] = &[
            0.0,
            0.047_058_8,
            0.062_745_1,
            0.098_039_2,
            0.215_686,
            0.470_588,
            0.501_961,
            0.549_02,
            0.705_882,
            0.843_137,
            0.921_569,
            0.941_177,
            0.980_392,
            1.0,
        ];

        for (input, output) in inputs.iter().copied().zip(outputs.iter().copied()) {
            let result = to_f32_luma::<_, 8, true>(input);
            assert!(
                (output - result).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                result,
                output
            );
            let result: u8 = from_f32_luma::<_, 8, true>(result);
            assert!(
                input == result,
                "Result {} differed from expected {}",
                result,
                input
            );
        }
    }

    #[test]
    fn to_f32_luma_limited() {
        let inputs: &[u8] = &[
            0, 12, 16, 25, 55, 120, 128, 140, 180, 215, 235, 240, 250, 255,
        ];
        let outputs: &[f32] = &[
            0.0,
            0.0,
            0.0,
            0.041_095_9,
            0.178_082,
            0.474_886,
            0.511_415,
            0.566_21,
            0.748_858,
            0.908_676,
            1.0,
            1.0,
            1.0,
            1.0,
        ];

        for (input, output) in inputs.iter().copied().zip(outputs.iter().copied()) {
            let result = to_f32_luma::<_, 8, false>(input);
            assert!(
                (output - result).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                result,
                output
            );
            let result: u8 = from_f32_luma::<_, 8, false>(result);
            let expected = clamp(input, 16, 235);
            assert!(
                expected == result,
                "Result {} differed from expected {}",
                result,
                expected
            );
        }
    }

    #[test]
    fn to_f32_chroma_full() {
        let inputs: &[u8] = &[
            0, 12, 16, 25, 55, 120, 128, 140, 180, 215, 235, 240, 250, 255,
        ];
        let outputs: &[f32] = &[
            -0.5,
            -0.454_902,
            -0.439_216,
            -0.403_922,
            -0.286_275,
            -0.031_372_6,
            0.0,
            0.047_058_8,
            0.203_922,
            0.341_176,
            0.419_608,
            0.439_216,
            0.478_431,
            0.498_039,
        ];

        for (input, output) in inputs.iter().copied().zip(outputs.iter().copied()) {
            let result = to_f32_chroma::<_, 8, true>(input);
            assert!(
                (output - result).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                result,
                output
            );
            let result: u8 = from_f32_chroma::<_, 8, true>(result);
            assert!(
                input == result,
                "Result {} differed from expected {}",
                result,
                input
            );
        }
    }

    #[test]
    fn to_f32_chroma_limited() {
        let inputs: &[u8] = &[
            0, 12, 16, 25, 55, 120, 128, 140, 180, 215, 235, 240, 250, 255,
        ];
        let outputs: &[f32] = &[
            -0.5,
            -0.5,
            -0.5,
            -0.459_821,
            -0.325_893,
            -0.0357_143,
            0.0,
            0.053_571_4,
            0.232_143,
            0.388_393,
            0.477_679,
            0.5,
            0.5,
            0.5,
        ];

        for (input, output) in inputs.iter().copied().zip(outputs.iter().copied()) {
            let result = to_f32_chroma::<_, 8, false>(input);
            assert!(
                (output - result).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                result,
                output
            );
            let result: u8 = from_f32_chroma::<_, 8, false>(result);
            let expected = clamp(input, 16, 240);
            assert!(
                expected == result,
                "Result {} differed from expected {}",
                result,
                expected
            );
        }
    }

    #[test]
    fn to_f32_luma_full_10b() {
        let inputs: &[u16] = &[
            0 << 2,
            12 << 2,
            16 << 2,
            25 << 2,
            55 << 2,
            120 << 2,
            140 << 2,
            180 << 2,
            215 << 2,
            235 << 2,
            240 << 2,
            250 << 2,
            (256 << 2) - 1,
        ];
        let outputs: &[f32] = &[
            0.0,
            0.046_920_8,
            0.062_561_1,
            0.097_751_7,
            0.215_054,
            0.469_208,
            0.547_41,
            0.703_812,
            0.840_665,
            0.918_866,
            0.938_416,
            0.977_517,
            1.0,
        ];

        for (input, output) in inputs.iter().copied().zip(outputs.iter().copied()) {
            let result = to_f32_luma::<_, 10, true>(input);
            assert!(
                (output - result).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                result,
                output
            );
            let result: u16 = from_f32_luma::<_, 10, true>(result);
            assert!(
                input == result,
                "Result {} differed from expected {}",
                result,
                input
            );
        }
    }

    #[test]
    fn to_f32_luma_limited_10b() {
        let inputs: &[u16] = &[
            0 << 2,
            12 << 2,
            16 << 2,
            25 << 2,
            55 << 2,
            120 << 2,
            140 << 2,
            180 << 2,
            215 << 2,
            235 << 2,
            240 << 2,
            250 << 2,
            (256 << 2) - 1,
        ];
        let outputs: &[f32] = &[
            0.0,
            0.0,
            0.0,
            0.041_095_9,
            0.178_082,
            0.474_886,
            0.566_21,
            0.748_858,
            0.908_676,
            1.0,
            1.0,
            1.0,
            1.0,
        ];

        for (input, output) in inputs.iter().copied().zip(outputs.iter().copied()) {
            let result = to_f32_luma::<_, 10, false>(input);
            assert!(
                (output - result).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                result,
                output
            );
            let result: u16 = from_f32_luma::<_, 10, false>(result);
            let expected = clamp(input, 16 << 2, 235 << 2);
            assert!(
                expected == result,
                "Result {} differed from expected {}",
                result,
                expected
            );
        }
    }

    #[test]
    fn to_f32_chroma_full_10b() {
        let inputs: &[u16] = &[
            0 << 2,
            12 << 2,
            16 << 2,
            25 << 2,
            55 << 2,
            120 << 2,
            140 << 2,
            180 << 2,
            215 << 2,
            235 << 2,
            240 << 2,
            250 << 2,
            (256 << 2) - 1,
        ];
        let outputs: &[f32] = &[
            -0.5,
            -0.453_568,
            -0.437_928,
            -0.402_737,
            -0.285_435,
            -0.031_280_5,
            0.046_920_8,
            0.203_324,
            0.340_176,
            0.418_377,
            0.437_928,
            0.477_028,
            0.499_511,
        ];

        for (input, output) in inputs.iter().copied().zip(outputs.iter().copied()) {
            let result = to_f32_chroma::<_, 10, true>(input);
            assert!(
                (output - result).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                result,
                output
            );
            let result: u16 = from_f32_chroma::<_, 10, true>(result);
            assert!(
                input == result,
                "Result {} differed from expected {}",
                result,
                input
            );
        }
    }

    #[test]
    fn to_f32_chroma_limited_10b() {
        let inputs: &[u16] = &[
            0 << 2,
            12 << 2,
            16 << 2,
            25 << 2,
            55 << 2,
            120 << 2,
            140 << 2,
            180 << 2,
            215 << 2,
            235 << 2,
            240 << 2,
            250 << 2,
            (256 << 2) - 1,
        ];
        let outputs: &[f32] = &[
            -0.5,
            -0.5,
            -0.5,
            -0.459_821,
            -0.325_893,
            -0.035_714_3,
            0.053_571_4,
            0.232_143,
            0.388_393,
            0.477_679,
            0.5,
            0.5,
            0.5,
        ];

        for (input, output) in inputs.iter().copied().zip(outputs.iter().copied()) {
            let result = to_f32_chroma::<_, 10, false>(input);
            assert!(
                (output - result).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                result,
                output
            );
            let result: u16 = from_f32_chroma::<_, 10, false>(result);
            let expected = clamp(input, 16 << 2, 240 << 2);
            assert!(
                expected == result,
                "Result {} differed from expected {}",
                result,
                expected
            );
        }
    }
}
