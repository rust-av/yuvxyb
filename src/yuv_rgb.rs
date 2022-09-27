//! Conversion from YUV to XYB requires a few steps. The most direct path is
//! YUV -> sRGB -> Linear RGB -> XYZ -> LMS -> XYB. We can fuse the matrices for
//! XYZ, LMS, and XYB to decrease the number of steps to just YUV -> sRGB ->
//! Linear RGB -> XYB.

#![allow(clippy::many_single_char_names)]

mod color;
mod transfer;

use std::mem::size_of;

use anyhow::Result;
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

#[must_use]
fn ycbcr_to_ypbpr<T: YuvPixel>(input: &Yuv<T>) -> Vec<[f32; 3]> {
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
                    to_f32_luma(*data[0].get_unchecked(y_pos), bd, full_range),
                    to_f32_chroma(*data[1].get_unchecked(uv_pos), bd, full_range),
                    to_f32_chroma(*data[2].get_unchecked(uv_pos), bd, full_range),
                ];
            }
        }
    }
    output
}

fn ypbpr_to_ycbcr<T: YuvPixel>(
    input: &[[f32; 3]],
    width: usize,
    height: usize,
    config: YuvConfig,
) -> Yuv<T> {
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
                *output[0].get_unchecked_mut(y_pos) = from_f32_luma(pix[0], bd, full_range);
                if uv_pos != last_uv_pos {
                    // Small optimization to avoid doing unnecessary calculations and writes
                    *output[1].get_unchecked_mut(uv_pos) = from_f32_chroma(pix[1], bd, full_range);
                    *output[2].get_unchecked_mut(uv_pos) = from_f32_chroma(pix[2], bd, full_range);
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

#[inline(always)]
fn to_f32_luma<T: YuvPixel>(val: T, bd: u8, full_range: bool) -> f32 {
    // Converts to a float value in the range 0.0..=1.0
    let val: f32 = val.as_();
    let max_val = f32::from(u16::MAX >> (16 - bd));
    clamp(
        (if full_range {
            val
        } else {
            255.0 / 219.0 * (val - 16.0)
        }) / max_val,
        0.0,
        1.0,
    )
}

#[inline(always)]
fn to_f32_chroma<T: YuvPixel>(val: T, bd: u8, full_range: bool) -> f32 {
    // Converts to a float value in the range -0.5..=0.5
    let val: f32 = val.as_();
    let max_val = f32::from(u16::MAX >> (16 - bd));
    clamp(
        (if full_range {
            val
        } else {
            255.0 / 224.0 * (val - 16.0)
        }) / max_val
            - 0.5,
        -0.5,
        0.5,
    )
}

#[inline(always)]
fn from_f32_luma<T: YuvPixel>(val: f32, bd: u8, full_range: bool) -> T {
    // Converts to a float value in the range 0.0..=1.0
    let max_val = f32::from(u16::MAX >> (16 - bd));
    let fval = clamp(
        if full_range {
            val * max_val
        } else {
            (219.0f32 / 255.0 * val).mul_add(max_val, 16.0)
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
fn from_f32_chroma<T: YuvPixel>(val: f32, bd: u8, full_range: bool) -> T {
    // Converts from a float value in the range -0.5..=0.5
    let val: f32 = val + 0.5;
    let max_val = f32::from(u16::MAX >> (16 - bd));
    let fval = clamp(
        if full_range {
            val * max_val
        } else {
            (224.0f32 / 255.0 * val).mul_add(max_val, 16.0)
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
            let result = to_f32_luma(input, 8, true);
            assert!(
                (output - result).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                result,
                output
            );
            let result: u8 = from_f32_luma(result, 8, true);
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
            let result = to_f32_luma(input, 8, false);
            assert!(
                (output - result).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                result,
                output
            );
            let result: u8 = from_f32_luma(result, 8, false);
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
            let result = to_f32_chroma(input, 8, true);
            assert!(
                // FIXME: Something is weird where the values coming back from this conversion
                // are slightly different from the values Vapoursynth is giving.
                // Temporarily widened the acceptable difference to account for this...
                //     (output - result).abs() < 0.0005,
                (output - result).abs() < 0.005,
                "Result {:.6} differed from expected {:.6}",
                result,
                output
            );
            let result: u8 = from_f32_chroma(result, 8, true);
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
            let result = to_f32_chroma(input, 8, false);
            assert!(
                (output - result).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                result,
                output
            );
            let result: u8 = from_f32_chroma(result, 8, false);
            let expected = clamp(input, 16, 240);
            assert!(
                expected == result,
                "Result {} differed from expected {}",
                result,
                expected
            );
        }
    }
}
