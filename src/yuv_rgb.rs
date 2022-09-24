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
                ];
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
