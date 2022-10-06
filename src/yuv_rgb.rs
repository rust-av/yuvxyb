//! Conversion from YUV to XYB requires a few steps. The most direct path is
//! YUV -> sRGB -> Linear RGB -> XYZ -> LMS -> XYB. We can fuse the matrices for
//! XYZ, LMS, and XYB to decrease the number of steps to just YUV -> sRGB ->
//! Linear RGB -> XYB.

#![allow(clippy::many_single_char_names)]

mod color;
mod transfer;

use anyhow::{bail, Result};
use num_traits::clamp;
use v_frame::{frame::Frame, plane::Plane};

use self::{
    color::{rgb_to_yuv, yuv_to_rgb},
    transfer::TransferFunction,
};
use crate::{CastFromPrimitive, Pixel, Yuv, YuvConfig};

/// Converts 8..=16-bit YUV data to 32-bit floating point Linear RGB
/// in a range of 0.0..=1.0;
pub fn yuv_to_linear_rgb<T: Pixel>(input: &Yuv<T>) -> Result<Vec<[f32; 3]>> {
    let rgb = yuv_to_rgb(input)?;
    input.config().transfer_characteristics.to_linear(&rgb)
}

/// Converts 32-bit floating point Linear RGB in a range of 0.0..=1.0
/// to 8..=16-bit YUV.
///
/// # Errors
/// - If the `YuvConfig` would produce an invalid image
pub fn linear_rgb_to_yuv<T: Pixel>(
    input: &[[f32; 3]],
    width: usize,
    height: usize,
    config: YuvConfig,
) -> Result<Yuv<T>> {
    let rgb = config.transfer_characteristics.to_gamma(input)?;
    rgb_to_yuv(&rgb, width, height, config)
}

fn ycbcr_to_ypbpr<T: Pixel>(input: &Yuv<T>) -> Result<Vec<[f32; 3]>> {
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
    let y_origin = data[0].data_origin();
    let u_origin = data[1].data_origin();
    let v_origin = data[2].data_origin();
    let mut output = vec![[0.0, 0.0, 0.0]; w * h];
    for y in 0..h {
        for x in 0..w {
            let y_pos = y * w + x;
            let uv_pos = (y >> ss_y) * (w >> ss_x) + (x >> ss_x);
            // SAFETY: The bounds of the YUV data are validated when we construct it.
            unsafe {
                *output.get_unchecked_mut(y_pos) = [
                    to_luma(*y_origin.get_unchecked(y_pos)),
                    to_chroma(*u_origin.get_unchecked(uv_pos)),
                    to_chroma(*v_origin.get_unchecked(uv_pos)),
                ];
            }
        }
    }
    Ok(output)
}

fn ypbpr_to_ycbcr<T: Pixel>(
    input: &[[f32; 3]],
    width: usize,
    height: usize,
    config: YuvConfig,
) -> Result<Yuv<T>> {
    let ss_x = config.subsampling_x;
    let ss_y = config.subsampling_y;
    let bd = config.bit_depth;
    let full_range = config.full_range;
    let chroma_width = width >> ss_x;
    let chroma_height = height >> ss_y;

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

    let mut output: Frame<T> = Frame {
        planes: [
            Plane::new(width, height, 0, 0, 0, 0),
            Plane::new(
                chroma_width,
                chroma_height,
                usize::from(ss_x),
                usize::from(ss_y),
                0,
                0,
            ),
            Plane::new(
                chroma_width,
                chroma_height,
                usize::from(ss_x),
                usize::from(ss_y),
                0,
                0,
            ),
        ],
    };

    // We setup the plane origins as mutable slices outside the loop
    // because `data_origin_mut` is _not_ just a simple array index,
    // so it would optimize poorly if called during each loop iteration.
    let (y_plane, rest) = output.planes.split_first_mut().expect("has 3 planes");
    let (u_plane, rest) = rest.split_first_mut().expect("has 3 planes");
    let (v_plane, _) = rest.split_first_mut().expect("has 3 planes");
    let y_stride = y_plane.cfg.stride;
    let u_stride = u_plane.cfg.stride;
    let v_stride = v_plane.cfg.stride;
    let y_origin = y_plane.data_origin_mut();
    let u_origin = u_plane.data_origin_mut();
    let v_origin = v_plane.data_origin_mut();
    let mut last_uv_pos = usize::MAX;
    for y in 0..height {
        for x in 0..width {
            let y_pos = y * y_stride + x;
            let u_pos = (y >> ss_y) * u_stride + (x >> ss_x);
            let v_pos = (y >> ss_y) * v_stride + (x >> ss_x);
            // SAFETY: The bounds of the YUV data are validated when we construct it.
            unsafe {
                let pix = input.get_unchecked(y_pos);
                *y_origin.get_unchecked_mut(y_pos) = from_luma(pix[0]);
                if u_pos != last_uv_pos {
                    // Small optimization to avoid doing unnecessary calculations and writes
                    // We can track this from just `u_pos`. We have `v_pos` separate for indexing
                    // on the off chance that the two planes have different strides.
                    *u_origin.get_unchecked_mut(u_pos) = from_chroma(pix[1]);
                    *v_origin.get_unchecked_mut(v_pos) = from_chroma(pix[2]);
                    last_uv_pos = u_pos;
                }
            }
        }
    }
    Ok(Yuv {
        data: output,
        config,
    })
}

#[inline(always)]
fn to_f32_luma<T: Pixel, const BD: u8, const FULL_RANGE: bool>(val: T) -> f32 {
    // Converts to a float value in the range 0.0..=1.0
    let val = f32::from(u16::cast_from(val));
    let (scale, offset) = get_scale_offset::<true>(BD, FULL_RANGE, false);
    clamp(val.mul_add(scale, offset), 0.0, 1.0)
}

#[inline(always)]
fn to_f32_chroma<T: Pixel, const BD: u8, const FULL_RANGE: bool>(val: T) -> f32 {
    // Converts to a float value in the range -0.5..=0.5
    let val = f32::from(u16::cast_from(val));
    let (scale, offset) = get_scale_offset::<true>(BD, FULL_RANGE, true);
    clamp(val.mul_add(scale, offset), -0.5, 0.5)
}

#[inline(always)]
fn from_f32_luma<T: Pixel, const BD: u8, const FULL_RANGE: bool>(val: f32) -> T {
    // Converts to a float value in the range 0.0..=1.0
    let (scale, offset) = get_scale_offset::<false>(BD, FULL_RANGE, false);
    T::cast_from(clamp(
        val.mul_add(scale, offset).round() as u16,
        0,
        ((1u32 << BD) - 1) as u16,
    ))
}

#[inline(always)]
fn from_f32_chroma<T: Pixel, const BD: u8, const FULL_RANGE: bool>(val: f32) -> T {
    // Accounts for rounding issues
    if FULL_RANGE && (val + 0.5).abs() < f32::EPSILON {
        return T::cast_from(0u16);
    }

    // Converts from a float value in the range -0.5..=0.5
    let (scale, offset) = get_scale_offset::<false>(BD, FULL_RANGE, true);
    T::cast_from(clamp(
        val.mul_add(scale, offset).round() as u16,
        0,
        ((1u32 << BD) - 1) as u16,
    ))
}

#[allow(clippy::useless_let_if_seq)]
fn get_scale_offset<const TO_FLOAT: bool>(
    bit_depth: u8,
    full_range: bool,
    chroma: bool,
) -> (f32, f32) {
    let range_in;
    let offset_in;
    let range_out;
    let offset_out;
    if TO_FLOAT {
        range_in = pixel_range(bit_depth, full_range, chroma);
        offset_in = pixel_offset(bit_depth, full_range, chroma);
        range_out = pixel_range(32, true, false);
        offset_out = pixel_offset(32, true, false);
    } else {
        range_in = pixel_range(32, true, false);
        offset_in = pixel_offset(32, true, false);
        range_out = pixel_range(bit_depth, full_range, chroma);
        offset_out = pixel_offset(bit_depth, full_range, chroma);
    }

    let scale = (range_out / range_in) as f32;
    let offset = (-offset_in * range_out / range_in + offset_out) as f32;

    (scale, offset)
}

fn pixel_range(bit_depth: u8, full_range: bool, chroma: bool) -> f64 {
    if bit_depth == 32 {
        // floating point
        1.0f64
    } else if full_range {
        f64::from((1u32 << bit_depth) - 1)
    } else if chroma {
        f64::from(224u16 << (bit_depth - 8))
    } else {
        f64::from(219u16 << (bit_depth - 8))
    }
}

fn pixel_offset(bit_depth: u8, full_range: bool, chroma: bool) -> f64 {
    if bit_depth == 32 {
        // floating point
        0.0f64
    } else if chroma {
        f64::from(1u32 << (bit_depth - 1))
    } else if full_range {
        0.0f64
    } else {
        f64::from(16u16 << (bit_depth - 8))
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
            let expected = clamp(input, 16 << 2u8, 235 << 2u8);
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
            let expected = clamp(input, 16 << 2u8, 240 << 2u8);
            assert!(
                expected == result,
                "Result {} differed from expected {}",
                result,
                expected
            );
        }
    }
}
