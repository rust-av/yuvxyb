//! Conversion from YUV to XYB requires a few steps. The most direct path is
//! YUV -> sRGB -> Linear RGB -> XYZ -> LMS -> XYB. We can fuse the matrices for
//! XYZ, LMS, and XYB to decrease the number of steps to just YUV -> sRGB ->
//! Linear RGB -> XYB.

#![allow(clippy::cast_sign_loss)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::doc_markdown)]

mod color;
mod transfer;

#[cfg(test)]
mod tests;

use num_traits::clamp;
use v_frame::{frame::Frame, plane::Plane};

pub use self::color::{rgb_to_yuv, transform_primaries, yuv_to_rgb};
pub use self::transfer::TransferFunction;
use crate::{CastFromPrimitive, Pixel, Yuv, YuvConfig};

fn ycbcr_to_ypbpr<T: Pixel>(input: &Yuv<T>) -> Vec<[f32; 3]> {
    let w = input.width();
    let h = input.height();
    let ss_x = input.config().subsampling_x;
    let ss_y = input.config().subsampling_y;
    let bd = input.config().bit_depth;
    let full_range = input.config().full_range;

    let (luma_scale, luma_offset) = get_scale_offset::<true>(bd, full_range, false);
    let (chroma_scale, chroma_offset) = get_scale_offset::<true>(bd, full_range, true);

    let data = input.data();
    let y_stride = data[0].cfg.stride;
    let u_stride = data[1].cfg.stride;
    let v_stride = data[2].cfg.stride;
    let y_origin = data[0].data_origin();
    let u_origin = data[1].data_origin();
    let v_origin = data[2].data_origin();
    let mut output = vec![[0.0, 0.0, 0.0]; w * h];
    for y in 0..h {
        for x in 0..w {
            let output_pos = y * w + x;
            let y_pos = y * y_stride + x;
            let u_pos = (y >> ss_y) * u_stride + (x >> ss_x);
            let v_pos = (y >> ss_y) * v_stride + (x >> ss_x);
            // SAFETY: The bounds of the YUV data are validated when we construct it.
            unsafe {
                *output.get_unchecked_mut(output_pos) = [
                    to_f32_luma(*y_origin.get_unchecked(y_pos), luma_scale, luma_offset),
                    to_f32_chroma(*u_origin.get_unchecked(u_pos), chroma_scale, chroma_offset),
                    to_f32_chroma(*v_origin.get_unchecked(v_pos), chroma_scale, chroma_offset),
                ];
            }
        }
    }
    output
}

#[allow(clippy::too_many_lines)]
fn ypbpr_to_ycbcr<T: Pixel>(
    input: &[[f32; 3]],
    width: usize,
    height: usize,
    config: YuvConfig,
) -> Yuv<T> {
    let ss_x = config.subsampling_x;
    let ss_y = config.subsampling_y;
    let bd = config.bit_depth;
    let full_range = config.full_range;
    let chroma_width = width >> ss_x;
    let chroma_height = height >> ss_y;

    let (luma_scale, luma_offset) = get_scale_offset::<false>(bd, full_range, false);
    let (chroma_scale, chroma_offset) = get_scale_offset::<false>(bd, full_range, true);

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
            let input_pos = y * width + x;
            let y_pos = y * y_stride + x;
            let u_pos = (y >> ss_y) * u_stride + (x >> ss_x);
            let v_pos = (y >> ss_y) * v_stride + (x >> ss_x);
            // SAFETY: The bounds of the YUV data are validated when we construct it.
            unsafe {
                let pix = input.get_unchecked(input_pos);
                *y_origin.get_unchecked_mut(y_pos) =
                    from_f32_luma(pix[0], luma_scale, luma_offset, bd);
                if u_pos != last_uv_pos {
                    // Small optimization to avoid doing unnecessary calculations and writes
                    // We can track this from just `u_pos`. We have `v_pos` separate for indexing
                    // on the off chance that the two planes have different strides.
                    *u_origin.get_unchecked_mut(u_pos) =
                        from_f32_chroma(pix[1], chroma_scale, chroma_offset, bd, full_range);
                    *v_origin.get_unchecked_mut(v_pos) =
                        from_f32_chroma(pix[2], chroma_scale, chroma_offset, bd, full_range);
                    last_uv_pos = u_pos;
                }
            }
        }
    }

    Yuv::new(output, config).unwrap()
}

#[inline(always)]
fn to_f32_luma<T: Pixel>(val: T, scale: f32, offset: f32) -> f32 {
    // Converts to a float value in the range 0.0..=1.0
    let val = f32::from(u16::cast_from(val));
    clamp(val.mul_add(scale, offset), 0.0, 1.0)
}

#[inline(always)]
fn to_f32_chroma<T: Pixel>(val: T, scale: f32, offset: f32) -> f32 {
    // Converts to a float value in the range -0.5..=0.5
    let val = f32::from(u16::cast_from(val));
    clamp(val.mul_add(scale, offset), -0.5, 0.5)
}

#[inline(always)]
fn from_f32_luma<T: Pixel>(val: f32, scale: f32, offset: f32, bd: u8) -> T {
    // Converts to a float value in the range 0.0..=1.0
    T::cast_from(clamp(
        val.mul_add(scale, offset).round() as u16,
        0,
        ((1u32 << bd) - 1) as u16,
    ))
}

#[inline(always)]
fn from_f32_chroma<T: Pixel>(val: f32, scale: f32, offset: f32, bd: u8, full_range: bool) -> T {
    // Accounts for rounding issues
    if full_range && (val + 0.5).abs() < f32::EPSILON {
        return T::cast_from(0u16);
    }

    // Converts from a float value in the range -0.5..=0.5
    T::cast_from(clamp(
        val.mul_add(scale, offset).round() as u16,
        0,
        ((1u32 << bd) - 1) as u16,
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
