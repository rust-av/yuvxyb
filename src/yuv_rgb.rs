//! Conversion from YUV to XYB requires a few steps. The most direct path is
//! YUV -> sRGB -> Linear RGB -> XYZ -> LMS -> XYB. We can fuse the matrices for
//! XYZ, LMS, and XYB to decrease the number of steps to just YUV -> sRGB ->
//! Linear RGB -> XYB.

#![allow(clippy::cast_sign_loss)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::doc_markdown)]

mod color;
mod transfer;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Yuv;
    use anyhow::Result;
    use av_data::pixel::{ColorPrimaries, MatrixCoefficients, TransferCharacteristic};
    use num_traits::clamp;
    use v_frame::{frame::Frame, plane::Plane};

    /// Converts 8..=16-bit YUV data to 32-bit floating point Linear RGB
    /// in a range of 0.0..=1.0;
    fn yuv_to_linear_rgb<T: Pixel>(input: &Yuv<T>) -> Result<Vec<[f32; 3]>> {
        let rgb = yuv_to_rgb(input)?;
        let config = input.config();
        let data = config.transfer_characteristics.to_linear(rgb)?;
        transform_primaries(data, config.color_primaries, ColorPrimaries::BT709)
    }

    /// Converts 32-bit floating point Linear RGB in a range of 0.0..=1.0
    /// to 8..=16-bit YUV.
    ///
    /// # Errors
    /// - If the `YuvConfig` would produce an invalid image
    fn linear_rgb_to_yuv<T: Pixel>(
        input: Vec<[f32; 3]>,
        width: usize,
        height: usize,
        config: YuvConfig,
    ) -> Result<Yuv<T>> {
        let data = transform_primaries(input, ColorPrimaries::BT709, config.color_primaries)?;
        let data = config.transfer_characteristics.to_gamma(data)?;
        rgb_to_yuv(&data, width, height, config)
    }

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
            let bd = 8;
            let full_range = true;
            let (scale, offset) = get_scale_offset::<true>(bd, full_range, false);
            let result = to_f32_luma(input, scale, offset);
            assert!(
                (output - result).abs() < 0.0005,
                "Result {result:.6} differed from expected {output:.6}"
            );
            let (scale, offset) = get_scale_offset::<false>(bd, full_range, false);
            let result: u8 = from_f32_luma(result, scale, offset, bd);
            assert!(
                input == result,
                "Result {result} differed from expected {input}"
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
            let bd = 8;
            let full_range = false;
            let (scale, offset) = get_scale_offset::<true>(bd, full_range, false);
            let result = to_f32_luma(input, scale, offset);
            assert!(
                (output - result).abs() < 0.0005,
                "Result {result:.6} differed from expected {output:.6}"
            );
            let (scale, offset) = get_scale_offset::<false>(bd, full_range, false);
            let result: u8 = from_f32_luma(result, scale, offset, bd);
            let expected = clamp(input, 16, 235);
            assert!(
                expected == result,
                "Result {result} differed from expected {input}"
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
            let bd = 8;
            let full_range = true;
            let (scale, offset) = get_scale_offset::<true>(bd, full_range, true);
            let result = to_f32_chroma(input, scale, offset);
            assert!(
                (output - result).abs() < 0.0005,
                "Result {result:.6} differed from expected {output:.6}"
            );
            let (scale, offset) = get_scale_offset::<false>(bd, full_range, true);
            let result: u8 = from_f32_chroma(result, scale, offset, bd, full_range);
            assert!(
                input == result,
                "Result {result} differed from expected {input}"
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
            let bd = 8;
            let full_range = false;
            let (scale, offset) = get_scale_offset::<true>(bd, full_range, true);
            let result = to_f32_chroma(input, scale, offset);
            assert!(
                (output - result).abs() < 0.0005,
                "Result {result:.6} differed from expected {output:.6}"
            );
            let (scale, offset) = get_scale_offset::<false>(bd, full_range, true);
            let result: u8 = from_f32_chroma(result, scale, offset, bd, full_range);
            let expected = clamp(input, 16, 240);
            assert!(
                expected == result,
                "Result {result} differed from expected {input}"
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
            let bd = 10;
            let full_range = true;
            let (scale, offset) = get_scale_offset::<true>(bd, full_range, false);
            let result = to_f32_luma(input, scale, offset);
            assert!(
                (output - result).abs() < 0.0005,
                "Result {result:.6} differed from expected {output:.6}"
            );
            let (scale, offset) = get_scale_offset::<false>(bd, full_range, false);
            let result: u16 = from_f32_luma(result, scale, offset, bd);
            assert!(
                input == result,
                "Result {result} differed from expected {input}"
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
            let bd = 10;
            let full_range = false;
            let (scale, offset) = get_scale_offset::<true>(bd, full_range, false);
            let result = to_f32_luma(input, scale, offset);
            assert!(
                (output - result).abs() < 0.0005,
                "Result {result:.6} differed from expected {output:.6}"
            );
            let (scale, offset) = get_scale_offset::<false>(bd, full_range, false);
            let result: u16 = from_f32_luma(result, scale, offset, bd);
            let expected = clamp(input, 16 << 2u8, 235 << 2u8);
            assert!(
                expected == result,
                "Result {result} differed from expected {input}"
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
            let bd = 10;
            let full_range = true;
            let (scale, offset) = get_scale_offset::<true>(bd, full_range, true);
            let result = to_f32_chroma(input, scale, offset);
            assert!(
                (output - result).abs() < 0.0005,
                "Result {result:.6} differed from expected {output:.6}"
            );
            let (scale, offset) = get_scale_offset::<false>(bd, full_range, true);
            let result: u16 = from_f32_chroma(result, scale, offset, bd, full_range);
            assert!(
                input == result,
                "Result {result} differed from expected {input}"
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
            let bd = 10;
            let full_range = false;
            let (scale, offset) = get_scale_offset::<true>(bd, full_range, true);
            let result = to_f32_chroma(input, scale, offset);
            assert!(
                (output - result).abs() < 0.0005,
                "Result {result:.6} differed from expected {output:.6}"
            );
            let (scale, offset) = get_scale_offset::<false>(bd, full_range, true);
            let result: u16 = from_f32_chroma(result, scale, offset, bd, full_range);
            let expected = clamp(input, 16 << 2u8, 240 << 2u8);
            assert!(
                expected == result,
                "Result {result} differed from expected {input}"
            );
        }
    }

    #[test]
    #[ignore = "8b tests are failing for unknown reason, need to be fixed, but can't find cause. 10b tests all pass"]
    fn bt601_full_to_rgb_8b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u8, u8, u8)> =
            vec![(168, 152, 92), (71, 57, 230), (122, 122, 79), (133, 96, 39)];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (0.176_271, 0.462_868, 0.632_797),
            (0.617_078, 0.014_544_7, -0.001_077_79),
            (0.039_517_2, 0.313_771, 0.136_314),
            (0.031_437_4, 0.590_267, 0.052_908_4),
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
            Frame {
                planes: [
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.0).collect::<Vec<_>>(), 2),
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.1).collect::<Vec<_>>(), 2),
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.2).collect::<Vec<_>>(), 2),
                ],
            },
            config,
        )
        .unwrap();
        let rgb = yuv_to_linear_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u8> = linear_rgb_to_yuv(rgb, 2, 2, config).unwrap();
        for y in 0..2 {
            for x in 0..2 {
                let expected = yuv_pixels[y * 2 + x];
                let dy = yuv.data()[0].p(x, y);
                let du = yuv.data()[1].p(x, y);
                let dv = yuv.data()[2].p(x, y);
                assert_eq!(dy, expected.0);
                assert_eq!(du, expected.1);
                assert_eq!(dv, expected.2);
            }
        }
    }

    #[test]
    #[ignore = "8b tests are failing for unknown reason, need to be fixed, but can't find cause. 10b tests all pass"]
    fn bt601_limited_to_rgb_8b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u8, u8, u8)> =
            vec![(168, 152, 92), (71, 57, 230), (122, 122, 79), (133, 96, 39)];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (0.187_073, 0.534_06, 0.745_515),
            (0.709_485, 0.013_730_4, -0.001_225_91),
            (0.034_007_4, 0.345_284, 0.136_045),
            (0.036_132, 0.686_773, 0.044_742_8),
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
            Frame {
                planes: [
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.0).collect::<Vec<_>>(), 2),
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.1).collect::<Vec<_>>(), 2),
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.2).collect::<Vec<_>>(), 2),
                ],
            },
            config,
        )
        .unwrap();
        let rgb = yuv_to_linear_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u8> = linear_rgb_to_yuv(rgb, 2, 2, config).unwrap();
        for y in 0..2 {
            for x in 0..2 {
                let expected = yuv_pixels[y * 2 + x];
                let dy = yuv.data()[0].p(x, y);
                let du = yuv.data()[1].p(x, y);
                let dv = yuv.data()[2].p(x, y);
                assert_eq!(dy, expected.0);
                assert_eq!(du, expected.1);
                assert_eq!(dv, expected.2);
            }
        }
    }

    #[test]
    fn bt601_full_to_rgb_10b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u16, u16, u16)> = vec![
            (649, 530, 380),
            (352, 428, 879),
            (449, 635, 445),
            (437, 784, 449),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (0.167_542, 0.448_585, 0.376_461),
            (0.631_3, 0.017_788, 0.019_665_5),
            (0.084_962_2, 0.145_111, 0.359_581),
            (0.083_808_2, 0.108_535, 0.777_17),
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
            Frame {
                planes: [
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.0).collect::<Vec<_>>(), 2),
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.1).collect::<Vec<_>>(), 2),
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.2).collect::<Vec<_>>(), 2),
                ],
            },
            config,
        )
        .unwrap();
        let rgb = yuv_to_linear_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u16> = linear_rgb_to_yuv(rgb, 2, 2, config).unwrap();
        for y in 0..2 {
            for x in 0..2 {
                let expected = yuv_pixels[y * 2 + x];
                let dy = yuv.data()[0].p(x, y);
                let du = yuv.data()[1].p(x, y);
                let dv = yuv.data()[2].p(x, y);
                assert_eq!(dy, expected.0);
                assert_eq!(du, expected.1);
                assert_eq!(dv, expected.2);
            }
        }
    }

    #[test]
    fn bt601_limited_to_rgb_10b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u16, u16, u16)> = vec![
            (649, 530, 380),
            (352, 428, 879),
            (449, 635, 445),
            (437, 784, 449),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (0.177_588, 0.519_389, 0.429_835),
            (0.735_715, 0.015_674_2, 0.011_590_1),
            (0.079_238_1, 0.146_685, 0.401_792),
            (0.078_409, 0.105_985, 0.920_082),
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
            Frame {
                planes: [
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.0).collect::<Vec<_>>(), 2),
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.1).collect::<Vec<_>>(), 2),
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.2).collect::<Vec<_>>(), 2),
                ],
            },
            config,
        )
        .unwrap();
        let rgb = yuv_to_linear_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u16> = linear_rgb_to_yuv(rgb, 2, 2, config).unwrap();
        for y in 0..2 {
            for x in 0..2 {
                let expected = yuv_pixels[y * 2 + x];
                let dy = yuv.data()[0].p(x, y);
                let du = yuv.data()[1].p(x, y);
                let dv = yuv.data()[2].p(x, y);
                assert_eq!(dy, expected.0);
                assert_eq!(du, expected.1);
                assert_eq!(dv, expected.2);
            }
        }
    }

    #[test]
    #[ignore = "8b tests are failing for unknown reason, need to be fixed, but can't find cause. 10b tests all pass"]
    fn bt709_full_to_rgb_8b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u8, u8, u8)> =
            vec![(168, 152, 92), (71, 57, 230), (122, 122, 79), (133, 96, 39)];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (0.136_756, 0.435_524, 0.645_848),
            (0.793_989, 0.009_446_9, 0.0),
            (0.015_421_1, 0.262_53, 0.135_474),
            (0.0, 0.437_285, 0.050_705_9),
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
            Frame {
                planes: [
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.0).collect::<Vec<_>>(), 2),
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.1).collect::<Vec<_>>(), 2),
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.2).collect::<Vec<_>>(), 2),
                ],
            },
            config,
        )
        .unwrap();
        let rgb = yuv_to_linear_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u8> = linear_rgb_to_yuv(rgb, 2, 2, config).unwrap();
        for y in 0..2 {
            for x in 0..2 {
                let expected = yuv_pixels[y * 2 + x];
                let dy = yuv.data()[0].p(x, y);
                let du = yuv.data()[1].p(x, y);
                let dv = yuv.data()[2].p(x, y);
                assert_eq!(dy, expected.0);
                assert_eq!(du, expected.1);
                assert_eq!(dv, expected.2);
            }
        }
    }

    #[test]
    #[ignore = "8b tests are failing for unknown reason, need to be fixed, but can't find cause. 10b tests all pass"]
    fn bt709_limited_to_rgb_8b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u8, u8, u8)> =
            vec![(168, 152, 92), (71, 57, 230), (122, 122, 79), (133, 96, 39)];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (0.140_153, 0.500_134, 0.761_916),
            (0.925_443, 0.003_726_5, 0.0),
            (0.008_854_72, 0.283_511, 0.135_132),
            (0.0, 0.496_572, 0.042_859_3),
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
            Frame {
                planes: [
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.0).collect::<Vec<_>>(), 2),
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.1).collect::<Vec<_>>(), 2),
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.2).collect::<Vec<_>>(), 2),
                ],
            },
            config,
        )
        .unwrap();
        let rgb = yuv_to_linear_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u8> = linear_rgb_to_yuv(rgb, 2, 2, config).unwrap();
        for y in 0..2 {
            for x in 0..2 {
                let expected = yuv_pixels[y * 2 + x];
                let dy = yuv.data()[0].p(x, y);
                let du = yuv.data()[1].p(x, y);
                let dv = yuv.data()[2].p(x, y);
                assert_eq!(dy, expected.0);
                assert_eq!(du, expected.1);
                assert_eq!(dv, expected.2);
            }
        }
    }

    #[test]
    fn bt709_full_to_rgb_10b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u16, u16, u16)> = vec![
            (649, 530, 380),
            (352, 428, 879),
            (449, 635, 445),
            (437, 784, 449),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (0.132_819, 0.412_614, 0.378_43),
            (0.795_462, 0.018_939_1, 0.018_982_6),
            (0.072_853_8, 0.144_83, 0.371_61),
            (0.069_994_8, 0.115_066, 0.819_838),
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
            Frame {
                planes: [
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.0).collect::<Vec<_>>(), 2),
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.1).collect::<Vec<_>>(), 2),
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.2).collect::<Vec<_>>(), 2),
                ],
            },
            config,
        )
        .unwrap();
        let rgb = yuv_to_linear_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u16> = linear_rgb_to_yuv(rgb, 2, 2, config).unwrap();
        for y in 0..2 {
            for x in 0..2 {
                let expected = yuv_pixels[y * 2 + x];
                let dy = yuv.data()[0].p(x, y);
                let du = yuv.data()[1].p(x, y);
                let dv = yuv.data()[2].p(x, y);
                assert_eq!(dy, expected.0);
                assert_eq!(du, expected.1);
                assert_eq!(dv, expected.2);
            }
        }
    }

    #[test]
    fn bt709_limited_to_rgb_10b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u16, u16, u16)> = vec![
            (649, 530, 380),
            (352, 428, 879),
            (449, 635, 445),
            (437, 784, 449),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (0.136_251, 0.474_517, 0.432_284),
            (0.938_306, 0.011_321_7, 0.011_364_6),
            (0.065_771, 0.146_186, 0.416_47),
            (0.062_539_6, 0.112_142, 0.974_052),
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
            Frame {
                planes: [
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.0).collect::<Vec<_>>(), 2),
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.1).collect::<Vec<_>>(), 2),
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.2).collect::<Vec<_>>(), 2),
                ],
            },
            config,
        )
        .unwrap();
        let rgb = yuv_to_linear_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u16> = linear_rgb_to_yuv(rgb, 2, 2, config).unwrap();
        for y in 0..2 {
            for x in 0..2 {
                let expected = yuv_pixels[y * 2 + x];
                let dy = yuv.data()[0].p(x, y);
                let du = yuv.data()[1].p(x, y);
                let dv = yuv.data()[2].p(x, y);
                assert_eq!(dy, expected.0);
                assert_eq!(du, expected.1);
                assert_eq!(dv, expected.2);
            }
        }
    }

    #[test]
    fn bt2020_full_to_rgb_10b() {
        // These values were manually chosen semi-randomly
        let yuv_pixels: Vec<(u16, u16, u16)> = vec![
            (649, 530, 380),
            (352, 428, 879),
            (449, 635, 445),
            (437, 784, 449),
        ];
        let rgb_pixels: Vec<(f32, f32, f32)> = vec![
            (-0.045_010_6, 0.469_065, 0.377_969),
            (1.191_15, -0.077_644_3, 0.006_470_73),
            (0.009_840_05, 0.159_92, 0.403_686),
            (-0.011_980_1, 0.124_049, 0.919_839),
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
            Frame {
                planes: [
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.0).collect::<Vec<_>>(), 2),
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.1).collect::<Vec<_>>(), 2),
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.2).collect::<Vec<_>>(), 2),
                ],
            },
            config,
        )
        .unwrap();
        let rgb = yuv_to_linear_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u16> = linear_rgb_to_yuv(rgb, 2, 2, config).unwrap();
        for y in 0..2 {
            for x in 0..2 {
                let expected = yuv_pixels[y * 2 + x];
                let dy = yuv.data()[0].p(x, y);
                let du = yuv.data()[1].p(x, y);
                let dv = yuv.data()[2].p(x, y);
                assert_eq!(dy, expected.0);
                assert_eq!(du, expected.1);
                assert_eq!(dv, expected.2);
            }
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
            (-0.079_884_5, 0.543_609, 0.431_571),
            (1.401_31, -0.099_792_3, -0.003_625_77),
            (-0.006_279_73, 0.163_196, 0.454_858),
            (-0.034_417, 0.121_553, 1.097_1),
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
            Frame {
                planes: [
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.0).collect::<Vec<_>>(), 2),
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.1).collect::<Vec<_>>(), 2),
                    Plane::from_slice(&yuv_pixels.iter().map(|pix| pix.2).collect::<Vec<_>>(), 2),
                ],
            },
            config,
        )
        .unwrap();
        let rgb = yuv_to_linear_rgb(&input).unwrap();
        for (output, expected) in rgb.iter().zip(rgb_pixels.iter()) {
            assert!(
                (output[0] - expected.0).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[0],
                expected.0
            );
            assert!(
                (output[1] - expected.1).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[1],
                expected.1
            );
            assert!(
                (output[2] - expected.2).abs() < 0.0005,
                "Result {:.6} differed from expected {:.6}",
                output[2],
                expected.2
            );
        }
        let yuv: Yuv<u16> = linear_rgb_to_yuv(rgb, 2, 2, config).unwrap();
        for y in 0..2 {
            for x in 0..2 {
                let expected = yuv_pixels[y * 2 + x];
                let dy = yuv.data()[0].p(x, y);
                let du = yuv.data()[1].p(x, y);
                let dv = yuv.data()[2].p(x, y);
                assert_eq!(dy, expected.0);
                assert_eq!(du, expected.1);
                assert_eq!(dv, expected.2);
            }
        }
    }
}
