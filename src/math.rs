//! Conversion from YUV to XYB requires a few steps. The most direct path is
//! YUV -> sRGB -> Linear RGB -> XYZ -> LMS -> XYB. We can fuse the matrices for
//! XYZ, LMS, and XYB to decrease the number of steps to just YUV -> sRGB ->
//! Linear RGB -> XYB.
//!
//! Conversion from YUV to sRGB is done using the matrix coefficients associated
//! with the YUV data. Conversion from sRGB to Linear RGB is done using the
//! gamma transfer function associated with the YUV data. Linear RGB to XYZ and
//! LMS to XYB are done using standard matrices.
//!
//! There are multiple models for XYZ to LMS, given that the purpose of LMS is
//! to attempt to approximate the human vision system. The XYZ to LMS matrix
//! chosen is from the Stockman & Sharpe model, which is the most recent model
//! based on physiological functionality of the human eye.
//! <https://en.wikipedia.org/wiki/LMS_color_space#Stockman_&_Sharpe_(2000)>

#![allow(clippy::many_single_char_names)]

use std::mem::size_of;

use anyhow::Result;
use av_data::pixel::{MatrixCoefficients, TransferCharacteristic};
use ndarray::{ArrayView1, ArrayView2};
use num_traits::clamp;

use crate::{Yuv, YuvConfig, YuvPixel};

// The direct Linear sRGB to XYB matrices were computed using the following:
// const LINEAR_SRGB_TO_XYZ_MATRIX: [[f32; 3]; 3] =
//     [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [
//         0.0193, 0.1192, 0.9505,
//     ]];
// const XYZ_TO_LMS_MATRIX: [[f32; 3]; 3] = [
//     [0.240_576, 0.855_098, -0.039_698_3],
//     [-0.417_076, 1.177_26, 0.0786_283],
//     [0.0, 0.0, 0.516_835],
// ];
// const LMS_TO_XYB_MATRIX: [[f32; 3]; 3] = [[1.0, -1.0, 0.0], [1.0, 1.0, 0.0],
// [0.0, 0.0, 1.0]];
//
// const XYB_TO_LMS_MATRIX: [[f32; 3]; 3] = [[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0],
// [0.0, 0.0, 1.0]];
// const LMS_TO_XYZ_MATRIX: [[f32; 3]; 3] = [
//     [1.947_354_7, -1.414_451_2, 0.364_763_27],
//     [0.689_902_7, 0.348_321_89, 0.0],
//     [0.0, 0.0, 1.934_853_4],
// ];
// const XYZ_TO_LINEAR_SRGB_MATRIX: [[f32; 3]; 3] =
//     [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [
//         0.0557, -0.2040, 1.057,
//     ]];

const LINEAR_SRGB_TO_XYB_MATRIX: [f32; 9] = [
    0.723_698,
    0.823_563,
    0.105_035,
    0.776_624,
    1.27092,
    0.085_110_6,
    0.11176,
    0.201_905,
    0.504_509,
];
const XYB_TO_LINEAR_SRGB_MATRIX: [f32; 9] = [
    4.51223, -2.85113, -0.458_423, -2.76445, 2.55527, 0.144_463, 0.106_775, -0.391_031, 2.02586,
];

/// Converts 32-bit floating point linear RGB to XYB. This function does assume
/// that the input is Linear RGB. If you pass it gamma-encoded RGB, the results
/// will be incorrect.
#[must_use]
pub fn linear_rgb_to_xyb(input: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let transform =
        ArrayView2::from_shape((3, 3), &LINEAR_SRGB_TO_XYB_MATRIX).expect("Matrix is valid");
    input
        .iter()
        .map(|pix| {
            let pix = ArrayView1::from_shape((3,), pix).expect("Matrix is valid");
            let res = transform.dot(&pix);
            [res[0], res[1], res[2]]
        })
        .collect()
}

/// Converts 32-bit floating point XYB to Lienar RGB. This does not perform
/// gamma encoding on the resulting RGB.
#[must_use]
pub fn xyb_to_linear_rgb(input: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let transform =
        ArrayView2::from_shape((3, 3), &XYB_TO_LINEAR_SRGB_MATRIX).expect("Matrix is valid");
    input
        .iter()
        .map(|pix| {
            let pix = ArrayView1::from_shape((3,), pix).expect("Matrix is valid");
            let res = transform.dot(&pix);
            [res[0], res[1], res[2]]
        })
        .collect()
}

/// Converts 8..=16-bit YUV data to 32-bit floating point Linear RGB
/// in a range of 0.0..=1.0;
#[must_use]
pub fn yuv_to_linear_rgb<T: YuvPixel>(input: &Yuv<T>) -> Vec<[f32; 3]> {
    let rgb = match input.config().matrix_coefficients {
        MatrixCoefficients::BT470BG | MatrixCoefficients::ST170M => {
            if input.config().full_range {
                bt601_full_to_rgb(input)
            } else {
                bt601_limited_to_rgb(input)
            }
        }
        MatrixCoefficients::BT2020ConstantLuminance
        | MatrixCoefficients::BT2020NonConstantLuminance => {
            if input.config().full_range {
                bt2020_full_to_rgb(input)
            } else {
                bt2020_limited_to_rgb(input)
            }
        }
        // If the matrix is unknown, unimplemented, or unspecified, fallback to BT.709, as that's
        // the safest guess.
        _ => {
            if input.config().full_range {
                bt709_full_to_rgb(input)
            } else {
                bt709_limited_to_rgb(input)
            }
        }
    };
    rgb_gamma_decode(&rgb, input.config().transfer_characteristics)
}

// TODO: Use const generics once float generic params are stable.
macro_rules! yuv_to_rgb {
    (
        $name:expr,
        $full_range:expr,
        $rv_coeff:expr,
        $gu_coeff:expr,
        $gv_coeff:expr,
        $bu_coeff:expr
    ) => {
        paste::item! {
            #[must_use]
            fn [<$name _to_rgb>]<T: YuvPixel>(input: &Yuv<T>) -> Vec<[f32; 3]> {
                let mut output = vec![[0.0, 0.0, 0.0]; input.data().len()];
                let height = input.height() as usize;
                let width = input.width() as usize;
                let ss_y = input.config().subsampling_y;
                let ss_x = input.config().subsampling_x;
                let data = input.data();
                let max_val = f32::from(u16::MAX >> (16 - input.config().bit_depth));
                let limited_shift = f32::from(16u16 << (input.config().bit_depth - 8));
                for y in 0..height {
                    for x in 0..width {
                        let y_pos = y * width + x;
                        let c_pos = (y >> ss_y) * (width >> ss_x) + (x >> ss_x);
                        // SAFETY: The YUV struct has its bounds validated when it is constructed, and
                        // it is immutable, so we know the length of the data array matches the
                        // specified bounds.
                        unsafe {
                            let y = if $full_range {
                                data[0].get_unchecked(y_pos).as_()
                            } else {
                                // This can be left as 255/219 because the resulting value
                                // will be the same for all bit depths
                                (255.0 / 219.0) * (data[0].get_unchecked(y_pos).as_() - limited_shift)
                            };
                            let u = data[1].get_unchecked(c_pos).as_() - 128.0;
                            let v = data[2].get_unchecked(c_pos).as_() - 128.0;

                            let r = clamp(v.mul_add($rv_coeff, y), 0.0, max_val) / max_val;
                            let g = clamp(v.mul_add($gv_coeff, u.mul_add($gu_coeff, y)), 0.0, max_val) / max_val;
                            let b = clamp(u.mul_add($bu_coeff, y), 0.0, max_val) / max_val;
                            *output.get_unchecked_mut(y_pos) = [r, g, b];
                        }
                    }
                }
                output
            }
        }
    };
}

yuv_to_rgb!(bt601_limited, false, 1.596, -0.391, -0.813, 2.018);
yuv_to_rgb!(bt601_full, true, 1.402, -0.34414, -0.71414, 1.772);
yuv_to_rgb!(bt709_limited, false, 1.793, -0.213, -0.533, 2.112);
yuv_to_rgb!(bt709_full, true, 1.5748, -0.18732, -0.46812, 1.772);
yuv_to_rgb!(
    bt2020_limited,
    false,
    1.67867,
    -0.187_326,
    -0.65042,
    2.14177
);
yuv_to_rgb!(bt2020_full, true, 1.4746, -0.164_553, -0.571_353, 1.8814);

// TODO: Use const generics once float generic params are stable.
macro_rules! rgb_to_yuv {
    (
        $name:expr,
        $full_range:expr,
        $yr_coeff:expr,
        $yg_coeff:expr,
        $yb_coeff:expr,
        $ur_coeff:expr,
        $ug_coeff:expr,
        $ub_coeff:expr,
        $vr_coeff:expr,
        $vg_coeff:expr,
        $vb_coeff:expr
    ) => {
        paste::item! {
            fn [<rgb_to_ $name>]<T: YuvPixel>(input: &[[f32; 3]], width: u32, height: u32, config: YuvConfig) -> Result<Yuv<T>> {
                if size_of::<T>() == 1 {
                    assert!(config.bit_depth == 8);
                } else {
                    assert!(config.bit_depth > 8 && config.bit_depth <= 16);
                }

                let ss_y = config.subsampling_y;
                let ss_x = config.subsampling_x;
                let y_len = input.len();
                let c_len = (input.len() >> ss_y) >> ss_x;
                let mut output: [Vec<T>; 3] = [
                    vec![T::from_u16(0).expect("u16 is the largest YuvPixel type"); y_len],
                    vec![T::from_u16(0).expect("u16 is the largest YuvPixel type"); c_len],
                    vec![T::from_u16(0).expect("u16 is the largest YuvPixel type"); c_len]
                ];
                let height = height as usize;
                let width = width as usize;

                let max_luma_val = f32::from(if $full_range { u16::MAX >> (16 - config.bit_depth) } else { 235u16 << (config.bit_depth - 8) });
                let max_chroma_val = f32::from(if $full_range { u16::MAX >> (16 - config.bit_depth) } else { 240u16 << (config.bit_depth - 8) });
                let mult = f32::from(1u16 << (config.bit_depth - 8));
                for y in 0..height {
                    for x in 0..width {
                        let y_pos = y * width + x;
                        let c_pos = (y >> ss_y) * (width >> ss_x) + (x >> ss_x);
                        // SAFETY: We are constructing the data and know the bounds of both input
                        // and output, so we can avoid any out-of-bounds accesses.
                        unsafe {
                            let [r, g, b] = *input.get_unchecked(y_pos);

                            let y = r.mul_add($yr_coeff, g.mul_add($yg_coeff, b.mul_add($yb_coeff, if $full_range { 0.0 } else { 16.0 }))) * mult;
                            let u = r.mul_add($ur_coeff, g.mul_add($ug_coeff, b.mul_add($ub_coeff, 128.0))) * mult;
                            let v = r.mul_add($vr_coeff, g.mul_add($vg_coeff, b.mul_add($vb_coeff, 128.0))) * mult;

                            let y = T::from_f32(clamp(y.round(), if $full_range { 0.0 } else { 16.0 }, max_luma_val)).expect("rounded floats can convert to f32");
                            let u = T::from_f32(clamp(u.round(), if $full_range { 0.0 } else { 16.0 }, max_chroma_val)).expect("rounded floats can convert to f32");
                            let v = T::from_f32(clamp(v.round(), if $full_range { 0.0 } else { 16.0 }, max_chroma_val)).expect("rounded floats can convert to f32");
                            *output[0].get_unchecked_mut(y_pos) = y;
                            *output[1].get_unchecked_mut(c_pos) = u;
                            *output[2].get_unchecked_mut(c_pos) = v;
                        }
                    }
                }
                Yuv::new(output, width as u32, height as u32, config)
            }
        }
    };
}

rgb_to_yuv!(
    bt601_limited,
    false,
    0.256_951_15,
    0.504_420_76,
    0.097_734_64,
    -0.148_211_67,
    -0.290_954_26,
    0.439_165_95,
    0.439_165_95,
    -0.367_885_8,
    -0.071_280_15
);
rgb_to_yuv!(
    bt601_full,
    true,
    0.299_000_7,
    0.586_998_34,
    0.114_000_92,
    -0.168_736_3,
    -0.331_263_18,
    0.499_999_5,
    0.499_999_5,
    -0.418_686_42,
    -0.081_313_06
);
rgb_to_yuv!(
    bt709_limited,
    false,
    0.182_662_62,
    0.614_472_9,
    0.061_970_99,
    -0.100_672_01,
    -0.338_658_36,
    0.439_330_37,
    0.439_141_5,
    -0.398_910_46,
    -0.040_231_028
);
rgb_to_yuv!(
    bt709_full,
    true,
    0.212_598_82,
    0.715_202_57,
    0.072_198_612,
    -0.114_571_471,
    -0.385_429_26,
    0.500_000_8,
    0.500_000_8,
    -0.454_154_55,
    -0.045_846_21
);
rgb_to_yuv!(
    bt2020_limited,
    false,
    0.225_612_15,
    0.582_282_8,
    0.050_928_3,
    -0.122_655_178,
    -0.316_560_95,
    0.439_216_1,
    0.439_217_25,
    -0.403_891_6,
    -0.035_325_642
);
rgb_to_yuv!(
    bt2020_full,
    true,
    0.2627,
    0.678,
    0.0593,
    -0.13963,
    -0.36037,
    0.5,
    0.5,
    -0.459_786,
    -0.040_214_3
);

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
    let rgb = rgb_gamma_encode(input, config.transfer_characteristics);
    match config.matrix_coefficients {
        MatrixCoefficients::BT470BG | MatrixCoefficients::ST170M => {
            if config.full_range {
                rgb_to_bt601_full(&rgb, width, height, config)
            } else {
                rgb_to_bt601_limited(&rgb, width, height, config)
            }
        }
        MatrixCoefficients::BT2020ConstantLuminance
        | MatrixCoefficients::BT2020NonConstantLuminance => {
            if config.full_range {
                rgb_to_bt2020_full(&rgb, width, height, config)
            } else {
                rgb_to_bt2020_limited(&rgb, width, height, config)
            }
        }
        // If the matrix is unknown, unimplemented, or unspecified, fallback to BT.709, as that's
        // the safest guess.
        _ => {
            if config.full_range {
                rgb_to_bt709_full(&rgb, width, height, config)
            } else {
                rgb_to_bt709_limited(&rgb, width, height, config)
            }
        }
    }
}

/// Converts Linear RGB to gamma-encoded RGB using the transfer's OETF.
#[must_use]
fn rgb_gamma_encode(input: &[[f32; 3]], transfer: TransferCharacteristic) -> Vec<[f32; 3]> {
    #[inline(always)]
    fn to_srgb(c: f32) -> f32 {
        clamp(
            if c <= 0.003_130_8 {
                c * 12.92
            } else {
                (1.055 * c).powf(1.0 / 2.4) - 0.055
            },
            0.0,
            1.0,
        )
    }

    #[inline(always)]
    fn to_pow(c: f32, gamma: f32) -> f32 {
        clamp(c.powf(1.0 / gamma), 0.0, 1.0)
    }

    #[inline(always)]
    fn to_bt1886(c: f32) -> f32 {
        const GAMMA: f32 = 2.4;
        const INV_GAMMA: f32 = 1.0 / GAMMA;
        const WHITEPOINT: f32 = 203.0;
        const BLACKPOINT: f32 = 0.01;
        let wpdiff = WHITEPOINT.powf(INV_GAMMA) - BLACKPOINT.powf(INV_GAMMA);
        let a = wpdiff.powf(GAMMA);
        let b = BLACKPOINT.powf(INV_GAMMA) / wpdiff;
        let c = (c / a).powf(1.0 / GAMMA) - b;
        clamp(c, 0.0, 1.0)
    }

    #[inline(always)]
    fn to_pq(c: f32) -> f32 {
        const M1: f32 = 1305.0 / 8192.0;
        const M2: f32 = 2523.0 / 32.0;
        const C1: f32 = 107.0 / 128.0;
        const C2: f32 = 2413.0 / 128.0;
        const C3: f32 = 2392.0 / 128.0;

        let cm1 = c.powf(M1);
        clamp(
            (C2.mul_add(cm1, C1) / C3.mul_add(cm1, 1.0)).powf(M2),
            0.0,
            1.0,
        )
    }

    match transfer {
        TransferCharacteristic::Linear => input.to_vec(),
        TransferCharacteristic::SRGB => input
            .iter()
            .map(|pix| [to_srgb(pix[0]), to_srgb(pix[1]), to_srgb(pix[2])])
            .collect(),
        TransferCharacteristic::PerceptualQuantizer => input
            .iter()
            .map(|pix| [to_pq(pix[0]), to_pq(pix[1]), to_pq(pix[2])])
            .collect(),
        TransferCharacteristic::BT470M | TransferCharacteristic::BT470BG => {
            let gamma: f32 = match transfer {
                TransferCharacteristic::BT470M => 2.2,
                TransferCharacteristic::BT470BG => 2.8,
                _ => 2.4,
            };
            input
                .iter()
                .map(|pix| {
                    [
                        to_pow(pix[0], gamma),
                        to_pow(pix[1], gamma),
                        to_pow(pix[2], gamma),
                    ]
                })
                .collect()
        }
        // If the transfer function is unknown, unimplemented, or unspecified, fallback to BT.1886,
        // as that's the safest guess.
        // This also covers BT.709 and BT.2020 intentionally.
        _ => input
            .iter()
            .map(|pix| [to_bt1886(pix[0]), to_bt1886(pix[1]), to_bt1886(pix[2])])
            .collect(),
    }
}

/// Converts gamma-encoded RGB to Linear RGB using the transfer's EOTF.
#[must_use]
fn rgb_gamma_decode(input: &[[f32; 3]], transfer: TransferCharacteristic) -> Vec<[f32; 3]> {
    #[inline(always)]
    fn from_srgb(c: f32) -> f32 {
        clamp(
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            },
            0.0,
            1.0,
        )
    }

    #[inline(always)]
    fn from_pow(c: f32, gamma: f32) -> f32 {
        clamp(c.powf(gamma), 0.0, 1.0)
    }

    #[inline(always)]
    fn from_bt1886(c: f32) -> f32 {
        const GAMMA: f32 = 2.4;
        const INV_GAMMA: f32 = 1.0 / GAMMA;
        const WHITEPOINT: f32 = 203.0;
        const BLACKPOINT: f32 = 0.01;
        let wpdiff = WHITEPOINT.powf(INV_GAMMA) - BLACKPOINT.powf(INV_GAMMA);
        let a = wpdiff.powf(GAMMA);
        let b = BLACKPOINT.powf(INV_GAMMA) / wpdiff;
        let c = a * (c + b).max(0.0).powf(GAMMA);
        clamp(c, 0.0, 1.0)
    }

    #[inline(always)]
    fn from_pq(c: f32) -> f32 {
        const M1: f32 = 1305.0 / 8192.0;
        const M2: f32 = 2523.0 / 32.0;
        const C1: f32 = 107.0 / 128.0;
        const C2: f32 = 2413.0 / 128.0;
        const C3: f32 = 2392.0 / 128.0;

        let cm2 = c.powf(1.0 / M2);
        clamp(
            10000.0 * ((cm2 - C1).max(0.0) / C3.mul_add(-cm2, C2)).powf(1.0 / M1),
            0.0,
            1.0,
        )
    }

    match transfer {
        TransferCharacteristic::Linear => input.to_vec(),
        TransferCharacteristic::SRGB => input
            .iter()
            .map(|pix| [from_srgb(pix[0]), from_srgb(pix[1]), from_srgb(pix[2])])
            .collect(),
        TransferCharacteristic::PerceptualQuantizer => input
            .iter()
            .map(|pix| [from_pq(pix[0]), from_pq(pix[1]), from_pq(pix[2])])
            .collect(),
        TransferCharacteristic::BT470M | TransferCharacteristic::BT470BG => {
            let gamma: f32 = match transfer {
                TransferCharacteristic::BT470M => 2.2,
                TransferCharacteristic::BT470BG => 2.8,
                _ => 2.4,
            };
            input
                .iter()
                .map(|pix| {
                    [
                        from_pow(pix[0], gamma),
                        from_pow(pix[1], gamma),
                        from_pow(pix[2], gamma),
                    ]
                })
                .collect()
        }
        // If the transfer function is unknown, unimplemented, or unspecified, fallback to BT.1886,
        // as that's the safest guess.
        _ => input
            .iter()
            .map(|pix| {
                [
                    from_bt1886(pix[0]),
                    from_bt1886(pix[1]),
                    from_bt1886(pix[2]),
                ]
            })
            .collect(),
    }
}
