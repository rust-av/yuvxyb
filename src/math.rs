#![allow(clippy::many_single_char_names)]

use av_data::pixel::{MatrixCoefficients, TransferCharacteristic};
use num_traits::clamp;

use crate::{YuvPixel, YUV};

// const BT709_TO_RGB_MATRIX: [[f32; 3]; 3] = todo!();
// const BT2020_TO_RGB_MATRIX: [[f32; 3]; 3] = todo!();
// const LINEAR_SRGB_TO_XYZ_MATRIX: [[f32; 3]; 3] = [
//     [0.412_410_86, 0.357_584_57, 0.180_453_8],
//     [0.212_649_35, 0.715_169_13, 0.072_181_52],
//     [0.019_331_759, 0.119_194_86, 0.950_390_04],
// ];
// const XYZ_TO_LMS_MATRIX: [[f32; 3]; 3] = todo!();
// const LMS_TO_XYB_MATRIX: [[f32; 3]; 3] = [[1.0, -1.0, 0.0], [1.0, 1.0, 0.0],
// [0.0, 0.0, 1.0]];

// fn build_yuv_to_xyb_matrix() -> [[f32; 3]; 3] {
//     todo!()
// }

// const XYB_TO_LMS_MATRIX: [[f32; 3]; 3] = todo!();
// const LMS_TO_XYZ_MATRIX: [[f32; 3]; 3] = todo!();
// const XYZ_TO_LINEAR_SRGB_MATRIX: [[f32; 3]; 3] = [
//     [3.240_812_3, -1.537_308_5, -0.498_586_54],
//     [-0.969_243, 1.875_966_3, 0.041_555_032],
//     [0.055_638_4, -0.204_007_46, 1.057_129_6],
// ];
// const RGB_TO_BT709_MATRIX: [[f32; 3]; 3] = todo!();
// const RGB_TO_BT2020_MATRIX: [[f32; 3]; 3] = todo!();
// #[must_use]
// fn build_xyb_to_yuv_matrix() -> [[f32; 3]; 3] {
//     todo!()
// }

/// Converts 8- or 16-bit YUV data to 32-bit floating point Linear RGB
/// in a range of 0.0..=1.0;
#[must_use]
pub fn yuv_to_linear_rgb<T: YuvPixel>(input: &YUV<T>) -> Vec<[f32; 3]> {
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
            fn [<$name _to_rgb>]<T: YuvPixel>(input: &YUV<T>) -> Vec<[f32; 3]> {
                let mut output = vec![[0.0, 0.0, 0.0]; input.data().len()];
                let height = input.height() as usize;
                let width = input.width() as usize;
                let ss_y = input.config().subsampling_y;
                let ss_x = input.config().subsampling_x;
                let data = input.data();
                let max_val = f32::from(255u16 << (input.config().bit_depth - 8));
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
                            let u = data[1].get_unchecked(c_pos).as_();
                            let v = data[2].get_unchecked(c_pos).as_();

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

/// Converts Linear RGB to gamma-encoded RGB
#[must_use]
pub fn rgb_gamma_encode(input: &[[f32; 3]], transfer: TransferCharacteristic) -> Vec<[f32; 3]> {
    todo!()
}

/// Converts gamma-encoded RGB to Linear RGB
#[must_use]
pub fn rgb_gamma_decode(input: &[[f32; 3]], transfer: TransferCharacteristic) -> Vec<[f32; 3]> {
    #[inline(always)]
    fn to_linear_srgb(c: f32) -> f32 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    #[inline(always)]
    fn to_linear_pow(c: f32, gamma: f32) -> f32 {
        c.powf(gamma)
    }

    match transfer {
        TransferCharacteristic::Linear => input.to_vec(),
        TransferCharacteristic::SRGB => input
            .iter()
            .map(|pix| {
                [
                    to_linear_srgb(pix[0]),
                    to_linear_srgb(pix[1]),
                    to_linear_srgb(pix[2]),
                ]
            })
            .collect(),
        TransferCharacteristic::PerceptualQuantizer => todo!("Handling HDR is important"),
        TransferCharacteristic::HybridLogGamma => todo!("Handling HDR is important"),
        // If the transfer function is unknown, unimplemented, or unspecified, fallback to BT.1886,
        // as that's the safest guess.
        _ => {
            let gamma: f32 = match transfer {
                TransferCharacteristic::BT470M => 2.2,
                TransferCharacteristic::BT470BG => 2.8,
                _ => 2.4,
            };
            input
                .iter()
                .map(|pix| {
                    [
                        to_linear_pow(pix[0], gamma),
                        to_linear_pow(pix[1], gamma),
                        to_linear_pow(pix[2], gamma),
                    ]
                })
                .collect()
        }
    }
}
