use anyhow::{bail, Result};
use av_data::pixel::TransferCharacteristic;
use debug_unreachable::debug_unreachable;
use num_traits::clamp;

use crate::YCbCr;

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

fn build_xyb_to_yuv_matrix() -> [[f32; 3]; 3] {
    todo!()
}

/// The BT.601 formula only works for 8-bit data. It's kind of garbage, given
/// that it was made in 1982 and designed for TVs from 1982. Those are the
/// limitations we hold to for this conversion. If you want higher precision
/// conversion, make sure your data is in a more recent color matrix such as
/// BT.709.
///
/// The returned RGB data is gamma encoded.
#[must_use]
pub fn bt601_limited_to_rgb(input: &YCbCr<u8>) -> Vec<[u8; 3]> {
    let mut output = vec![[0, 0, 0]; input.data().len()];
    let height = input.height() as usize;
    let width = input.width() as usize;
    let ss_y = input.config().subsampling_y;
    let ss_x = input.config().subsampling_x;
    let data = input.data();
    for y in 0..height {
        for x in 0..width {
            let y_pos = y * width + x;
            let c_pos = (y >> ss_y) * (width >> ss_x) + (x >> ss_x);
            // SAFETY: The YCbCr struct has its bounds validated when it is constructed, and
            // it is immutable, so we know the length of the data array matches the
            // specified bounds.
            unsafe {
                let y_adj = (255.0 / 219.0) * (f32::from(*data[0].get_unchecked(y_pos)) - 16.0);
                let cr_adj = (255.0 / 224.0) * (f32::from(*data[1].get_unchecked(c_pos)) - 128.0);
                let cb_adj = (255.0 / 224.0) * (f32::from(*data[2].get_unchecked(c_pos)) - 128.0);

                let r = clamp(1.402f32.mul_add(cr_adj, y_adj), 0.0, 255.0) as u8;
                let g = clamp(
                    y_adj - 1.772 * 0.114 / 0.587 * cb_adj - 1.402 * 0.299 / 0.587 * cr_adj,
                    0.0,
                    255.0,
                ) as u8;
                let b = clamp(1.772f32.mul_add(cb_adj, y_adj), 0.0, 255.0) as u8;
                *output.get_unchecked_mut(y_pos) = [r, g, b];
            }
        }
    }
    output
}

/// # Errors
/// - If the specified transfer function is not implemented
pub fn rgb_gamma_encode(
    input: &[[f32; 3]],
    transfer: TransferCharacteristic,
) -> Result<Vec<[f32; 3]>> {
    todo!()
}

/// # Errors
/// - If the specified transfer function is not implemented
pub fn rgb_gamma_decode(
    input: &[[f32; 3]],
    transfer: TransferCharacteristic,
) -> Result<Vec<[f32; 3]>> {
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
        TransferCharacteristic::Linear => Ok(input.to_vec()),
        TransferCharacteristic::BT1886
        | TransferCharacteristic::ST170M
        | TransferCharacteristic::BT2020Ten
        | TransferCharacteristic::BT2020Twelve
        | TransferCharacteristic::BT470M
        | TransferCharacteristic::BT470BG => {
            let gamma: f32 = match transfer {
                TransferCharacteristic::BT1886
                | TransferCharacteristic::ST170M
                | TransferCharacteristic::BT2020Ten
                | TransferCharacteristic::BT2020Twelve => 2.4,
                TransferCharacteristic::BT470M => 2.2,
                TransferCharacteristic::BT470BG => 2.8,
                // Safety: We are only in this code block if the transfer is one of the above
                _ => unsafe { debug_unreachable!() },
            };
            Ok(input
                .iter()
                .map(|pix| {
                    [
                        to_linear_pow(pix[0], gamma),
                        to_linear_pow(pix[1], gamma),
                        to_linear_pow(pix[2], gamma),
                    ]
                })
                .collect())
        }
        TransferCharacteristic::SRGB => Ok(input
            .iter()
            .map(|pix| {
                [
                    to_linear_srgb(pix[0]),
                    to_linear_srgb(pix[1]),
                    to_linear_srgb(pix[2]),
                ]
            })
            .collect()),
        TransferCharacteristic::Reserved0
        | TransferCharacteristic::Unspecified
        | TransferCharacteristic::Reserved
        | TransferCharacteristic::ST240M
        | TransferCharacteristic::Logarithmic100
        | TransferCharacteristic::Logarithmic316
        | TransferCharacteristic::XVYCC
        | TransferCharacteristic::BT1361E
        | TransferCharacteristic::PerceptualQuantizer
        | TransferCharacteristic::ST428
        | TransferCharacteristic::HybridLogGamma => {
            bail!("Conversion not yet implemented for this transfer function")
        }
    }
}
