#[cfg(test)]
mod tests;

use av_data::pixel::TransferCharacteristic;
use std::slice::from_raw_parts_mut;

use crate::ConversionError;
use yuvxyb_math::{expf, powf};

pub trait TransferFunction {
    fn to_linear(&self, input: Vec<[f32; 3]>) -> Result<Vec<[f32; 3]>, ConversionError>;
    fn to_gamma(&self, input: Vec<[f32; 3]>) -> Result<Vec<[f32; 3]>, ConversionError>;
}

impl TransferFunction for TransferCharacteristic {
    fn to_linear(&self, input: Vec<[f32; 3]>) -> Result<Vec<[f32; 3]>, ConversionError> {
        Ok(match *self {
            Self::Logarithmic100 => image_log100_gamma_to_lin(input),
            Self::Logarithmic316 => image_log316_gamma_to_lin(input),
            Self::BT1886 | Self::ST170M | Self::ST240M | Self::BT2020Ten | Self::BT2020Twelve => {
                image_rec_1886_gamma_to_lin(input)
            }
            Self::BT470M => image_rec_470m_gamma_to_lin(input),
            Self::BT470BG => image_rec_470bg_gamma_to_lin(input),
            Self::XVYCC => image_xvycc_gamma_to_lin(input),
            Self::SRGB => image_srgb_gamma_to_lin(input),
            Self::PerceptualQuantizer => image_st_2084_gamma_to_lin(input),
            Self::HybridLogGamma => image_arib_b67_gamma_to_lin(input),
            Self::Linear => input,
            Self::Reserved0 | Self::Reserved | Self::BT1361E | Self::ST428 => {
                return Err(ConversionError::UnsupportedTransferCharacteristic)
            }
            Self::Unspecified => return Err(ConversionError::UnspecifiedTransferCharacteristic),
        })
    }

    #[allow(clippy::too_many_lines)]
    fn to_gamma(&self, input: Vec<[f32; 3]>) -> Result<Vec<[f32; 3]>, ConversionError> {
        Ok(match *self {
            Self::Logarithmic100 => image_log100_lin_to_gamma(input),
            Self::Logarithmic316 => image_log316_lin_to_gamma(input),
            Self::BT1886 | Self::ST170M | Self::ST240M | Self::BT2020Ten | Self::BT2020Twelve => {
                image_rec_1886_lin_to_gamma(input)
            }
            Self::BT470M => image_rec_470m_lin_to_gamma(input),
            Self::BT470BG => image_rec_470bg_lin_to_gamma(input),
            Self::XVYCC => image_xvycc_lin_to_gamma(input),
            Self::SRGB => image_srgb_lin_to_gamma(input),
            Self::PerceptualQuantizer => image_st_2084_lin_to_gamma(input),
            Self::HybridLogGamma => image_arib_b67_lin_to_gamma(input),
            Self::Linear => input,
            // Unsupported
            Self::Reserved0 | Self::Reserved | Self::BT1361E | Self::ST428 => {
                return Err(ConversionError::UnsupportedTransferCharacteristic)
            }
            Self::Unspecified => return Err(ConversionError::UnspecifiedTransferCharacteristic),
        })
    }
}

macro_rules! image_transfer_fn {
    ($image_name:ident, $name:ident) => {
        fn $image_name(mut input: Vec<[f32; 3]>) -> Vec<[f32; 3]> {
            // SAFETY: Referencing preallocated memory (input)
            let input_flat =
                unsafe { from_raw_parts_mut(input.as_mut_ptr().cast::<f32>(), input.len() * 3) };

            for val in input_flat {
                *val = $name(*val);
            }

            input
        }
    };
}

image_transfer_fn!(image_log100_gamma_to_lin, log100_gamma_to_lin);
image_transfer_fn!(image_log316_gamma_to_lin, log316_gamma_to_lin);
image_transfer_fn!(image_rec_1886_gamma_to_lin, rec_1886_gamma_to_lin);
image_transfer_fn!(image_rec_470m_gamma_to_lin, rec_470m_gamma_to_lin);
image_transfer_fn!(image_rec_470bg_gamma_to_lin, rec_470bg_gamma_to_lin);
image_transfer_fn!(image_xvycc_gamma_to_lin, xvycc_gamma_to_lin);
image_transfer_fn!(image_srgb_gamma_to_lin, srgb_gamma_to_lin);
image_transfer_fn!(image_st_2084_gamma_to_lin, st_2084_gamma_to_lin);
image_transfer_fn!(image_arib_b67_gamma_to_lin, arib_b67_gamma_to_lin);

image_transfer_fn!(image_log100_lin_to_gamma, log100_lin_to_gamma);
image_transfer_fn!(image_log316_lin_to_gamma, log316_lin_to_gamma);
image_transfer_fn!(image_rec_1886_lin_to_gamma, rec_1886_lin_to_gamma);
image_transfer_fn!(image_rec_470m_lin_to_gamma, rec_470m_lin_to_gamma);
image_transfer_fn!(image_rec_470bg_lin_to_gamma, rec_470bg_lin_to_gamma);
image_transfer_fn!(image_xvycc_lin_to_gamma, xvycc_lin_to_gamma);
image_transfer_fn!(image_srgb_lin_to_gamma, srgb_lin_to_gamma);
image_transfer_fn!(image_st_2084_lin_to_gamma, st_2084_lin_to_gamma);
image_transfer_fn!(image_arib_b67_lin_to_gamma, arib_b67_lin_to_gamma);

const REC709_ALPHA: f32 = 1.099_296_8;
const REC709_BETA: f32 = 0.018_053_97;

// Adjusted for continuity of first derivative.
const SRGB_ALPHA: f32 = 1.055_010_7;
const SRGB_BETA: f32 = 0.003_041_282_5;

const ST2084_M1: f32 = 0.159_301_76;
const ST2084_M2: f32 = 78.84375;
const ST2084_C1: f32 = 0.835_937_5;
const ST2084_C2: f32 = 18.851_563;
const ST2084_C3: f32 = 18.6875;

// Chosen for compatibility with higher precision REC709_ALPHA/REC709_BETA.
// See: ITU-R BT.2390-2 5.3.1
const ST2084_OOTF_SCALE: f32 = 59.490_803;

const ARIB_B67_A: f32 = 0.178_832_77;
const ARIB_B67_B: f32 = 0.284_668_92;
const ARIB_B67_C: f32 = 0.559_910_7;
fn log100_lin_to_gamma(x: f32) -> f32 {
    if x <= 0.01 {
        0.0
    } else {
        1.0 + x.log10() / 2.0
    }
}

fn log100_gamma_to_lin(x: f32) -> f32 {
    if x <= 0.0 {
        0.01
    } else {
        powf(10.0, 2.0 * (x - 1.0))
    }
}

fn log316_lin_to_gamma(x: f32) -> f32 {
    if x <= 0.003_162_277_6 {
        0.0
    } else {
        1.0 + x.log10() / 2.5
    }
}

fn log316_gamma_to_lin(x: f32) -> f32 {
    if x <= 0.0 {
        0.003_162_277_6
    } else {
        powf(10.0, 2.5 * (x - 1.0))
    }
}

// Ignore the BT.1886 provisions for limited contrast and assume an ideal CRT.
fn rec_1886_gamma_to_lin(x: f32) -> f32 {
    if x <= 0.0 {
        0.0
    } else {
        powf(x, 2.4)
    }
}

fn rec_1886_lin_to_gamma(x: f32) -> f32 {
    if x <= 0.0 {
        0.0
    } else {
        powf(x, 1.0 / 2.4)
    }
}

fn rec_470m_gamma_to_lin(x: f32) -> f32 {
    if x <= 0.0 {
        0.0
    } else {
        powf(x, 2.2)
    }
}

fn rec_470m_lin_to_gamma(x: f32) -> f32 {
    if x <= 0.0 {
        0.0
    } else {
        powf(x, 1.0 / 2.2)
    }
}

fn rec_470bg_gamma_to_lin(x: f32) -> f32 {
    if x <= 0.0 {
        0.0
    } else {
        powf(x, 2.8)
    }
}

fn rec_470bg_lin_to_gamma(x: f32) -> f32 {
    if x <= 0.0 {
        0.0
    } else {
        powf(x, 1.0 / 2.8)
    }
}

fn rec_709_oetf(x: f32) -> f32 {
    let x = x.max(0.0);

    if x < REC709_BETA {
        x * 4.5
    } else {
        // REC709_ALPHA * x.powf(0.45) - (REC709_ALPHA - 1.0)
        REC709_ALPHA.mul_add(powf(x, 0.45), -(REC709_ALPHA - 1.0))
    }
}

fn rec_709_inverse_oetf(x: f32) -> f32 {
    let x = x.max(0.0);

    if x < 4.5 * REC709_BETA {
        x / 4.5
    } else {
        powf((x + (REC709_ALPHA - 1.0)) / REC709_ALPHA, 1.0 / 0.45)
    }
}

fn xvycc_gamma_to_lin(x: f32) -> f32 {
    if (0.0..=1.0).contains(&x) {
        rec_1886_gamma_to_lin(x.abs()).copysign(x)
    } else {
        rec_709_inverse_oetf(x.abs()).copysign(x)
    }
}

fn xvycc_lin_to_gamma(x: f32) -> f32 {
    if (0.0..=1.0).contains(&x) {
        rec_1886_lin_to_gamma(x.abs()).copysign(x)
    } else {
        rec_709_oetf(x.abs()).copysign(x)
    }
}

fn srgb_gamma_to_lin(x: f32) -> f32 {
    let x = x.max(0.0);

    if x < 12.92 * SRGB_BETA {
        x / 12.92
    } else {
        powf((x + (SRGB_ALPHA - 1.0)) / SRGB_ALPHA, 2.4)
    }
}

fn srgb_lin_to_gamma(x: f32) -> f32 {
    let x = x.max(0.0);

    if x < SRGB_BETA {
        x * 12.92
    } else {
        // SRGB_ALPHA * x.powf(1.0 / 2.4) - (SRGB_ALPHA - 1.0)
        SRGB_ALPHA.mul_add(powf(x, 1.0 / 2.4), -(SRGB_ALPHA - 1.0))
    }
}

fn st_2084_inverse_eotf(x: f32) -> f32 {
    // Filter negative values to avoid NAN, and also special-case 0 so that (f(g(0))
    // == 0).

    if x > 0.0 {
        let xpow = powf(x, ST2084_M1);

        // More stable arrangement that avoids some cancellation error.
        // (ST2084_C1 - 1.0) + (ST2084_C2 - ST2084_C3) * xpow
        let num = (ST2084_C2 - ST2084_C3).mul_add(xpow, ST2084_C1 - 1.0);
        // 1.0 + ST2084_C3 * xpow
        let den = ST2084_C3.mul_add(xpow, 1.0);
        powf(1.0 + num / den, ST2084_M2)
    } else {
        0.0
    }
}

fn inverse_ootf_st2084(x: f32) -> f32 {
    rec_709_inverse_oetf(rec_1886_lin_to_gamma(x * 100.0)) / ST2084_OOTF_SCALE
}

fn ootf_st2084(x: f32) -> f32 {
    rec_1886_gamma_to_lin(rec_709_oetf(x * ST2084_OOTF_SCALE)) / 100.0
}

fn st_2084_eotf(x: f32) -> f32 {
    if x > 0.0 {
        let xpow = powf(x, 1.0 / ST2084_M2);
        let num = (xpow - ST2084_C1).max(0.0);
        let den = ST2084_C3.mul_add(-xpow, ST2084_C2).max(f32::EPSILON);
        powf(num / den, 1.0 / ST2084_M1)
    } else {
        0.0
    }
}

fn st_2084_gamma_to_lin(x: f32) -> f32 {
    inverse_ootf_st2084(st_2084_eotf(x))
}

fn st_2084_lin_to_gamma(x: f32) -> f32 {
    st_2084_inverse_eotf(ootf_st2084(x))
}

fn arib_b67_gamma_to_lin(x: f32) -> f32 {
    let x = x.max(0.0);

    if x <= 0.5 {
        (x * x) * (1.0 / 3.0)
    } else {
        (expf((x - ARIB_B67_C) / ARIB_B67_A) + ARIB_B67_B) / 12.0
    }
}

fn arib_b67_lin_to_gamma(x: f32) -> f32 {
    let x = x.max(0.0);

    if x <= 1.0 / 12.0 {
        (3.0 * x).sqrt()
    } else {
        // ARIB_B67_A * (12.0 * x - ARIB_B67_B).ln() + ARIB_B67_C
        ARIB_B67_A.mul_add((12.0f32.mul_add(x, -ARIB_B67_B)).ln(), ARIB_B67_C)
    }
}
