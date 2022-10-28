use std::ops::BitAnd;

use anyhow::{bail, Result};
use av_data::pixel::TransferCharacteristic;
use debug_unreachable::debug_unreachable;
use wide::{f32x4, CmpGe, CmpGt, CmpLe, CmpLt};

pub trait TransferFunction {
    fn to_linear(&self, input: &[[f32; 3]]) -> Result<Vec<[f32; 3]>>;
    fn to_gamma(&self, input: &[[f32; 3]]) -> Result<Vec<[f32; 3]>>;
}

impl TransferFunction for TransferCharacteristic {
    fn to_linear(&self, input: &[[f32; 3]]) -> Result<Vec<[f32; 3]>> {
        Ok(match *self {
            TransferCharacteristic::Logarithmic100 => input
                .iter()
                .map(|pix| {
                    let input = f32x4::new([pix[0], pix[1], pix[2], 0f32]);
                    let output = log100_inverse_oetf(input).to_array();
                    [output[0], output[1], output[2]]
                })
                .collect(),
            TransferCharacteristic::Logarithmic316 => input
                .iter()
                .map(|pix| {
                    let input = f32x4::new([pix[0], pix[1], pix[2], 0f32]);
                    let output = log316_inverse_oetf(input).to_array();
                    [output[0], output[1], output[2]]
                })
                .collect(),
            TransferCharacteristic::BT1886
            | TransferCharacteristic::ST170M
            | TransferCharacteristic::ST240M
            | TransferCharacteristic::BT2020Ten
            | TransferCharacteristic::BT2020Twelve => input
                .iter()
                .map(|pix| {
                    let input = f32x4::new([pix[0], pix[1], pix[2], 0f32]);
                    let output = rec_1886_eotf(input).to_array();
                    [output[0], output[1], output[2]]
                })
                .collect(),
            TransferCharacteristic::BT470M => input
                .iter()
                .map(|pix| {
                    let input = f32x4::new([pix[0], pix[1], pix[2], 0f32]);
                    let output = rec_470m_oetf(input).to_array();
                    [output[0], output[1], output[2]]
                })
                .collect(),
            TransferCharacteristic::BT470BG => input
                .iter()
                .map(|pix| {
                    let input = f32x4::new([pix[0], pix[1], pix[2], 0f32]);
                    let output = rec_470bg_oetf(input).to_array();
                    [output[0], output[1], output[2]]
                })
                .collect(),
            TransferCharacteristic::XVYCC => input
                .iter()
                .map(|pix| {
                    let input = f32x4::new([pix[0], pix[1], pix[2], 0f32]);
                    let output = xvycc_eotf(input).to_array();
                    [output[0], output[1], output[2]]
                })
                .collect(),
            TransferCharacteristic::SRGB => input
                .iter()
                .map(|pix| {
                    let input = f32x4::new([pix[0], pix[1], pix[2], 0f32]);
                    let output = srgb_eotf(input).to_array();
                    [output[0], output[1], output[2]]
                })
                .collect(),
            TransferCharacteristic::PerceptualQuantizer => input
                .iter()
                .map(|pix| {
                    let input = f32x4::new([pix[0], pix[1], pix[2], 0f32]);
                    let output = st_2084_inverse_oetf(input).to_array();
                    [output[0], output[1], output[2]]
                })
                .collect(),
            TransferCharacteristic::HybridLogGamma => input
                .iter()
                .map(|pix| {
                    let input = f32x4::new([pix[0], pix[1], pix[2], 0f32]);
                    let output = arib_b67_inverse_oetf(input).to_array();
                    [output[0], output[1], output[2]]
                })
                .collect(),
            TransferCharacteristic::Linear => input.to_owned(),
            // Unsupported
            TransferCharacteristic::Reserved0
            | TransferCharacteristic::Reserved
            | TransferCharacteristic::BT1361E
            | TransferCharacteristic::ST428 => {
                bail!("Cannot convert YUV<->RGB using this transfer function")
            }
            // SAFETY: We guess any unspecified data when beginning conversion
            TransferCharacteristic::Unspecified => unsafe { debug_unreachable!() },
        })
    }

    #[allow(clippy::too_many_lines)]
    fn to_gamma(&self, input: &[[f32; 3]]) -> Result<Vec<[f32; 3]>> {
        Ok(match *self {
            TransferCharacteristic::Logarithmic100 => input
                .iter()
                .map(|pix| {
                    let input = f32x4::new([pix[0], pix[1], pix[2], 0f32]);
                    let output = log100_oetf(input).to_array();
                    [output[0], output[1], output[2]]
                })
                .collect(),
            TransferCharacteristic::Logarithmic316 => input
                .iter()
                .map(|pix| {
                    let input = f32x4::new([pix[0], pix[1], pix[2], 0f32]);
                    let output = log316_oetf(input).to_array();
                    [output[0], output[1], output[2]]
                })
                .collect(),
            TransferCharacteristic::BT1886
            | TransferCharacteristic::ST170M
            | TransferCharacteristic::ST240M
            | TransferCharacteristic::BT2020Ten
            | TransferCharacteristic::BT2020Twelve => input
                .iter()
                .map(|pix| {
                    let input = f32x4::new([pix[0], pix[1], pix[2], 0f32]);
                    let output = rec_1886_inverse_eotf(input).to_array();
                    [output[0], output[1], output[2]]
                })
                .collect(),
            TransferCharacteristic::BT470M => input
                .iter()
                .map(|pix| {
                    let input = f32x4::new([pix[0], pix[1], pix[2], 0f32]);
                    let output = rec_470m_inverse_oetf(input).to_array();
                    [output[0], output[1], output[2]]
                })
                .collect(),
            TransferCharacteristic::BT470BG => input
                .iter()
                .map(|pix| {
                    let input = f32x4::new([pix[0], pix[1], pix[2], 0f32]);
                    let output = rec_470bg_inverse_oetf(input).to_array();
                    [output[0], output[1], output[2]]
                })
                .collect(),
            TransferCharacteristic::XVYCC => input
                .iter()
                .map(|pix| {
                    let input = f32x4::new([pix[0], pix[1], pix[2], 0f32]);
                    let output = xvycc_inverse_eotf(input).to_array();
                    [output[0], output[1], output[2]]
                })
                .collect(),
            TransferCharacteristic::SRGB => input
                .iter()
                .map(|pix| {
                    let input = f32x4::new([pix[0], pix[1], pix[2], 0f32]);
                    let output = srgb_inverse_eotf(input).to_array();
                    [output[0], output[1], output[2]]
                })
                .collect(),
            TransferCharacteristic::PerceptualQuantizer => input
                .iter()
                .map(|pix| {
                    let input = f32x4::new([pix[0], pix[1], pix[2], 0f32]);
                    let output = st_2084_oetf(input).to_array();
                    [output[0], output[1], output[2]]
                })
                .collect(),
            TransferCharacteristic::HybridLogGamma => input
                .iter()
                .map(|pix| {
                    let input = f32x4::new([pix[0], pix[1], pix[2], 0f32]);
                    let output = arib_b67_oetf(input).to_array();
                    [output[0], output[1], output[2]]
                })
                .collect(),
            TransferCharacteristic::Linear => input.to_owned(),
            // Unsupported
            TransferCharacteristic::Reserved0
            | TransferCharacteristic::Reserved
            | TransferCharacteristic::BT1361E
            | TransferCharacteristic::ST428 => {
                bail!("Cannot convert YUV<->RGB using this transfer function")
            }
            // SAFETY: We guess any unspecified data when beginning conversion
            TransferCharacteristic::Unspecified => unsafe { debug_unreachable!() },
        })
    }
}

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

#[inline(always)]
fn log100_oetf(x: f32x4) -> f32x4 {
    x.cmp_le(0.01).blend(f32x4::ZERO, 1.0 + x.log10() / 2.0)
}

#[inline(always)]
fn log100_inverse_oetf(x: f32x4) -> f32x4 {
    x.cmp_le(0.00).blend(
        f32x4::from(0.01),
        f32x4::from(10.0f32).pow_f32x4(2.0 * (x - 1.0)),
    )
}

#[inline(always)]
fn log316_oetf(x: f32x4) -> f32x4 {
    x.cmp_le(0.003_162_277_6)
        .blend(f32x4::ZERO, 1.0 + x.log10() / 2.5)
}

#[inline(always)]
fn log316_inverse_oetf(x: f32x4) -> f32x4 {
    x.cmp_le(f32x4::ZERO).blend(
        f32x4::from(0.003_162_277_6),
        f32x4::from(10.0f32).pow_f32x4(2.5 * (x - 1.0)),
    )
}

// Ignore the BT.1886 provisions for limited contrast and assume an ideal CRT.
#[inline(always)]
fn rec_1886_eotf(x: f32x4) -> f32x4 {
    x.cmp_lt(f32x4::ZERO).blend(f32x4::ZERO, x.powf(2.4))
}

#[inline(always)]
fn rec_1886_inverse_eotf(x: f32x4) -> f32x4 {
    x.cmp_lt(f32x4::ZERO).blend(f32x4::ZERO, x.powf(1.0 / 2.4))
}

#[inline(always)]
fn rec_470m_oetf(x: f32x4) -> f32x4 {
    x.cmp_lt(f32x4::ZERO).blend(f32x4::ZERO, x.powf(2.2))
}

#[inline(always)]
fn rec_470m_inverse_oetf(x: f32x4) -> f32x4 {
    x.cmp_lt(f32x4::ZERO).blend(f32x4::ZERO, x.powf(1.0 / 2.2))
}

#[inline(always)]
fn rec_470bg_oetf(x: f32x4) -> f32x4 {
    x.cmp_lt(f32x4::ZERO).blend(f32x4::ZERO, x.powf(2.8))
}

#[inline(always)]
fn rec_470bg_inverse_oetf(x: f32x4) -> f32x4 {
    x.cmp_lt(f32x4::ZERO).blend(f32x4::ZERO, x.powf(1.0 / 2.8))
}

#[inline(always)]
fn rec_709_oetf(x: f32x4) -> f32x4 {
    let x = x.fast_max(f32x4::ZERO);

    x.cmp_lt(f32x4::from(REC709_BETA)).blend(
        x * 4.5,
        // REC709_ALPHA * x.powf(0.45) - (REC709_ALPHA - 1.0)
        f32x4::from(REC709_ALPHA).mul_sub(x.powf(0.45), f32x4::from(REC709_ALPHA - 1.0)),
    )
}

#[inline(always)]
fn rec_709_inverse_oetf(x: f32x4) -> f32x4 {
    let x = x.fast_max(f32x4::ZERO);

    x.cmp_lt(f32x4::from(4.5 * REC709_BETA)).blend(
        x / 4.5,
        ((x + (REC709_ALPHA - 1.0)) / REC709_ALPHA).powf(1.0 / 0.45),
    )
}

#[inline(always)]
fn xvycc_eotf(x: f32x4) -> f32x4 {
    x.cmp_ge(f32x4::ZERO).bitand(x.cmp_lt(f32x4::ONE)).blend(
        rec_1886_eotf(x.abs()).copysign(x),
        rec_709_inverse_oetf(x.abs()).copysign(x),
    )
}

#[inline(always)]
fn xvycc_inverse_eotf(x: f32x4) -> f32x4 {
    x.cmp_ge(f32x4::ZERO).bitand(x.cmp_lt(f32x4::ONE)).blend(
        rec_1886_inverse_eotf(x.abs()).copysign(x),
        rec_709_oetf(x.abs()).copysign(x),
    )
}

#[inline(always)]
fn srgb_eotf(x: f32x4) -> f32x4 {
    let x = x.fast_max(f32x4::ZERO);

    x.cmp_lt(f32x4::from(12.92 * SRGB_BETA))
        .blend(x / 12.92, ((x + (SRGB_ALPHA - 1.0)) / SRGB_ALPHA).powf(2.4))
}

#[inline(always)]
fn srgb_inverse_eotf(x: f32x4) -> f32x4 {
    let x = x.fast_max(f32x4::ZERO);

    x.cmp_lt(SRGB_BETA).blend(
        x * 12.92,
        // SRGB_ALPHA * x.powf(1.0 / 2.4) - (SRGB_ALPHA - 1.0)
        f32x4::from(SRGB_ALPHA).mul_sub(x.powf(1.0 / 2.4), f32x4::from(SRGB_ALPHA - 1.0)),
    )
}

#[inline(always)]
fn st_2084_inverse_eotf(x: f32x4) -> f32x4 {
    // Filter negative values to avoid NAN, and also special-case 0 so that (f(g(0))
    // == 0).

    x.cmp_gt(f32x4::ZERO).blend(
        {
            let xpow = x.powf(ST2084_M1);

            // More stable arrangement that avoids some cancellation error.
            // (ST2084_C1 - 1.0) + (ST2084_C2 - ST2084_C3) * xpow
            let num =
                f32x4::from(ST2084_C2 - ST2084_C3).mul_add(xpow, f32x4::from(ST2084_C1 - 1.0));
            // 1.0 + ST2084_C3 * xpow
            let den = f32x4::from(ST2084_C3).mul_add(xpow, f32x4::ONE);
            (1.0 + num / den).powf(ST2084_M2)
        },
        f32x4::ZERO,
    )
}

#[inline(always)]
fn inverse_ootf_st2084(x: f32x4) -> f32x4 {
    rec_709_inverse_oetf(rec_1886_inverse_eotf(x * 100.0)) / ST2084_OOTF_SCALE
}

#[inline(always)]
fn ootf_st2084(x: f32x4) -> f32x4 {
    rec_1886_eotf(rec_709_oetf(x * ST2084_OOTF_SCALE)) / 100.0
}

#[inline(always)]
fn st_2084_eotf(x: f32x4) -> f32x4 {
    x.cmp_gt(f32x4::ZERO).blend(
        {
            let xpow = x.powf(1.0 / ST2084_M2);
            let num = (xpow - ST2084_C1).fast_max(f32x4::ZERO);
            // ST2084_C2 - (ST2084_C3 * xpow)
            let den = xpow
                .mul_neg_add(f32x4::from(ST2084_C3), f32x4::from(ST2084_C2))
                .fast_max(f32x4::from(f32::EPSILON));
            (num / den).powf(1.0 / ST2084_M1)
        },
        f32x4::ZERO,
    )
}

#[inline(always)]
fn st_2084_inverse_oetf(x: f32x4) -> f32x4 {
    inverse_ootf_st2084(st_2084_eotf(x))
}

#[inline(always)]
fn st_2084_oetf(x: f32x4) -> f32x4 {
    st_2084_inverse_eotf(ootf_st2084(x))
}

#[inline(always)]
fn arib_b67_inverse_oetf(x: f32x4) -> f32x4 {
    let x = x.fast_max(f32x4::ZERO);

    x.cmp_le(f32x4::from(0.5)).blend(
        (x * x) * (1.0 / 3.0),
        (((x - ARIB_B67_C) / ARIB_B67_A).exp() + ARIB_B67_B) / 12.0,
    )
}

#[inline(always)]
fn arib_b67_oetf(x: f32x4) -> f32x4 {
    let x = x.fast_max(f32x4::ZERO);

    x.cmp_le(1.0 / 12.0).blend(
        (3.0 * x).sqrt(),
        // ARIB_B67_A * (12.0 * x - ARIB_B67_B).ln() + ARIB_B67_C
        f32x4::from(ARIB_B67_A).mul_add(
            f32x4::from(12.0).mul_sub(x, f32x4::from(ARIB_B67_B)).ln(),
            f32x4::from(ARIB_B67_C),
        ),
    )
}
