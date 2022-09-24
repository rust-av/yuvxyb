use anyhow::{bail, Result};
use av_data::pixel::TransferCharacteristic;
use debug_unreachable::debug_unreachable;

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
                    [
                        log100_inverse_oetf(pix[0]),
                        log100_inverse_oetf(pix[1]),
                        log100_inverse_oetf(pix[2]),
                    ]
                })
                .collect(),
            TransferCharacteristic::Logarithmic316 => input
                .iter()
                .map(|pix| {
                    [
                        log316_inverse_oetf(pix[0]),
                        log316_inverse_oetf(pix[1]),
                        log316_inverse_oetf(pix[2]),
                    ]
                })
                .collect(),
            TransferCharacteristic::BT1886
            | TransferCharacteristic::ST170M
            | TransferCharacteristic::ST240M
            | TransferCharacteristic::BT2020Ten
            | TransferCharacteristic::BT2020Twelve => input
                .iter()
                .map(|pix| {
                    [
                        rec_1886_eotf(pix[0]),
                        rec_1886_eotf(pix[1]),
                        rec_1886_eotf(pix[2]),
                    ]
                })
                .collect(),
            TransferCharacteristic::BT470M => input
                .iter()
                .map(|pix| {
                    [
                        rec_470m_oetf(pix[0]),
                        rec_470m_oetf(pix[1]),
                        rec_470m_oetf(pix[2]),
                    ]
                })
                .collect(),
            TransferCharacteristic::BT470BG => input
                .iter()
                .map(|pix| {
                    [
                        rec_470bg_oetf(pix[0]),
                        rec_470bg_oetf(pix[1]),
                        rec_470bg_oetf(pix[2]),
                    ]
                })
                .collect(),
            TransferCharacteristic::XVYCC => input
                .iter()
                .map(|pix| [xvycc_eotf(pix[0]), xvycc_eotf(pix[1]), xvycc_eotf(pix[2])])
                .collect(),
            TransferCharacteristic::SRGB => input
                .iter()
                .map(|pix| [srgb_eotf(pix[0]), srgb_eotf(pix[1]), srgb_eotf(pix[2])])
                .collect(),
            TransferCharacteristic::PerceptualQuantizer => input
                .iter()
                .map(|pix| {
                    [
                        st_2084_inverse_oetf(pix[0]),
                        st_2084_inverse_oetf(pix[1]),
                        st_2084_inverse_oetf(pix[2]),
                    ]
                })
                .collect(),
            TransferCharacteristic::HybridLogGamma => input
                .iter()
                .map(|pix| {
                    [
                        arib_b67_inverse_oetf(pix[0]),
                        arib_b67_inverse_oetf(pix[1]),
                        arib_b67_inverse_oetf(pix[2]),
                    ]
                })
                .collect(),
            // Unsupported
            TransferCharacteristic::Reserved0
            | TransferCharacteristic::Reserved
            | TransferCharacteristic::Linear
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
                    [
                        log100_oetf(pix[0]),
                        log100_oetf(pix[1]),
                        log100_oetf(pix[2]),
                    ]
                })
                .collect(),
            TransferCharacteristic::Logarithmic316 => input
                .iter()
                .map(|pix| {
                    [
                        log316_oetf(pix[0]),
                        log316_oetf(pix[1]),
                        log316_oetf(pix[2]),
                    ]
                })
                .collect(),
            TransferCharacteristic::BT1886
            | TransferCharacteristic::ST170M
            | TransferCharacteristic::ST240M
            | TransferCharacteristic::BT2020Ten
            | TransferCharacteristic::BT2020Twelve => input
                .iter()
                .map(|pix| {
                    [
                        rec_1886_inverse_eotf(pix[0]),
                        rec_1886_inverse_eotf(pix[1]),
                        rec_1886_inverse_eotf(pix[2]),
                    ]
                })
                .collect(),
            TransferCharacteristic::BT470M => input
                .iter()
                .map(|pix| {
                    [
                        rec_470m_inverse_oetf(pix[0]),
                        rec_470m_inverse_oetf(pix[1]),
                        rec_470m_inverse_oetf(pix[2]),
                    ]
                })
                .collect(),
            TransferCharacteristic::BT470BG => input
                .iter()
                .map(|pix| {
                    [
                        rec_470bg_inverse_oetf(pix[0]),
                        rec_470bg_inverse_oetf(pix[1]),
                        rec_470bg_inverse_oetf(pix[2]),
                    ]
                })
                .collect(),
            TransferCharacteristic::XVYCC => input
                .iter()
                .map(|pix| {
                    [
                        xvycc_inverse_eotf(pix[0]),
                        xvycc_inverse_eotf(pix[1]),
                        xvycc_inverse_eotf(pix[2]),
                    ]
                })
                .collect(),
            TransferCharacteristic::SRGB => input
                .iter()
                .map(|pix| {
                    [
                        srgb_inverse_eotf(pix[0]),
                        srgb_inverse_eotf(pix[1]),
                        srgb_inverse_eotf(pix[2]),
                    ]
                })
                .collect(),
            TransferCharacteristic::PerceptualQuantizer => input
                .iter()
                .map(|pix| {
                    [
                        st_2084_oetf(pix[0]),
                        st_2084_oetf(pix[1]),
                        st_2084_oetf(pix[2]),
                    ]
                })
                .collect(),
            TransferCharacteristic::HybridLogGamma => input
                .iter()
                .map(|pix| {
                    [
                        arib_b67_oetf(pix[0]),
                        arib_b67_oetf(pix[1]),
                        arib_b67_oetf(pix[2]),
                    ]
                })
                .collect(),
            // Unsupported
            TransferCharacteristic::Reserved0
            | TransferCharacteristic::Reserved
            | TransferCharacteristic::Linear
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
fn log100_oetf(x: f32) -> f32 {
    if x <= 0.01 {
        0.0
    } else {
        1.0 + x.log10() / 2.0
    }
}

#[inline(always)]
fn log100_inverse_oetf(x: f32) -> f32 {
    if x <= 0.0 {
        0.01
    } else {
        10.0f32.powf(2.0 * (x - 1.0))
    }
}

#[inline(always)]
fn log316_oetf(x: f32) -> f32 {
    if x <= 0.003_162_277_6 {
        0.0
    } else {
        1.0 + x.log10() / 2.5
    }
}

#[inline(always)]
fn log316_inverse_oetf(x: f32) -> f32 {
    if x <= 0.0 {
        0.003_162_277_6
    } else {
        10.0f32.powf(2.5 * (x - 1.0))
    }
}

// Ignore the BT.1886 provisions for limited contrast and assume an ideal CRT.
#[inline(always)]
fn rec_1886_eotf(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else {
        x.powf(2.4)
    }
}

#[inline(always)]
fn rec_1886_inverse_eotf(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else {
        x.powf(1.0 / 2.4)
    }
}

#[inline(always)]
fn rec_470m_oetf(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else {
        x.powf(2.2)
    }
}

#[inline(always)]
fn rec_470m_inverse_oetf(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else {
        x.powf(1.0 / 2.2)
    }
}

#[inline(always)]
fn rec_470bg_oetf(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else {
        x.powf(2.8)
    }
}

#[inline(always)]
fn rec_470bg_inverse_oetf(x: f32) -> f32 {
    if x < 0.0 {
        0.0
    } else {
        x.powf(1.0 / 2.8)
    }
}

#[inline(always)]
fn rec_709_oetf(x: f32) -> f32 {
    let x = x.max(0.0);

    if x < REC709_BETA {
        x * 4.5
    } else {
        // REC709_ALPHA * x.powf(0.45) - (REC709_ALPHA - 1.0)
        REC709_ALPHA.mul_add(x.powf(0.45), -(REC709_ALPHA - 1.0))
    }
}

#[inline(always)]
fn rec_709_inverse_oetf(x: f32) -> f32 {
    let x = x.max(0.0);

    if x < 4.5 * REC709_BETA {
        x / 4.5
    } else {
        ((x + (REC709_ALPHA - 1.0)) / REC709_ALPHA).powf(1.0 / 0.45)
    }
}

#[inline(always)]
fn xvycc_eotf(x: f32) -> f32 {
    if (0.0..=1.0).contains(&x) {
        rec_1886_eotf(x.abs()).copysign(x)
    } else {
        rec_709_inverse_oetf(x.abs()).copysign(x)
    }
}

#[inline(always)]
fn xvycc_inverse_eotf(x: f32) -> f32 {
    if (0.0..=1.0).contains(&x) {
        rec_1886_inverse_eotf(x.abs()).copysign(x)
    } else {
        rec_709_oetf(x.abs()).copysign(x)
    }
}

#[inline(always)]
fn srgb_eotf(x: f32) -> f32 {
    let x = x.max(0.0);

    if x < 12.92 * SRGB_BETA {
        x / 12.92
    } else {
        ((x + (SRGB_ALPHA - 1.0)) / SRGB_ALPHA).powf(2.4)
    }
}

#[inline(always)]
fn srgb_inverse_eotf(x: f32) -> f32 {
    let x = x.max(0.0);

    if x < SRGB_BETA {
        x * 12.92
    } else {
        // SRGB_ALPHA * x.powf(1.0 / 2.4) - (SRGB_ALPHA - 1.0)
        SRGB_ALPHA.mul_add(x.powf(1.0 / 2.4), -(SRGB_ALPHA - 1.0))
    }
}

#[inline(always)]
fn st_2084_inverse_eotf(x: f32) -> f32 {
    // Filter negative values to avoid NAN, and also special-case 0 so that (f(g(0))
    // == 0).

    if x > 0.0 {
        let xpow = x.powf(ST2084_M1);

        // More stable arrangement that avoids some cancellation error.
        // (ST2084_C1 - 1.0) + (ST2084_C2 - ST2084_C3) * xpow
        let num = (ST2084_C2 - ST2084_C3).mul_add(xpow, ST2084_C1 - 1.0);
        // 1.0 + ST2084_C3 * xpow
        let den = ST2084_C3.mul_add(xpow, 1.0);
        (1.0 + num / den).powf(ST2084_M2)
    } else {
        0.0
    }
}

#[inline(always)]
fn inverse_ootf_st2084(x: f32) -> f32 {
    rec_709_inverse_oetf(rec_1886_inverse_eotf(x * 100.0)) / ST2084_OOTF_SCALE
}

#[inline(always)]
fn ootf_st2084(x: f32) -> f32 {
    rec_1886_eotf(rec_709_oetf(x * ST2084_OOTF_SCALE)) / 100.0
}

#[inline(always)]
fn st_2084_eotf(x: f32) -> f32 {
    if x > 0.0 {
        let xpow = x.powf(1.0 / ST2084_M2);
        let num = (xpow - ST2084_C1).max(0.0);
        let den = (ST2084_C2 - ST2084_C3 * xpow).max(f32::EPSILON);
        (num / den).powf(1.0 / ST2084_M1)
    } else {
        0.0
    }
}

#[inline(always)]
fn st_2084_inverse_oetf(x: f32) -> f32 {
    inverse_ootf_st2084(st_2084_eotf(x))
}

#[inline(always)]
fn st_2084_oetf(x: f32) -> f32 {
    st_2084_inverse_eotf(ootf_st2084(x))
}

#[inline(always)]
fn arib_b67_inverse_oetf(x: f32) -> f32 {
    let x = x.max(0.0);

    if x <= 0.5 {
        (x * x) * (1.0 / 3.0)
    } else {
        (((x - ARIB_B67_C) / ARIB_B67_A).exp() + ARIB_B67_B) / 12.0
    }
}

#[inline(always)]
fn arib_b67_oetf(x: f32) -> f32 {
    let x = x.max(0.0);

    if x <= 1.0 / 12.0 {
        (3.0 * x).sqrt()
    } else {
        // ARIB_B67_A * (12.0 * x - ARIB_B67_B).ln() + ARIB_B67_C
        ARIB_B67_A.mul_add((12.0f32.mul_add(x, -ARIB_B67_B)).ln(), ARIB_B67_C)
    }
}
