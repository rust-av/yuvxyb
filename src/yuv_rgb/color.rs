use anyhow::{bail, Result};
use av_data::pixel::{ColorPrimaries, MatrixCoefficients};
use debug_unreachable::debug_unreachable;
use nalgebra::{Matrix1x3, Matrix3};

pub(crate) fn ncl_rgb_to_yuv_matrix_from_primaries(
    primaries: ColorPrimaries,
) -> Result<Matrix3<f32>> {
    match primaries {
        ColorPrimaries::BT709 => ncl_rgb_to_yuv_matrix(MatrixCoefficients::BT709),
        ColorPrimaries::BT2020 => {
            ncl_rgb_to_yuv_matrix(MatrixCoefficients::BT2020NonConstantLuminance)
        }
        _ => {
            let (kr, kb) = get_yuv_constants_from_primaries(primaries)?;
            Ok(ncl_rgb_to_yuv_matrix_from_kr_kb(kr, kb))
        }
    }
}

pub(crate) fn ncl_rgb_to_yuv_matrix(matrix: MatrixCoefficients) -> Result<Matrix3<f32>> {
    Ok(match matrix {
        MatrixCoefficients::YCgCo => {
            Matrix3::from_row_slice(&[0.25, 0.5, 0.25, -0.25, 0.5, -0.25, 0.5, 0.0, -0.5])
        }
        MatrixCoefficients::ST2085 => Matrix3::from_row_slice(&[
            1688.0 / 4096.0,
            2146.0 / 4096.0,
            262.0 / 4096.0,
            683.0 / 4096.0,
            2951.0 / 4096.0,
            462.0 / 4096.0,
            99.0 / 4096.0,
            309.0 / 4096.0,
            3688.0 / 4096.0,
        ]),
        _ => {
            let (kr, kb) = get_yuv_constants(matrix)?;
            ncl_rgb_to_yuv_matrix_from_kr_kb(kr, kb)
        }
    })
}

pub(crate) fn get_yuv_constants_from_primaries(primaries: ColorPrimaries) -> Result<(f32, f32)> {
    // ITU-T H.265 Annex E, Eq (E-22) to (E-27).
    let primaries_xy = get_primaries_xy(primaries)?;

    let r_xyz = Matrix1x3::from_row_slice(&xy_to_xyz(primaries_xy[0][0], primaries_xy[0][1]));
    let g_xyz = Matrix1x3::from_row_slice(&xy_to_xyz(primaries_xy[1][0], primaries_xy[1][1]));
    let b_xyz = Matrix1x3::from_row_slice(&xy_to_xyz(primaries_xy[2][0], primaries_xy[2][1]));
    let white_xyz = Matrix1x3::from_row_slice(&get_white_point(primaries));

    let x_rgb = Matrix1x3::from_row_slice(&[r_xyz[0], g_xyz[0], b_xyz[0]]);
    let y_rgb = Matrix1x3::from_row_slice(&[r_xyz[1], g_xyz[1], b_xyz[1]]);
    let z_rgb = Matrix1x3::from_row_slice(&[r_xyz[2], g_xyz[2], b_xyz[2]]);

    let denom = x_rgb.dot(&y_rgb.cross(&z_rgb));
    let kr = white_xyz.dot(&g_xyz.cross(&b_xyz)) / denom;
    let kb = white_xyz.dot(&r_xyz.cross(&g_xyz)) / denom;

    Ok((kr, kb))
}

pub(crate) fn get_yuv_constants(mut matrix: MatrixCoefficients) -> Result<(f32, f32)> {
    Ok(match matrix {
        MatrixCoefficients::Identity => (0.0, 0.0),
        MatrixCoefficients::BT470M => (0.3, 0.11),
        MatrixCoefficients::ST240M => (0.212, 0.087),
        MatrixCoefficients::BT470BG | MatrixCoefficients::ST170M => (0.299, 0.114),
        MatrixCoefficients::BT709 => (0.2126, 0.0722),
        MatrixCoefficients::BT2020NonConstantLuminance
        | MatrixCoefficients::BT2020ConstantLuminance => (0.2627, 0.0593),
        // Unusable
        MatrixCoefficients::Reserved
        | MatrixCoefficients::YCgCo
        | MatrixCoefficients::ST2085
        | MatrixCoefficients::ChromaticityDerivedNonConstantLuminance
        | MatrixCoefficients::ChromaticityDerivedConstantLuminance
        | MatrixCoefficients::ICtCp => {
            bail!("Cannot convert YUV<->RGB using these matrix coefficients")
        }
        // SAFETY: We guess any unspecified data when beginning conversion
        MatrixCoefficients::Unspecified => unsafe { debug_unreachable!() },
    })
}

pub(crate) fn ncl_rgb_to_yuv_matrix_from_kr_kb(kr: f32, kb: f32) -> Matrix3<f32> {
    let mut ret = [0.0; 9];
    let kg = 1.0 - kr - kb;
    let uscale = 1.0 / (2.0 - 2.0 * kb);
    let vscale = 1.0 / (2.0 - 2.0 * kr);

    ret[0] = kr;
    ret[1] = kg;
    ret[2] = kb;

    ret[3] = -kr * uscale;
    ret[4] = -kg * uscale;
    ret[5] = (1.0 - kb) * uscale;

    ret[6] = (1.0 - kr) * vscale;
    ret[7] = -kg * vscale;
    ret[8] = -kb * vscale;

    Matrix3::from_row_slice(&ret)
}

pub(crate) fn get_primaries_xy(primaries: ColorPrimaries) -> Result<[[f32; 2]; 3]> {
    Ok(match primaries {
        ColorPrimaries::BT470M => [[0.670, 0.330], [0.210, 0.710], [0.140, 0.080]],
        ColorPrimaries::BT470BG => [[0.640, 0.330], [0.290, 0.600], [0.150, 0.060]],
        ColorPrimaries::ST170M | ColorPrimaries::ST240M => {
            [[0.630, 0.340], [0.310, 0.595], [0.155, 0.070]]
        }
        ColorPrimaries::BT709 => [[0.640, 0.330], [0.300, 0.600], [0.150, 0.060]],
        ColorPrimaries::Film => [[0.681, 0.319], [0.243, 0.692], [0.145, 0.049]],
        ColorPrimaries::BT2020 => [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
        ColorPrimaries::P3DCI | ColorPrimaries::P3Display => {
            [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]]
        }
        ColorPrimaries::Tech3213 => [[0.630, 0.340], [0.295, 0.605], [0.155, 0.077]],
        ColorPrimaries::Reserved0 | ColorPrimaries::Reserved | ColorPrimaries::ST428 => {
            bail!("Cannot convert YUV<->RGB using these primaries")
        }
        // SAFETY: We guess any unspecified data when beginning conversion
        ColorPrimaries::Unspecified => unsafe { debug_unreachable!() },
    })
}

pub(crate) fn get_white_point(primaries: ColorPrimaries) -> [f32; 3] {
    // White points in XY.
    const ILLUMINANT_C: [f32; 2] = [0.31, 0.316];
    const ILLUMINANT_DCI: [f32; 2] = [0.314, 0.351];
    const ILLUMINANT_D65: [f32; 2] = [0.3127, 0.3290];
    const ILLUMINANT_E: [f32; 2] = [1.0 / 3.0, 1.0 / 3.0];

    match primaries {
        ColorPrimaries::BT470M | ColorPrimaries::Film => {
            xy_to_xyz(ILLUMINANT_C[0], ILLUMINANT_C[1])
        }
        ColorPrimaries::ST428 => xy_to_xyz(ILLUMINANT_E[0], ILLUMINANT_E[1]),
        ColorPrimaries::P3DCI => xy_to_xyz(ILLUMINANT_DCI[0], ILLUMINANT_DCI[1]),
        _ => xy_to_xyz(ILLUMINANT_D65[0], ILLUMINANT_D65[1]),
    }
}

fn xy_to_xyz(x: f32, y: f32) -> [f32; 3] {
    [x / y, 1.0, (1.0 - x - y) / y]
}
