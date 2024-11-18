use crate::matrix::{ColVector, Matrix, RowVector};
use av_data::pixel::{ColorPrimaries, MatrixCoefficients};

use super::{ycbcr_to_ypbpr, ypbpr_to_ycbcr};
use crate::{ConversionError, Pixel, Yuv, YuvConfig};

pub fn get_yuv_to_rgb_matrix(config: YuvConfig) -> Result<Matrix, ConversionError> {
    get_rgb_to_yuv_matrix(config).map(|m| m.invert())
}

pub fn get_rgb_to_yuv_matrix(config: YuvConfig) -> Result<Matrix, ConversionError> {
    match config.matrix_coefficients {
        MatrixCoefficients::Identity
        | MatrixCoefficients::BT2020ConstantLuminance
        | MatrixCoefficients::ChromaticityDerivedConstantLuminance
        | MatrixCoefficients::ST2085
        | MatrixCoefficients::ICtCp => ncl_rgb_to_yuv_matrix_from_primaries(config.color_primaries),
        MatrixCoefficients::BT709
        | MatrixCoefficients::BT470M
        | MatrixCoefficients::BT470BG
        | MatrixCoefficients::ST170M
        | MatrixCoefficients::ST240M
        | MatrixCoefficients::YCgCo
        | MatrixCoefficients::ChromaticityDerivedNonConstantLuminance
        | MatrixCoefficients::BT2020NonConstantLuminance => {
            ncl_rgb_to_yuv_matrix(config.matrix_coefficients)
        }
        MatrixCoefficients::Reserved => Err(ConversionError::UnsupportedMatrixCoefficients),
        MatrixCoefficients::Unspecified => Err(ConversionError::UnspecifiedMatrixCoefficients),
    }
}

pub fn ncl_rgb_to_yuv_matrix_from_primaries(
    primaries: ColorPrimaries,
) -> Result<Matrix, ConversionError> {
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

pub fn ncl_rgb_to_yuv_matrix(matrix: MatrixCoefficients) -> Result<Matrix, ConversionError> {
    Ok(match matrix {
        MatrixCoefficients::YCgCo => Matrix::new(
            RowVector::new(0.25, 0.5, 0.25),
            RowVector::new(-0.25, 0.5, -0.25),
            RowVector::new(0.5, 0.0, -0.5),
        ),
        MatrixCoefficients::ST2085 => Matrix::new(
            RowVector::new(1688.0, 2146.0, 262.0),
            RowVector::new(683.0, 2951.0, 462.0),
            RowVector::new(99.0, 309.0, 3688.0),
        )
        .scalar_div(4096.0),
        _ => {
            let (kr, kb) = get_yuv_constants(matrix)?;
            ncl_rgb_to_yuv_matrix_from_kr_kb(kr, kb)
        }
    })
}

pub fn get_yuv_constants_from_primaries(
    primaries: ColorPrimaries,
) -> Result<(f32, f32), ConversionError> {
    // ITU-T H.265 Annex E, Eq (E-22) to (E-27).
    let primaries_xy = get_primaries_xy(primaries)?;

    let r_xyz = RowVector::from(xy_to_xyz(primaries_xy[0][0], primaries_xy[0][1]));
    let g_xyz = RowVector::from(xy_to_xyz(primaries_xy[1][0], primaries_xy[1][1]));
    let b_xyz = RowVector::from(xy_to_xyz(primaries_xy[2][0], primaries_xy[2][1]));
    let white_xyz = RowVector::from(get_white_point(primaries));

    let x_rgb = RowVector::new(r_xyz.x(), g_xyz.x(), b_xyz.x());
    let y_rgb = RowVector::new(r_xyz.y(), g_xyz.y(), b_xyz.y());
    let z_rgb = RowVector::new(r_xyz.z(), g_xyz.z(), b_xyz.z());

    let denom = x_rgb.dot(&y_rgb.cross(&z_rgb));
    let kr = white_xyz.dot(&g_xyz.cross(&b_xyz)) / denom;
    let kb = white_xyz.dot(&r_xyz.cross(&g_xyz)) / denom;

    Ok((kr, kb))
}

pub const fn get_yuv_constants(matrix: MatrixCoefficients) -> Result<(f32, f32), ConversionError> {
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
        | MatrixCoefficients::ICtCp => return Err(ConversionError::UnsupportedMatrixCoefficients),
        MatrixCoefficients::Unspecified => {
            return Err(ConversionError::UnspecifiedMatrixCoefficients)
        }
    })
}

pub fn ncl_rgb_to_yuv_matrix_from_kr_kb(kr: f32, kb: f32) -> Matrix {
    let kg = 1.0 - kr - kb;
    let uscale = 1.0 / 2.0f32.mul_add(-kb, 2.0);
    let vscale = 1.0 / 2.0f32.mul_add(-kr, 2.0);

    Matrix::new(
        RowVector::new(kr, kg, kb),
        RowVector::new(-kr * uscale, -kg * uscale, (1.0 - kb) * uscale),
        RowVector::new((1.0 - kr) * vscale, -kg * vscale, -kb * vscale),
    )
}

pub const fn get_primaries_xy(primaries: ColorPrimaries) -> Result<[[f32; 2]; 3], ConversionError> {
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
            return Err(ConversionError::UnsupportedColorPrimaries)
        }
        // SAFETY: We guess any unspecified data when beginning conversion
        ColorPrimaries::Unspecified => return Err(ConversionError::UnspecifiedColorPrimaries),
    })
}

pub fn get_white_point(primaries: ColorPrimaries) -> [f32; 3] {
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

/// Converts 8..=16-bit YUV data to 32-bit floating point gamma-corrected RGB
/// in a range of 0.0..=1.0;
pub fn yuv_to_rgb<T: Pixel>(input: &Yuv<T>) -> Result<Vec<[f32; 3]>, ConversionError> {
    let transform = get_yuv_to_rgb_matrix(input.config())?;
    let mut data = ycbcr_to_ypbpr(input);

    for pix in &mut data {
        *pix = transform.mul_arr(*pix);
    }

    Ok(data)
}

/// Converts 32-bit floating point gamma-corrected RGB in a range of 0.0..=1.0
/// to 8..=16-bit YUV.
///
/// # Errors
/// - If the `YuvConfig` would produce an invalid image
pub fn rgb_to_yuv<T: Pixel>(
    input: &[[f32; 3]],
    width: usize,
    height: usize,
    config: YuvConfig,
) -> Result<Yuv<T>, ConversionError> {
    let transform = get_rgb_to_yuv_matrix(config)?;
    let yuv: Vec<_> = input.iter().map(|pix| transform.mul_arr(*pix)).collect();
    Ok(ypbpr_to_ycbcr(&yuv, width, height, config))
}

pub fn transform_primaries(
    mut input: Vec<[f32; 3]>,
    in_primaries: ColorPrimaries,
    out_primaries: ColorPrimaries,
) -> Result<Vec<[f32; 3]>, ConversionError> {
    if in_primaries == out_primaries {
        return Ok(input);
    }

    let transform = gamut_xyz_to_rgb_matrix(out_primaries)?
        .mul_mat(white_point_adaptation_matrix(in_primaries, out_primaries))
        .mul_mat(gamut_rgb_to_xyz_matrix(in_primaries)?);

    for pix in &mut input {
        *pix = transform.mul_arr(*pix);
    }

    Ok(input)
}

fn gamut_rgb_to_xyz_matrix(primaries: ColorPrimaries) -> Result<Matrix, ConversionError> {
    if primaries == ColorPrimaries::ST428 {
        return Ok(Matrix::identity());
    }

    let xyz_matrix = get_primaries_xyz(primaries)?;
    let white_xyz = ColVector::from(get_white_point(primaries));

    let s = xyz_matrix.invert().mul_vec(&white_xyz).transpose();
    Ok(Matrix::new(
        xyz_matrix.r1().component_mul(&s),
        xyz_matrix.r2().component_mul(&s),
        xyz_matrix.r3().component_mul(&s),
    ))
}

fn gamut_xyz_to_rgb_matrix(primaries: ColorPrimaries) -> Result<Matrix, ConversionError> {
    if primaries == ColorPrimaries::ST428 {
        return Ok(Matrix::identity());
    }

    gamut_rgb_to_xyz_matrix(primaries).map(|m| m.invert())
}

fn get_primaries_xyz(primaries: ColorPrimaries) -> Result<Matrix, ConversionError> {
    // Columns: R G B
    // Rows: X Y Z

    get_primaries_xy(primaries)
        .map(|[r, g, b]| {
            Matrix::new(
                RowVector::from(xy_to_xyz(r[0], r[1])),
                RowVector::from(xy_to_xyz(g[0], g[1])),
                RowVector::from(xy_to_xyz(b[0], b[1])),
            )
        })
        .map(Matrix::transpose)
}

fn white_point_adaptation_matrix(
    in_primaries: ColorPrimaries,
    out_primaries: ColorPrimaries,
) -> Matrix {
    let bradford = Matrix::new(
        RowVector::new(0.8951, 0.2664, -0.1614),
        RowVector::new(-0.7502, 1.7135, 0.0367),
        RowVector::new(0.0389, -0.0685, 1.0296),
    );

    let white_in = ColVector::from(get_white_point(in_primaries));
    let white_out = ColVector::from(get_white_point(out_primaries));

    if white_in == white_out {
        return Matrix::identity();
    }

    let rgb_in = bradford.mul_vec(&white_in);
    let rgb_out = bradford.mul_vec(&white_out);

    let m = Matrix::new(
        RowVector::new(rgb_out.r() / rgb_in.r(), 0.0, 0.0),
        RowVector::new(0.0, rgb_out.g() / rgb_in.g(), 0.0),
        RowVector::new(0.0, 0.0, rgb_out.b() / rgb_in.b()),
    );

    bradford.invert().mul_mat(m).mul_mat(bradford)
}
