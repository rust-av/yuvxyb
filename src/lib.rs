#![deny(clippy::all)]
#![warn(clippy::nursery)]
#![warn(clippy::pedantic)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::inconsistent_struct_constructor)]
#![allow(clippy::inline_always)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::similar_names)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::use_self)]
#![warn(clippy::clone_on_ref_ptr)]
#![warn(clippy::create_dir)]
#![warn(clippy::dbg_macro)]
#![warn(clippy::default_numeric_fallback)]
#![warn(clippy::exit)]
#![warn(clippy::filetype_is_file)]
#![warn(clippy::float_cmp_const)]
#![warn(clippy::if_then_some_else_none)]
#![warn(clippy::lossy_float_literal)]
#![warn(clippy::map_err_ignore)]
#![warn(clippy::mem_forget)]
#![warn(clippy::mod_module_files)]
#![warn(clippy::multiple_inherent_impl)]
#![warn(clippy::pattern_type_mismatch)]
#![warn(clippy::rc_buffer)]
#![warn(clippy::rc_mutex)]
#![warn(clippy::rest_pat_in_fully_bound_structs)]
#![warn(clippy::same_name_method)]
#![warn(clippy::str_to_string)]
#![warn(clippy::string_to_string)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![warn(clippy::unnecessary_self_imports)]
#![warn(clippy::unneeded_field_pattern)]
#![warn(clippy::use_debug)]
#![warn(clippy::verbose_file_reads)]

// This is pub and doc hidden so it can be run through cargo asm for
// optimization easier
mod fastmath;
#[doc(hidden)]
pub mod rgb_xyz;
#[doc(hidden)]
pub mod yuv_rgb;

#[doc(hidden)]
pub mod hsl;

use anyhow::{bail, Result};
pub use av_data::pixel::{ColorPrimaries, MatrixCoefficients, TransferCharacteristic};
pub use hsl::Hsl;
pub use num_traits::{FromPrimitive, ToPrimitive};
use rgb_xyz::{linear_rgb_to_xyb, xyb_to_linear_rgb};
use std::mem::size_of;
pub use v_frame::{
    frame::Frame,
    plane::Plane,
    prelude::{CastFromPrimitive, Pixel},
};
use yuv_rgb::{rgb_to_yuv, transform_primaries, yuv_to_rgb, TransferFunction};

#[derive(Debug, Clone)]
pub struct Xyb {
    data: Vec<[f32; 3]>,
    width: usize,
    height: usize,
}

impl Xyb {
    /// # Errors
    /// - If data length does not match `width * height`
    pub fn new(data: Vec<[f32; 3]>, width: usize, height: usize) -> Result<Self> {
        if data.len() != width * height {
            bail!("Data length does not match specified dimensions");
        }

        Ok(Self {
            data,
            width,
            height,
        })
    }

    #[must_use]
    #[inline(always)]
    pub fn data(&self) -> &[[f32; 3]] {
        &self.data
    }

    #[must_use]
    #[inline(always)]
    pub fn data_mut(&mut self) -> &mut [[f32; 3]] {
        &mut self.data
    }

    #[must_use]
    #[inline(always)]
    pub const fn width(&self) -> usize {
        self.width
    }

    #[must_use]
    #[inline(always)]
    pub const fn height(&self) -> usize {
        self.height
    }
}

#[derive(Debug, Clone)]
pub struct LinearRgb {
    data: Vec<[f32; 3]>,
    width: usize,
    height: usize,
}

impl LinearRgb {
    /// # Errors
    /// - If data length does not match `width * height`
    pub fn new(data: Vec<[f32; 3]>, width: usize, height: usize) -> Result<Self> {
        if data.len() != width * height {
            bail!("Data length does not match specified dimensions");
        }

        Ok(Self {
            data,
            width,
            height,
        })
    }

    #[must_use]
    #[inline(always)]
    pub fn data(&self) -> &[[f32; 3]] {
        &self.data
    }

    #[must_use]
    #[inline(always)]
    pub fn data_mut(&mut self) -> &mut [[f32; 3]] {
        &mut self.data
    }

    #[must_use]
    #[inline(always)]
    pub const fn width(&self) -> usize {
        self.width
    }

    #[must_use]
    #[inline(always)]
    pub const fn height(&self) -> usize {
        self.height
    }
}

#[derive(Debug, Clone)]
pub struct Rgb {
    data: Vec<[f32; 3]>,
    width: usize,
    height: usize,
    transfer: TransferCharacteristic,
    primaries: ColorPrimaries,
}

impl Rgb {
    /// # Errors
    /// - If data length does not match `width * height`
    pub fn new(
        data: Vec<[f32; 3]>,
        width: usize,
        height: usize,
        mut transfer: TransferCharacteristic,
        mut primaries: ColorPrimaries,
    ) -> Result<Self> {
        if data.len() != width * height {
            bail!("Data length does not match specified dimensions");
        }

        if transfer == TransferCharacteristic::Unspecified {
            transfer = TransferCharacteristic::SRGB;
            log::warn!(
                "Transfer characteristics not specified. Guessing {}",
                transfer
            );
        }

        if primaries == ColorPrimaries::Unspecified {
            primaries = ColorPrimaries::BT709;
            log::warn!("Color primaries not specified. Guessing {}", primaries);
        }

        Ok(Self {
            data,
            width,
            height,
            transfer,
            primaries,
        })
    }

    #[must_use]
    #[inline(always)]
    pub fn data(&self) -> &[[f32; 3]] {
        &self.data
    }

    #[must_use]
    #[inline(always)]
    pub fn data_mut(&mut self) -> &mut [[f32; 3]] {
        &mut self.data
    }

    #[must_use]
    #[inline(always)]
    pub const fn width(&self) -> usize {
        self.width
    }

    #[must_use]
    #[inline(always)]
    pub const fn height(&self) -> usize {
        self.height
    }
}

#[derive(Debug, Clone)]
pub struct Yuv<T: Pixel> {
    data: Frame<T>,
    config: YuvConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YuvConfig {
    pub bit_depth: u8,
    pub subsampling_x: u8,
    pub subsampling_y: u8,
    pub full_range: bool,
    pub matrix_coefficients: MatrixCoefficients,
    pub transfer_characteristics: TransferCharacteristic,
    pub color_primaries: ColorPrimaries,
}

impl<T: Pixel> Yuv<T> {
    /// # Errors
    /// - If luma plane length does not match `width * height`
    /// - If chroma plane lengths do not match `(width * height) >>
    ///   (subsampling_x + subsampling_y)`
    /// - If chroma subsampling is enabled and dimensions are not a multiple of
    ///   2
    /// - If chroma sampling set in `config` does not match subsampling in the
    ///   frame data
    /// - If `data` contains values which are not valid for the specified bit
    ///   depth (note: out-of-range values for limited range are allowed)
    pub fn new(data: Frame<T>, config: YuvConfig) -> Result<Self> {
        if config.subsampling_x != data.planes[1].cfg.xdec as u8
            || config.subsampling_x != data.planes[2].cfg.xdec as u8
            || config.subsampling_y != data.planes[1].cfg.ydec as u8
            || config.subsampling_y != data.planes[2].cfg.ydec as u8
        {
            bail!("Configured subsampling does not match subsampling of Frame data");
        }

        let width = data.planes[0].cfg.width;
        let height = data.planes[0].cfg.height;
        if width % (1 << config.subsampling_x) != 0 {
            bail!(
                "Width must be a multiple of {} to support this chroma subsampling",
                1u32 << config.subsampling_x
            );
        }
        if height % (1 << config.subsampling_y) != 0 {
            bail!(
                "Height must be a multiple of {} to support this chroma subsampling",
                1u32 << config.subsampling_y
            );
        }
        if size_of::<T>() == 2 && config.bit_depth < 16 {
            let max_value = u16::MAX >> (16 - config.bit_depth);
            if data.planes.iter().any(|plane| {
                plane
                    .iter()
                    .any(|pix| pix.to_u16().expect("This is a u16") > max_value)
            }) {
                bail!(
                    "Data contains values which are not valid for a bit depth of {}",
                    config.bit_depth
                );
            }
        }

        Ok(Self {
            data,
            config: config.fix_unspecified_data(width, height),
        })
    }

    #[must_use]
    #[inline(always)]
    pub const fn data(&self) -> &[Plane<T>] {
        &self.data.planes
    }

    #[must_use]
    #[inline(always)]
    pub const fn width(&self) -> usize {
        self.data.planes[0].cfg.width
    }

    #[must_use]
    #[inline(always)]
    pub const fn height(&self) -> usize {
        self.data.planes[0].cfg.height
    }

    #[must_use]
    #[inline(always)]
    pub const fn config(&self) -> YuvConfig {
        self.config
    }
}

impl YuvConfig {
    pub(crate) fn fix_unspecified_data(mut self, width: usize, height: usize) -> Self {
        if self.matrix_coefficients == MatrixCoefficients::Unspecified {
            self.matrix_coefficients = guess_matrix_coefficients(width, height);
            log::warn!(
                "Matrix coefficients not specified. Guessing {}",
                self.matrix_coefficients
            );
        }

        if self.color_primaries == ColorPrimaries::Unspecified {
            self.color_primaries = guess_color_primaries(self.matrix_coefficients, width, height);
            log::warn!(
                "Color primaries not specified. Guessing {}",
                self.color_primaries
            );
        }

        if self.transfer_characteristics == TransferCharacteristic::Unspecified {
            self.transfer_characteristics = TransferCharacteristic::BT1886;
            log::warn!(
                "Transfer characteristics not specified. Guessing {}",
                self.transfer_characteristics
            );
        }

        self
    }
}

// Heuristic taken from mpv
const fn guess_matrix_coefficients(width: usize, height: usize) -> MatrixCoefficients {
    if width >= 1280 || height > 576 {
        MatrixCoefficients::BT709
    } else if height == 576 {
        MatrixCoefficients::BT470BG
    } else {
        MatrixCoefficients::ST170M
    }
}

// Heuristic taken from mpv
fn guess_color_primaries(
    matrix: MatrixCoefficients,
    width: usize,
    height: usize,
) -> ColorPrimaries {
    if matrix == MatrixCoefficients::BT2020NonConstantLuminance
        || matrix == MatrixCoefficients::BT2020ConstantLuminance
    {
        ColorPrimaries::BT2020
    } else if matrix == MatrixCoefficients::BT709 || width >= 1280 || height > 576 {
        ColorPrimaries::BT709
    } else if height == 576 {
        ColorPrimaries::BT470BG
    } else if height == 480 || height == 488 {
        ColorPrimaries::ST170M
    } else {
        ColorPrimaries::BT709
    }
}

// To XYB
impl<T: Pixel> TryFrom<Yuv<T>> for Xyb {
    type Error = anyhow::Error;

    fn try_from(yuv: Yuv<T>) -> Result<Self> {
        Xyb::try_from(&yuv)
    }
}

impl<T: Pixel> TryFrom<&Yuv<T>> for Xyb {
    type Error = anyhow::Error;

    fn try_from(yuv: &Yuv<T>) -> Result<Self> {
        let rgb = Rgb::try_from(yuv)?;
        let lrgb = LinearRgb::try_from(rgb)?;
        Ok(Xyb::from(lrgb))
    }
}

impl TryFrom<Rgb> for Xyb {
    type Error = anyhow::Error;

    fn try_from(rgb: Rgb) -> Result<Self> {
        Xyb::try_from(&rgb)
    }
}

impl TryFrom<&Rgb> for Xyb {
    type Error = anyhow::Error;

    fn try_from(rgb: &Rgb) -> Result<Self> {
        let lrgb = LinearRgb::try_from(rgb)?;
        Ok(Xyb::from(lrgb))
    }
}

impl From<LinearRgb> for Xyb {
    fn from(lrgb: LinearRgb) -> Self {
        Xyb::from(&lrgb)
    }
}

impl From<&LinearRgb> for Xyb {
    fn from(lrgb: &LinearRgb) -> Self {
        let xyb = linear_rgb_to_xyb(lrgb.data());
        Xyb {
            data: xyb,
            width: lrgb.width(),
            height: lrgb.height(),
        }
    }
}

// To Linear RGB
impl<T: Pixel> TryFrom<Yuv<T>> for LinearRgb {
    type Error = anyhow::Error;

    fn try_from(yuv: Yuv<T>) -> Result<Self> {
        LinearRgb::try_from(&yuv)
    }
}

impl<T: Pixel> TryFrom<&Yuv<T>> for LinearRgb {
    type Error = anyhow::Error;

    fn try_from(yuv: &Yuv<T>) -> Result<Self> {
        let rgb = Rgb::try_from(yuv)?;
        LinearRgb::try_from(rgb)
    }
}

impl TryFrom<Rgb> for LinearRgb {
    type Error = anyhow::Error;

    fn try_from(rgb: Rgb) -> Result<Self> {
        LinearRgb::try_from(&rgb)
    }
}

impl TryFrom<&Rgb> for LinearRgb {
    type Error = anyhow::Error;

    fn try_from(rgb: &Rgb) -> Result<Self> {
        let lrgb = rgb.transfer.to_linear(&rgb.data)?;
        let lrgb = transform_primaries(&lrgb, rgb.primaries, ColorPrimaries::BT709)?;
        Ok(LinearRgb {
            data: lrgb,
            width: rgb.width(),
            height: rgb.height(),
        })
    }
}

// To RGB
impl<T: Pixel> TryFrom<Yuv<T>> for Rgb {
    type Error = anyhow::Error;

    fn try_from(yuv: Yuv<T>) -> Result<Self> {
        Rgb::try_from(&yuv)
    }
}

impl<T: Pixel> TryFrom<&Yuv<T>> for Rgb {
    type Error = anyhow::Error;

    fn try_from(yuv: &Yuv<T>) -> Result<Self> {
        let rgb = yuv_to_rgb(yuv)?;
        Ok(Rgb {
            data: rgb,
            width: yuv.width(),
            height: yuv.height(),
            transfer: yuv.config.transfer_characteristics,
            primaries: yuv.config.color_primaries,
        })
    }
}

// From XYB
impl<T: Pixel> TryFrom<(Xyb, YuvConfig)> for Yuv<T> {
    type Error = anyhow::Error;

    /// # Errors
    /// - If the `YuvConfig` would produce an invalid image
    fn try_from(other: (Xyb, YuvConfig)) -> Result<Self> {
        Yuv::<T>::try_from((&other.0, other.1))
    }
}

impl<T: Pixel> TryFrom<(&Xyb, YuvConfig)> for Yuv<T> {
    type Error = anyhow::Error;

    /// # Errors
    /// - If the `YuvConfig` would produce an invalid image
    fn try_from(other: (&Xyb, YuvConfig)) -> Result<Self> {
        let xyb = other.0;
        let lrgb = LinearRgb::from(xyb);
        Yuv::try_from((&lrgb, other.1))
    }
}

impl TryFrom<(Xyb, TransferCharacteristic, ColorPrimaries)> for Rgb {
    type Error = anyhow::Error;

    fn try_from(other: (Xyb, TransferCharacteristic, ColorPrimaries)) -> Result<Self> {
        Rgb::try_from((&other.0, other.1, other.2))
    }
}

impl TryFrom<(&Xyb, TransferCharacteristic, ColorPrimaries)> for Rgb {
    type Error = anyhow::Error;

    fn try_from(other: (&Xyb, TransferCharacteristic, ColorPrimaries)) -> Result<Self> {
        let xyb = other.0;
        let transfer = other.1;
        let primaries = other.2;
        let lrgb = LinearRgb::from(xyb);
        Rgb::try_from((&lrgb, transfer, primaries))
    }
}

impl From<Xyb> for LinearRgb {
    fn from(xyb: Xyb) -> Self {
        LinearRgb::from(&xyb)
    }
}

impl From<&Xyb> for LinearRgb {
    fn from(xyb: &Xyb) -> Self {
        let lrgb = xyb_to_linear_rgb(xyb.data());
        LinearRgb {
            data: lrgb,
            width: xyb.width(),
            height: xyb.height(),
        }
    }
}

// From Linear RGB
impl<T: Pixel> TryFrom<(LinearRgb, YuvConfig)> for Yuv<T> {
    type Error = anyhow::Error;

    fn try_from(other: (LinearRgb, YuvConfig)) -> Result<Self> {
        Yuv::<T>::try_from((&other.0, other.1))
    }
}

impl<T: Pixel> TryFrom<(&LinearRgb, YuvConfig)> for Yuv<T> {
    type Error = anyhow::Error;

    fn try_from(other: (&LinearRgb, YuvConfig)) -> Result<Self> {
        let rgb = Rgb::try_from((
            other.0,
            other.1.transfer_characteristics,
            other.1.color_primaries,
        ))?;
        Yuv::try_from((&rgb, other.1))
    }
}

impl TryFrom<(LinearRgb, TransferCharacteristic, ColorPrimaries)> for Rgb {
    type Error = anyhow::Error;

    fn try_from(other: (LinearRgb, TransferCharacteristic, ColorPrimaries)) -> Result<Self> {
        Rgb::try_from((&other.0, other.1, other.2))
    }
}

impl TryFrom<(&LinearRgb, TransferCharacteristic, ColorPrimaries)> for Rgb {
    type Error = anyhow::Error;

    fn try_from(other: (&LinearRgb, TransferCharacteristic, ColorPrimaries)) -> Result<Self> {
        let lrgb = other.0;
        let mut transfer = other.1;
        let mut primaries = other.2;

        if transfer == TransferCharacteristic::Unspecified {
            transfer = TransferCharacteristic::SRGB;
            log::warn!(
                "Transfer characteristics not specified. Guessing {}",
                transfer
            );
        }

        if primaries == ColorPrimaries::Unspecified {
            primaries = ColorPrimaries::BT709;
            log::warn!("Color primaries not specified. Guessing {}", primaries);
        }

        let rgb = transform_primaries(lrgb.data(), ColorPrimaries::BT709, primaries)?;
        let rgb = transfer.to_gamma(&rgb)?;
        Ok(Rgb {
            data: rgb,
            width: lrgb.width(),
            height: lrgb.height(),
            transfer,
            primaries,
        })
    }
}

// From RGB
impl<T: Pixel> TryFrom<(Rgb, YuvConfig)> for Yuv<T> {
    type Error = anyhow::Error;

    fn try_from(other: (Rgb, YuvConfig)) -> Result<Self> {
        Yuv::<T>::try_from((&other.0, other.1))
    }
}

impl<T: Pixel> TryFrom<(&Rgb, YuvConfig)> for Yuv<T> {
    type Error = anyhow::Error;

    fn try_from(other: (&Rgb, YuvConfig)) -> Result<Self> {
        let rgb = other.0;
        let config = other.1;
        rgb_to_yuv(rgb.data(), rgb.width(), rgb.height(), config)
    }
}

#[cfg(test)]
mod tests {
    use interpolate_name::interpolate_test;
    use rand::Rng;
    use v_frame::plane::Plane;

    use super::*;

    #[interpolate_test(bt601_420, MatrixCoefficients::ST170M, TransferCharacteristic::ST170M, ColorPrimaries::ST170M, (1, 1), false)]
    #[interpolate_test(bt601_422, MatrixCoefficients::ST170M, TransferCharacteristic::ST170M, ColorPrimaries::ST170M, (1, 0), false)]
    #[interpolate_test(bt601_444, MatrixCoefficients::ST170M, TransferCharacteristic::ST170M, ColorPrimaries::ST170M, (0, 0), false)]
    #[interpolate_test(bt601_444_full, MatrixCoefficients::ST170M, TransferCharacteristic::ST170M, ColorPrimaries::ST170M, (0, 0), true)]
    #[interpolate_test(bt709_420, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, ColorPrimaries::BT709, (1, 1), false)]
    #[interpolate_test(bt709_422, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, ColorPrimaries::BT709, (1, 0), false)]
    #[interpolate_test(bt709_444, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, ColorPrimaries::BT709, (0, 0), false)]
    #[interpolate_test(bt709_444_full, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, ColorPrimaries::BT709, (0, 0), true)]
    fn yuv_xyb_yuv_ident_8b(
        mc: MatrixCoefficients,
        tc: TransferCharacteristic,
        cp: ColorPrimaries,
        ss: (u8, u8),
        full_range: bool,
    ) {
        let y_dims = (320usize, 240usize);
        let uv_dims = (y_dims.0 >> ss.0, y_dims.1 >> ss.1);
        let mut data: Frame<u8> = Frame {
            planes: [
                Plane::new(y_dims.0, y_dims.1, 0, 0, 0, 0),
                Plane::new(
                    uv_dims.0,
                    uv_dims.1,
                    usize::from(ss.0),
                    usize::from(ss.1),
                    0,
                    0,
                ),
                Plane::new(
                    uv_dims.0,
                    uv_dims.1,
                    usize::from(ss.0),
                    usize::from(ss.1),
                    0,
                    0,
                ),
            ],
        };
        let mut rng = rand::thread_rng();
        for (i, plane) in data.planes.iter_mut().enumerate() {
            for val in plane.data_origin_mut().iter_mut() {
                *val = rng.gen_range(if full_range {
                    0..=255
                } else if i == 0 {
                    16..=235
                } else {
                    16..=240
                });
            }
        }
        let yuv = Yuv::new(
            data,
            YuvConfig {
                bit_depth: 8,
                subsampling_x: ss.0,
                subsampling_y: ss.1,
                full_range,
                matrix_coefficients: mc,
                transfer_characteristics: tc,
                color_primaries: cp,
            },
        )
        .unwrap();
        let xyb = Xyb::try_from(yuv.clone()).unwrap();
        let yuv2 = Yuv::<u8>::try_from((xyb, yuv.config())).unwrap();
        // assert_eq!(yuv.data(), yuv2.data());
        assert_eq!(yuv.width(), yuv2.width());
        assert_eq!(yuv.height(), yuv2.height());
        assert_eq!(yuv.config(), yuv2.config());
    }

    #[interpolate_test(bt709_420, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, ColorPrimaries::BT709, (1, 1), false)]
    #[interpolate_test(bt709_422, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, ColorPrimaries::BT709, (1, 0), false)]
    #[interpolate_test(bt709_444, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, ColorPrimaries::BT709, (0, 0), false)]
    #[interpolate_test(bt709_444_full, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, ColorPrimaries::BT709, (0, 0), true)]
    #[interpolate_test(
        bt2020_420,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::BT2020Ten,
        ColorPrimaries::BT2020,
        (1, 1),
        false
    )]
    #[interpolate_test(
        bt2020_422,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::BT2020Ten,
        ColorPrimaries::BT2020,
        (1, 0),
        false
    )]
    #[interpolate_test(
        bt2020_444,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::BT2020Ten,
        ColorPrimaries::BT2020,
        (0, 0),
        false
    )]
    #[interpolate_test(
        bt2020_444_full,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::BT2020Ten,
        ColorPrimaries::BT2020,
        (0, 0),
        true
    )]
    #[interpolate_test(
        pq_420,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::PerceptualQuantizer,
        ColorPrimaries::BT2020,
        (1, 1),
        false
    )]
    #[interpolate_test(
        pq_422,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::PerceptualQuantizer,
        ColorPrimaries::BT2020,
        (1, 0),
        false
    )]
    #[interpolate_test(
        pq_444,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::PerceptualQuantizer,
        ColorPrimaries::BT2020,
        (0, 0),
        false
    )]
    #[interpolate_test(
        pq_444_full,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::PerceptualQuantizer,
        ColorPrimaries::BT2020,
        (0, 0),
        true
    )]
    #[interpolate_test(
        hlg_420,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::HybridLogGamma,
        ColorPrimaries::BT2020,
        (1, 1),
        false
    )]
    #[interpolate_test(
        hlg_422,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::HybridLogGamma,
        ColorPrimaries::BT2020,
        (1, 0),
        false
    )]
    #[interpolate_test(
        hlg_444,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::HybridLogGamma,
        ColorPrimaries::BT2020,
        (0, 0),
        false
    )]
    #[interpolate_test(
        hlg_444_full,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::HybridLogGamma,
        ColorPrimaries::BT2020,
        (0, 0),
        true
    )]
    fn yuv_xyb_yuv_ident_10b(
        mc: MatrixCoefficients,
        tc: TransferCharacteristic,
        cp: ColorPrimaries,
        ss: (u8, u8),
        full_range: bool,
    ) {
        let y_dims = (320usize, 240usize);
        let uv_dims = (y_dims.0 >> ss.0, y_dims.1 >> ss.1);
        let mut data: Frame<u16> = Frame {
            planes: [
                Plane::new(y_dims.0, y_dims.1, 0, 0, 0, 0),
                Plane::new(
                    uv_dims.0,
                    uv_dims.1,
                    usize::from(ss.0),
                    usize::from(ss.1),
                    0,
                    0,
                ),
                Plane::new(
                    uv_dims.0,
                    uv_dims.1,
                    usize::from(ss.0),
                    usize::from(ss.1),
                    0,
                    0,
                ),
            ],
        };
        let mut rng = rand::thread_rng();
        for (i, plane) in data.planes.iter_mut().enumerate() {
            for val in plane.data_origin_mut().iter_mut() {
                *val = rng.gen_range(if full_range {
                    0..=1023
                } else if i == 0 {
                    64..=940
                } else {
                    64..=960
                });
            }
        }
        let yuv = Yuv::new(
            data,
            YuvConfig {
                bit_depth: 10,
                subsampling_x: ss.0,
                subsampling_y: ss.1,
                full_range,
                matrix_coefficients: mc,
                transfer_characteristics: tc,
                color_primaries: cp,
            },
        )
        .unwrap();
        let xyb = Xyb::try_from(yuv.clone()).unwrap();
        let yuv2 = Yuv::<u16>::try_from((xyb, yuv.config())).unwrap();
        // assert_eq!(yuv.data(), yuv2.data());
        assert_eq!(yuv.width(), yuv2.width());
        assert_eq!(yuv.height(), yuv2.height());
        assert_eq!(yuv.config(), yuv2.config());
    }
}
