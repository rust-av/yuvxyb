use std::mem::size_of;

use av_data::pixel::{ColorPrimaries, MatrixCoefficients, TransferCharacteristic};
use thiserror::Error;
use v_frame::{frame::Frame, plane::Plane, prelude::Pixel};

use crate::{yuv_rgb::rgb_to_yuv, ConversionError, LinearRgb, Rgb, Xyb};

/// Contains a YCbCr image in a color space defined by [`YuvConfig`].
///
/// YCbCr (often called YUV) representations of images are transformed from a "true" RGB image.
/// These transformations are dependent on the original RGB color space and use one luminance (Y)
/// as well as two chrominance (U, V) components to represent a color.
///
/// The data in this structure supports chroma subsampling (e.g. 4:2:0), multiple bit depths,
/// and limited range support.
#[derive(Debug, Clone)]
pub struct Yuv<T: Pixel> {
    data: Frame<T>,
    config: YuvConfig,
}

/// Contains the configuration data for a YCbCr image.
///
/// This includes color space information, bit depth, chroma subsampling and whether the data
/// is full or limited range.
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

/// Error type for when creating a [`Yuv`] struct goes wrong.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum YuvError {
    /// The configured subsampling does not match
    /// the actual subsampling in the frame data.
    #[error("Configured subsampling does not match subsampling of frame data.")]
    SubsamplingMismatch,

    /// The width of the luminance plane is not
    /// compatible with the configured subsampling.
    ///
    /// For example, for 4:2:0 chroma subsampling, the width
    /// of the luminance plane must be divisible by 2.
    #[error("The frame width does not support the configured subsampling.")]
    InvalidLumaWidth,

    /// The height of the luminance plane is not
    /// compatible with the configured subsampling.
    ///
    /// For example, for 4:2:0 chroma subsampling, the height
    /// of the luminance plane must be divisible by 2.
    #[error("The frame height does not support the configured subsampling.")]
    InvalidLumaHeight,

    /// The supplied data contains values which are outside
    /// the valid range for the configured bit depth.
    ///
    /// This error can not occur for bit depths of 8 and 16,
    /// as the valid range of values matches the available
    /// range in the underlying data type exactly.
    #[error("Data contains values which are not valid for the configured bit depth.")]
    InvalidData,
}

impl<T: Pixel> Yuv<T> {
    /// Create a new [`Yuv`] with the given data and configuration.
    ///
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
    // Clippy complains about T::to_u16 maybe panicking, but it can be assumed
    // to never panic because the Pixel trait is only implemented by u8 and
    // u16, both of which will successfully return a u16 from to_u16.
    #[allow(clippy::missing_panics_doc)]
    pub fn new(data: Frame<T>, config: YuvConfig) -> Result<Self, YuvError> {
        if config.subsampling_x != data.planes[1].cfg.xdec as u8
            || config.subsampling_x != data.planes[2].cfg.xdec as u8
            || config.subsampling_y != data.planes[1].cfg.ydec as u8
            || config.subsampling_y != data.planes[2].cfg.ydec as u8
        {
            return Err(YuvError::SubsamplingMismatch);
        }

        let width = data.planes[0].cfg.width;
        let height = data.planes[0].cfg.height;
        if width % (1 << config.subsampling_x) != 0 {
            return Err(YuvError::InvalidLumaWidth);
        }
        if height % (1 << config.subsampling_y) != 0 {
            return Err(YuvError::InvalidLumaHeight);
        }
        if size_of::<T>() == 2 && config.bit_depth < 16 {
            let max_value = u16::MAX >> (16 - config.bit_depth);
            if data.planes.iter().any(|plane| {
                plane
                    .iter()
                    .any(|pix| pix.to_u16().expect("This is a u16") > max_value)
            }) {
                return Err(YuvError::InvalidData);
            }
        }

        Ok(Self {
            data,
            config: config.fix_unspecified_data(width, height),
        })
    }

    #[must_use]
    #[inline]
    pub const fn data(&self) -> &[Plane<T>] {
        &self.data.planes
    }

    #[must_use]
    #[inline]
    pub const fn width(&self) -> usize {
        self.data.planes[0].cfg.width
    }

    #[must_use]
    #[inline]
    pub const fn height(&self) -> usize {
        self.data.planes[0].cfg.height
    }

    #[must_use]
    #[inline]
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

impl<T: Pixel> TryFrom<(Xyb, YuvConfig)> for Yuv<T> {
    type Error = ConversionError;

    /// # Errors
    /// - If the `YuvConfig` would produce an invalid image
    fn try_from(other: (Xyb, YuvConfig)) -> Result<Self, Self::Error> {
        let lrgb = LinearRgb::from(other.0);
        Self::try_from((lrgb, other.1))
    }
}

impl<T: Pixel> TryFrom<(Rgb, YuvConfig)> for Yuv<T> {
    type Error = ConversionError;

    fn try_from(other: (Rgb, YuvConfig)) -> Result<Self, Self::Error> {
        Self::try_from((&other.0, other.1))
    }
}

impl<T: Pixel> TryFrom<(LinearRgb, YuvConfig)> for Yuv<T> {
    type Error = ConversionError;

    fn try_from(other: (LinearRgb, YuvConfig)) -> Result<Self, Self::Error> {
        let config = other.1;
        let rgb = Rgb::try_from((
            other.0,
            config.transfer_characteristics,
            config.color_primaries,
        ))?;
        Self::try_from((&rgb, config))
    }
}

impl<T: Pixel> TryFrom<(&Rgb, YuvConfig)> for Yuv<T> {
    type Error = ConversionError;

    fn try_from(other: (&Rgb, YuvConfig)) -> Result<Self, Self::Error> {
        let rgb = other.0;
        let config = other.1;
        rgb_to_yuv(rgb.data(), rgb.width(), rgb.height(), config)
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
