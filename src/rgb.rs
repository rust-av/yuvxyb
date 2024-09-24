use av_data::pixel::{ColorPrimaries, TransferCharacteristic};
use v_frame::prelude::Pixel;

use crate::{
    yuv_rgb::{transform_primaries, yuv_to_rgb, TransferFunction},
    ConversionError, CreationError, LinearRgb, Xyb, Yuv,
};

/// Contains an RGB image.
///
/// The image is stored as pixels made of three 32-bit floating-point RGB components which are
/// converted from/to linear color space by the specified transfer functions and color primaries.
#[derive(Debug, Clone)]
pub struct Rgb {
    data: Vec<[f32; 3]>,
    width: usize,
    height: usize,
    transfer: TransferCharacteristic,
    primaries: ColorPrimaries,
}

impl Rgb {
    /// Create a new [`Rgb`] with the given data and configuration.
    ///
    /// It is up to the caller to ensure that the transfer characteristics and
    /// color primaries are correct for the data.
    ///
    /// # Errors
    /// - If data length does not match `width * height`
    pub fn new(
        data: Vec<[f32; 3]>,
        width: usize,
        height: usize,
        mut transfer: TransferCharacteristic,
        mut primaries: ColorPrimaries,
    ) -> Result<Self, CreationError> {
        if data.len() != width * height {
            return Err(CreationError::ResolutionMismatch);
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
    #[inline]
    pub fn data(&self) -> &[[f32; 3]] {
        &self.data
    }

    #[must_use]
    #[inline]
    pub fn data_mut(&mut self) -> &mut [[f32; 3]] {
        &mut self.data
    }

    #[must_use]
    #[inline]
    pub fn into_data(self) -> Vec<[f32; 3]> {
        self.data
    }

    #[must_use]
    #[inline]
    pub const fn width(&self) -> usize {
        self.width
    }

    #[must_use]
    #[inline]
    pub const fn height(&self) -> usize {
        self.height
    }

    #[must_use]
    #[inline]
    pub const fn transfer(&self) -> TransferCharacteristic {
        self.transfer
    }

    #[must_use]
    #[inline]
    pub const fn primaries(&self) -> ColorPrimaries {
        self.primaries
    }
}

impl<T: Pixel> TryFrom<Yuv<T>> for Rgb {
    type Error = ConversionError;

    fn try_from(yuv: Yuv<T>) -> Result<Self, Self::Error> {
        Self::try_from(&yuv)
    }
}

impl<T: Pixel> TryFrom<&Yuv<T>> for Rgb {
    type Error = ConversionError;

    fn try_from(yuv: &Yuv<T>) -> Result<Self, Self::Error> {
        let data = yuv_to_rgb(yuv)?;

        Ok(Self {
            data,
            width: yuv.width(),
            height: yuv.height(),
            transfer: yuv.config().transfer_characteristics,
            primaries: yuv.config().color_primaries,
        })
    }
}

impl TryFrom<(Xyb, TransferCharacteristic, ColorPrimaries)> for Rgb {
    type Error = ConversionError;

    fn try_from(other: (Xyb, TransferCharacteristic, ColorPrimaries)) -> Result<Self, Self::Error> {
        let lrgb = LinearRgb::from(other.0);
        Self::try_from((lrgb, other.1, other.2))
    }
}

impl TryFrom<(LinearRgb, TransferCharacteristic, ColorPrimaries)> for Rgb {
    type Error = ConversionError;

    fn try_from(
        other: (LinearRgb, TransferCharacteristic, ColorPrimaries),
    ) -> Result<Self, Self::Error> {
        let lrgb = other.0;
        let (mut transfer, mut primaries) = (other.1, other.2);

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

        let width = lrgb.width();
        let height = lrgb.height();
        let data = transform_primaries(lrgb.into_data(), ColorPrimaries::BT709, primaries)?;
        let data = transfer.to_gamma(data)?;

        Ok(Self {
            data,
            width,
            height,
            transfer,
            primaries,
        })
    }
}
