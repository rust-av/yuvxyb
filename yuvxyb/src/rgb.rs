use std::num::NonZeroUsize;

use av_data::pixel::{ColorPrimaries, TransferCharacteristic};
use v_frame::pixel::Pixel;

use crate::{
    ConversionError, CreationError, LinearRgb, Xyb, Yuv,
    yuv_rgb::{TransferFunction, transform_primaries, yuv_to_rgb},
};

/// Contains an RGB image.
///
/// The image is stored as pixels made of three 32-bit floating-point RGB components which are
/// converted from/to linear color space by the specified transfer functions and color primaries.
#[derive(Debug, Clone)]
pub struct Rgb {
    data: Vec<[f32; 3]>,
    width: NonZeroUsize,
    height: NonZeroUsize,
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
        let Some(width) = NonZeroUsize::new(width) else {
            return Err(CreationError::ZeroResolution);
        };

        let Some(height) = NonZeroUsize::new(height) else {
            return Err(CreationError::ZeroResolution);
        };

        if data.len() != width.saturating_mul(height).get() {
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

    /// Guaranteed to be non-zero.
    #[must_use]
    #[inline]
    pub const fn width(&self) -> usize {
        self.width.get()
    }

    /// Guaranteed to be non-zero.
    #[must_use]
    #[inline]
    pub const fn height(&self) -> usize {
        self.height.get()
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
            width: NonZeroUsize::new(yuv.width()).expect("is non-zero1"),
            height: NonZeroUsize::new(yuv.height()).expect("is non-zero2"),
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

        let width = NonZeroUsize::new(lrgb.width()).expect("is non-zero");
        let height = NonZeroUsize::new(lrgb.height()).expect("is non-zero");
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
