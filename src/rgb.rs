use anyhow::{Result, bail};
use av_data::pixel::{TransferCharacteristic, ColorPrimaries};
use v_frame::prelude::Pixel;

use crate::{Yuv, LinearRgb, Xyb, yuv_rgb::{yuv_to_rgb, transform_primaries, TransferFunction}};

#[derive(Debug, Clone)]
pub struct Rgb {
    pub(crate) data: Vec<[f32; 3]>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) transfer: TransferCharacteristic,
    pub(crate) primaries: ColorPrimaries,
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

impl<T: Pixel> TryFrom<Yuv<T>> for Rgb {
    type Error = anyhow::Error;

    fn try_from(yuv: Yuv<T>) -> Result<Self> {
        Rgb::try_from(&yuv)
    }
}

impl<T: Pixel> TryFrom<&Yuv<T>> for Rgb {
    type Error = anyhow::Error;

    fn try_from(yuv: &Yuv<T>) -> Result<Self> {
        let data = yuv_to_rgb(yuv)?;

        Ok(Self {
            data,
            width: yuv.width(),
            height: yuv.height(),
            transfer: yuv.config.transfer_characteristics,
            primaries: yuv.config.color_primaries,
        })
    }
}

// From XYB
impl TryFrom<(Xyb, TransferCharacteristic, ColorPrimaries)> for Rgb {
    type Error = anyhow::Error;

    fn try_from(other: (Xyb, TransferCharacteristic, ColorPrimaries)) -> Result<Self> {
        let lrgb = LinearRgb::from(other.0);
        Self::try_from((lrgb, other.1, other.2))
    }
}

impl TryFrom<(LinearRgb, TransferCharacteristic, ColorPrimaries)> for Rgb {
    type Error = anyhow::Error;

    fn try_from(other: (LinearRgb, TransferCharacteristic, ColorPrimaries)) -> Result<Self> {
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

        let data = transform_primaries(lrgb.data, ColorPrimaries::BT709, primaries)?;
        let data = transfer.to_gamma(data)?;

        Ok(Self {
            data,
            width: lrgb.width,
            height: lrgb.height,
            transfer,
            primaries,
        })
    }
}
