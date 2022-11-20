use anyhow::{bail, Result};
use av_data::pixel::ColorPrimaries;
use v_frame::prelude::Pixel;

use crate::{Rgb, Yuv, yuv_rgb::{transform_primaries, TransferFunction}, rgb_xyb::xyb_to_linear_rgb, Xyb};

#[derive(Debug, Clone)]
pub struct LinearRgb {
    pub(crate) data: Vec<[f32; 3]>,
    pub(crate) width: usize,
    pub(crate) height: usize,
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
        Self::try_from(rgb)
    }
}

impl TryFrom<Rgb> for LinearRgb {
    type Error = anyhow::Error;

    fn try_from(rgb: Rgb) -> Result<Self> {
        let data = rgb.transfer.to_linear(rgb.data)?;
        let data = transform_primaries(data, rgb.primaries, ColorPrimaries::BT709)?;

        Ok(Self {
            data,
            width: rgb.width,
            height: rgb.height,
        })
    }
}

impl From<Xyb> for LinearRgb {
    fn from(xyb: Xyb) -> Self {
        let data = xyb_to_linear_rgb(xyb.data);

        Self {
            data,
            width: xyb.width,
            height: xyb.height,
        }
    }
}
