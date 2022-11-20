use anyhow::{bail, Result};
use v_frame::prelude::Pixel;

use crate::{Yuv, Rgb, LinearRgb, rgb_xyb::linear_rgb_to_xyb};

#[derive(Debug, Clone)]
pub struct Xyb {
    pub(crate) data: Vec<[f32; 3]>,
    pub(crate) width: usize,
    pub(crate) height: usize,
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

impl<T: Pixel> TryFrom<Yuv<T>> for Xyb {
    type Error = anyhow::Error;

    fn try_from(yuv: Yuv<T>) -> Result<Self> {
        Self::try_from(&yuv)
    }
}

impl<T: Pixel> TryFrom<&Yuv<T>> for Xyb {
    type Error = anyhow::Error;

    fn try_from(yuv: &Yuv<T>) -> Result<Self> {
        let rgb = Rgb::try_from(yuv)?;
        Self::try_from(rgb)
    }
}

impl TryFrom<Rgb> for Xyb {
    type Error = anyhow::Error;

    fn try_from(rgb: Rgb) -> Result<Self> {
        let lrgb = LinearRgb::try_from(rgb)?;
        Ok(Self::from(lrgb))
    }
}

impl From<LinearRgb> for Xyb {
    fn from(lrgb: LinearRgb) -> Self {
        let data = linear_rgb_to_xyb(lrgb.data);

        Self {
            data,
            width: lrgb.width,
            height: lrgb.height,
        }
    }
}
