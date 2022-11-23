use anyhow::{bail, Result};
use av_data::pixel::ColorPrimaries;
use v_frame::prelude::Pixel;

use crate::{
    rgb_xyb::xyb_to_linear_rgb,
    yuv_rgb::{transform_primaries, TransferFunction},
    Hsl, Rgb, Xyb, Yuv,
};

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
    #[allow(clippy::missing_const_for_fn)]
    pub fn into_data(self) -> Vec<[f32; 3]> {
        self.data
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
        Self::try_from(&yuv)
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
        let width = rgb.width();
        let height = rgb.height();
        let transfer = rgb.transfer();
        let primaries = rgb.primaries();

        let data = transfer.to_linear(rgb.into_data())?;
        let data = transform_primaries(data, primaries, ColorPrimaries::BT709)?;

        Ok(Self {
            data,
            width,
            height,
        })
    }
}

impl From<Xyb> for LinearRgb {
    fn from(xyb: Xyb) -> Self {
        let width = xyb.width();
        let height = xyb.height();
        let data = xyb_to_linear_rgb(xyb.into_data());

        Self {
            data,
            width,
            height,
        }
    }
}

impl From<Hsl> for LinearRgb {
    fn from(hsl: Hsl) -> Self {
        let width = hsl.width();
        let height = hsl.height();
        let mut data = hsl.into_data();
        for pix in &mut data {
            *pix = hsl_to_lrgb(*pix);
        }

        Self {
            data,
            width,
            height,
        }
    }
}

#[inline(always)]
fn hsl_to_lrgb(hsl: [f32; 3]) -> [f32; 3] {
    let c = (1.0 - (2.0 * hsl[2] - 1.0).abs()) * hsl[1];
    let h_prime = hsl[0] / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
    let (r1, g1, b1) = if (0.0..1.0).contains(&h_prime) {
        (c, x, 0.0)
    } else if (1.0..2.0).contains(&h_prime) {
        (x, c, 0.0)
    } else if (2.0..3.0).contains(&h_prime) {
        (0.0, c, x)
    } else if (3.0..4.0).contains(&h_prime) {
        (0.0, x, c)
    } else if (4.0..5.0).contains(&h_prime) {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };
    let m = hsl[2] - c / 2.0;
    [r1 + m, g1 + m, b1 + m]
}
