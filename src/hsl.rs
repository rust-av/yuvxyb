use anyhow::{bail, Result};

use crate::LinearRgb;

/// HSL Color Space: Hue, Saturation, Lightness.
///
/// The Y channel in YUV only has a loose correlation
/// with human-perceived brightness.
/// The L channel in this space is the closest
/// to a pure measure of human-perceived brightness.
///
/// An L value of 0.0 is always black, and an L value
/// of 1.0 is always white. Values in between gradiate
/// as one would expect. Therefore, this is extremely
/// useful for applications of measuring perceptual
/// brightness.
#[derive(Debug, Clone)]
pub struct Hsl {
    /// H is a value between 0 and 360 (degrees).
    /// S and L are values betwen 0.0 and 1.0.
    data: Vec<[f32; 3]>,
    width: usize,
    height: usize,
}

impl Hsl {
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

impl From<LinearRgb> for Hsl {
    fn from(lrgb: LinearRgb) -> Self {
        let width = lrgb.width();
        let height = lrgb.height();
        let mut data = lrgb.into_data();
        for pix in &mut data {
            *pix = lrgb_to_hsl(*pix);
        }

        Self {
            data,
            width,
            height,
        }
    }
}

#[inline(always)]
#[allow(clippy::many_single_char_names)]
fn lrgb_to_hsl(rgb: [f32; 3]) -> [f32; 3] {
    let x_max = rgb[0].max(rgb[1]).max(rgb[2]);
    let x_min = rgb[0].min(rgb[1]).min(rgb[2]);
    let v = x_max;
    let c = x_max - x_min;
    let l = (x_max + x_min) / 2.0;
    let h = if c.abs() < f32::EPSILON {
        0.0
    } else if (v - rgb[0]).abs() < f32::EPSILON {
        60.0 * ((rgb[1] - rgb[2]) / c)
    } else if (v - rgb[1]).abs() < f32::EPSILON {
        60.0 * (2.0 + (rgb[2] - rgb[0]) / c)
    } else {
        60.0 * (4.0 + (rgb[0] - rgb[1]) / c)
    };
    let s = if l.abs() < f32::EPSILON || (l - 1.0).abs() < f32::EPSILON {
        0.0
    } else {
        (2.0 * (v - l)) / (1.0 - (2.0 * l - 1.0).abs())
    };
    [h, s, l]
}
