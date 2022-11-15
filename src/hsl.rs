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
        let mut data = lrgb.data;
        for pix in &mut data {
            *pix = lrgb_to_hsl(*pix);
        }

        Hsl {
            data,
            width: lrgb.width,
            height: lrgb.height,
        }
    }
}

impl From<Hsl> for LinearRgb {
    fn from(hsl: Hsl) -> Self {
        let mut data = hsl.data;
        for pix in &mut data {
            *pix = hsl_to_lrgb(*pix);
        }

        LinearRgb {
            data,
            width: hsl.width,
            height: hsl.height,
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
