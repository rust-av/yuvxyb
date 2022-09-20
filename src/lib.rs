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

mod pixel;

use std::mem::size_of;

use anyhow::{bail, Result};
pub use av_data::pixel::{ColorPrimaries, MatrixCoefficients, TransferCharacteristic};
pub use pixel::*;

#[derive(Clone)]
pub struct Xyb<T: Pixel> {
    data: Vec<[T; 3]>,
    width: u32,
    height: u32,
}

impl<T: Pixel> Xyb<T> {
    /// # Errors
    /// - If data length does not match `width * height`
    pub fn new(data: Vec<[T; 3]>, width: u32, height: u32) -> Result<Self> {
        if data.len() != (width * height) as usize {
            bail!("Data length does not match specified dimensions");
        }

        Ok(Self {
            data,
            width,
            height,
        })
    }

    #[must_use]
    pub fn data(&self) -> &[[T; 3]] {
        &self.data
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }
}

#[derive(Clone)]
pub struct Yuv<T: Pixel> {
    data: [Vec<T>; 3],
    width: u32,
    height: u32,
    config: YuvConfig,
}

#[derive(Clone, Copy)]
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
    /// - If `data` contains values which are not valid for the specified bit
    ///   depth (note: out-of-range values for limited range are allowed)
    pub fn new(data: [Vec<T>; 3], width: u32, height: u32, config: YuvConfig) -> Result<Self> {
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
        if data[0].len() != (width * height) as usize {
            bail!("Luma plane length does not match specified dimensions");
        }
        let chroma_len = (width * height) as usize >> (config.subsampling_x + config.subsampling_y);
        if data[1].len() != chroma_len {
            bail!("Cb plane length does not match specified dimensions");
        }
        if data[2].len() != chroma_len {
            bail!("Cr plane length does not match specified dimensions");
        }
        if size_of::<T>() == 2 && config.bit_depth < 16 {
            let max_value = 255u16 << (config.bit_depth - 8);
            if data.iter().any(|plane| {
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
            width,
            height,
            config,
        })
    }

    #[must_use]
    pub const fn data(&self) -> &[Vec<T>; 3] {
        &self.data
    }

    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    #[must_use]
    pub const fn config(&self) -> YuvConfig {
        self.config
    }
}

impl From<Yuv<u8>> for Yuv<f32> {
    fn from(other: Yuv<u8>) -> Self {
        if other.config.full_range {
            let data = [
                other.data[0]
                    .iter()
                    .map(|pix| f32::from(*pix) / 255.0)
                    .collect(),
                other.data[1]
                    .iter()
                    .map(|pix| f32::from(*pix) / 255.0)
                    .collect(),
                other.data[2]
                    .iter()
                    .map(|pix| f32::from(*pix) / 255.0)
                    .collect(),
            ];
            Yuv {
                data,
                width: other.width,
                height: other.height,
                config: other.config,
            }
        } else {
            todo!()
        }
    }
}

impl From<Yuv<u16>> for Yuv<f32> {
    fn from(other: Yuv<u16>) -> Self {
        if other.config.full_range {
            let max_val = f32::from(255u16 << (other.config.bit_depth - 8));
            let data = [
                other.data[0]
                    .iter()
                    .map(|pix| f32::from(*pix) / max_val)
                    .collect(),
                other.data[1]
                    .iter()
                    .map(|pix| f32::from(*pix) / max_val)
                    .collect(),
                other.data[2]
                    .iter()
                    .map(|pix| f32::from(*pix) / max_val)
                    .collect(),
            ];
            Yuv {
                data,
                width: other.width,
                height: other.height,
                config: other.config,
            }
        } else {
            todo!()
        }
    }
}

impl From<Yuv<f32>> for Yuv<u8> {
    fn from(other: Yuv<f32>) -> Self {
        todo!()
    }
}

impl From<Yuv<f32>> for Yuv<u16> {
    fn from(other: Yuv<f32>) -> Self {
        todo!()
    }
}

impl From<Yuv<u8>> for Xyb<f32> {
    fn from(other: Yuv<u8>) -> Self {
        Xyb::<f32>::from(Yuv::<f32>::from(other))
    }
}

impl From<Yuv<u16>> for Xyb<f32> {
    fn from(other: Yuv<u16>) -> Self {
        Xyb::<f32>::from(Yuv::<f32>::from(other))
    }
}

impl From<Yuv<f32>> for Xyb<f32> {
    fn from(other: Yuv<f32>) -> Self {
        todo!()
    }
}

impl TryFrom<(Xyb<f32>, YuvConfig)> for Yuv<u8> {
    type Error = anyhow::Error;

    /// # Errors
    /// - If subsampling is requested for a dimension that is not a multiple of
    ///   2
    fn try_from(other: (Xyb<f32>, YuvConfig)) -> Result<Self> {
        Ok(Yuv::<u8>::from(Yuv::<f32>::try_from(other)?))
    }
}

impl TryFrom<(Xyb<f32>, YuvConfig)> for Yuv<u16> {
    type Error = anyhow::Error;

    /// # Errors
    /// - If subsampling is requested for a dimension that is not a multiple of
    ///   2
    fn try_from(other: (Xyb<f32>, YuvConfig)) -> Result<Self> {
        Ok(Yuv::<u16>::from(Yuv::<f32>::try_from(other)?))
    }
}

impl TryFrom<(Xyb<f32>, YuvConfig)> for Yuv<f32> {
    type Error = anyhow::Error;

    /// # Errors
    /// - If subsampling is requested for a dimension that is not a multiple of
    ///   2
    fn try_from(other: (Xyb<f32>, YuvConfig)) -> Result<Self> {
        todo!()
    }
}
