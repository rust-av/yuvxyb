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

// This is pub and doc hidden so it can be run through cargo asm for
// optimization easier
#[doc(hidden)]
pub mod math;
mod pixel;

use std::mem::size_of;

use anyhow::{bail, Result};
pub use av_data::pixel::{ColorPrimaries, MatrixCoefficients, TransferCharacteristic};
use math::{linear_rgb_to_xyb, linear_rgb_to_yuv, xyb_to_linear_rgb, yuv_to_linear_rgb};
pub use pixel::*;

#[derive(Debug, Clone)]
pub struct Xyb {
    data: Vec<[f32; 3]>,
    width: u32,
    height: u32,
}

impl Xyb {
    /// # Errors
    /// - If data length does not match `width * height`
    pub fn new(data: Vec<[f32; 3]>, width: u32, height: u32) -> Result<Self> {
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
    pub fn data(&self) -> &[[f32; 3]] {
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

#[derive(Debug, Clone)]
pub struct Yuv<T: YuvPixel> {
    data: [Vec<T>; 3],
    width: u32,
    height: u32,
    config: YuvConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YuvConfig {
    pub bit_depth: u8,
    pub subsampling_x: u8,
    pub subsampling_y: u8,
    pub full_range: bool,
    pub matrix_coefficients: MatrixCoefficients,
    pub transfer_characteristics: TransferCharacteristic,
}

impl<T: YuvPixel> Yuv<T> {
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
            let max_value = u16::MAX >> (16 - config.bit_depth);
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

impl<T: YuvPixel> From<Yuv<T>> for Xyb {
    fn from(other: Yuv<T>) -> Self {
        let lrgb = yuv_to_linear_rgb(&other);
        Xyb {
            data: linear_rgb_to_xyb(&lrgb),
            width: other.width(),
            height: other.height(),
        }
    }
}

impl<T: YuvPixel> TryFrom<(Xyb, YuvConfig)> for Yuv<T> {
    type Error = anyhow::Error;

    /// # Errors
    /// - If the `YuvConfig` would produce an invalid image
    fn try_from(other: (Xyb, YuvConfig)) -> Result<Self> {
        let lrgb = xyb_to_linear_rgb(&other.0.data);
        linear_rgb_to_yuv(&lrgb, other.0.width(), other.0.height(), other.1)
    }
}

#[cfg(test)]
mod tests {
    use interpolate_name::interpolate_test;
    use rand::Rng;

    use super::*;

    #[interpolate_test(bt601_420, MatrixCoefficients::ST170M, TransferCharacteristic::ST170M, (1, 1), false)]
    #[interpolate_test(bt601_422, MatrixCoefficients::ST170M, TransferCharacteristic::ST170M, (1, 0), false)]
    #[interpolate_test(bt601_444, MatrixCoefficients::ST170M, TransferCharacteristic::ST170M, (0, 0), false)]
    #[interpolate_test(bt601_444_full, MatrixCoefficients::ST170M, TransferCharacteristic::ST170M, (0, 0), true)]
    #[interpolate_test(bt709_420, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, (1, 1), false)]
    #[interpolate_test(bt709_422, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, (1, 0), false)]
    #[interpolate_test(bt709_444, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, (0, 0), false)]
    #[interpolate_test(bt709_444_full, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, (0, 0), true)]
    fn yuv_xyb_yuv_ident_8b(
        mc: MatrixCoefficients,
        tc: TransferCharacteristic,
        ss: (u8, u8),
        full_range: bool,
    ) {
        let y_dims = (320usize, 240usize);
        let uv_dims = (y_dims.0 >> ss.0, y_dims.1 >> ss.1);
        let mut data = [
            vec![0u8; y_dims.0 * y_dims.1],
            vec![0u8; uv_dims.0 * uv_dims.1],
            vec![0u8; uv_dims.0 * uv_dims.1],
        ];
        let mut rng = rand::thread_rng();
        for (i, plane) in data.iter_mut().enumerate() {
            for val in plane.iter_mut() {
                *val = rng.gen_range(if full_range {
                    0..=255
                } else if i == 0 {
                    16..=235
                } else {
                    16..=240
                });
            }
        }
        let yuv = Yuv::new(data, y_dims.0 as u32, y_dims.1 as u32, YuvConfig {
            bit_depth: 8,
            subsampling_x: ss.0,
            subsampling_y: ss.1,
            full_range,
            matrix_coefficients: mc,
            transfer_characteristics: tc,
        })
        .unwrap();
        let xyb = Xyb::from(yuv.clone());
        let yuv2 = Yuv::<u8>::try_from((xyb, yuv.config())).unwrap();
        // assert_eq!(yuv.data(), yuv2.data());
        assert_eq!(yuv.width(), yuv2.width());
        assert_eq!(yuv.height(), yuv2.height());
        assert_eq!(yuv.config(), yuv2.config());
    }

    #[interpolate_test(bt601_420, MatrixCoefficients::ST170M, TransferCharacteristic::ST170M, (1, 1), false)]
    #[interpolate_test(bt601_422, MatrixCoefficients::ST170M, TransferCharacteristic::ST170M, (1, 0), false)]
    #[interpolate_test(bt601_444, MatrixCoefficients::ST170M, TransferCharacteristic::ST170M, (0, 0), false)]
    #[interpolate_test(bt601_444_full, MatrixCoefficients::ST170M, TransferCharacteristic::ST170M, (0, 0), true)]
    #[interpolate_test(bt709_420, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, (1, 1), false)]
    #[interpolate_test(bt709_422, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, (1, 0), false)]
    #[interpolate_test(bt709_444, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, (0, 0), false)]
    #[interpolate_test(bt709_444_full, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, (0, 0), true)]
    fn xyb_yuv_xyb_ident_8b(
        mc: MatrixCoefficients,
        tc: TransferCharacteristic,
        ss: (u8, u8),
        full_range: bool,
    ) {
        let dims = (320usize, 240usize);
        let mut data = vec![[0.0f32; 3]; dims.0 * dims.1];
        let mut rng = rand::thread_rng();
        for plane in &mut data {
            for val in plane.iter_mut() {
                *val = rng.gen_range(0.0..=1.0);
            }
        }
        let config = YuvConfig {
            bit_depth: 8,
            subsampling_x: ss.0,
            subsampling_y: ss.1,
            full_range,
            matrix_coefficients: mc,
            transfer_characteristics: tc,
        };
        let xyb = Xyb::new(data, dims.0 as u32, dims.1 as u32).unwrap();
        let yuv = Yuv::<u8>::try_from((xyb.clone(), config)).unwrap();
        let xyb2 = Xyb::from(yuv);
        assert_eq!(xyb.data().len(), xyb2.data().len());
        for (pix1, pix2) in xyb.data().iter().zip(xyb2.data().iter()) {
            assert!((pix1[0] - pix2[0]).abs() < f32::EPSILON);
            assert!((pix1[1] - pix2[1]).abs() < f32::EPSILON);
            assert!((pix1[2] - pix2[2]).abs() < f32::EPSILON);
        }
        assert_eq!(xyb.width(), xyb2.width());
        assert_eq!(xyb.height(), xyb2.height());
    }

    #[interpolate_test(bt709_420, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, (1, 1), false)]
    #[interpolate_test(bt709_422, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, (1, 0), false)]
    #[interpolate_test(bt709_444, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, (0, 0), false)]
    #[interpolate_test(bt709_444_full, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, (0, 0), true)]
    #[interpolate_test(
        bt2020_420,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::BT2020Ten,
        (1, 1),
        false
    )]
    #[interpolate_test(
        bt2020_422,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::BT2020Ten,
        (1, 0),
        false
    )]
    #[interpolate_test(
        bt2020_444,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::BT2020Ten,
        (0, 0),
        false
    )]
    #[interpolate_test(
        bt2020_444_full,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::BT2020Ten,
        (0, 0),
        true
    )]
    #[interpolate_test(
        pq_420,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::PerceptualQuantizer,
        (1, 1),
        false
    )]
    #[interpolate_test(
        pq_422,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::PerceptualQuantizer,
        (1, 0),
        false
    )]
    #[interpolate_test(
        pq_444,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::PerceptualQuantizer,
        (0, 0),
        false
    )]
    #[interpolate_test(
        pq_444_full,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::PerceptualQuantizer,
        (0, 0),
        true
    )]
    #[interpolate_test(
        hlg_420,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::HybridLogGamma,
        (1, 1),
        false
    )]
    #[interpolate_test(
        hlg_422,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::HybridLogGamma,
        (1, 0),
        false
    )]
    #[interpolate_test(
        hlg_444,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::HybridLogGamma,
        (0, 0),
        false
    )]
    #[interpolate_test(
        hlg_444_full,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::HybridLogGamma,
        (0, 0),
        true
    )]
    fn yuv_xyb_yuv_ident_10b(
        mc: MatrixCoefficients,
        tc: TransferCharacteristic,
        ss: (u8, u8),
        full_range: bool,
    ) {
        let y_dims = (320usize, 240usize);
        let uv_dims = (y_dims.0 >> ss.0, y_dims.1 >> ss.1);
        let mut data = [
            vec![0u16; y_dims.0 * y_dims.1],
            vec![0u16; uv_dims.0 * uv_dims.1],
            vec![0u16; uv_dims.0 * uv_dims.1],
        ];
        let mut rng = rand::thread_rng();
        for (i, plane) in data.iter_mut().enumerate() {
            for val in plane.iter_mut() {
                *val = rng.gen_range(if full_range {
                    0..=1023
                } else if i == 0 {
                    64..=940
                } else {
                    64..=960
                });
            }
        }
        let yuv = Yuv::new(data, y_dims.0 as u32, y_dims.1 as u32, YuvConfig {
            bit_depth: 10,
            subsampling_x: ss.0,
            subsampling_y: ss.1,
            full_range,
            matrix_coefficients: mc,
            transfer_characteristics: tc,
        })
        .unwrap();
        let xyb = Xyb::from(yuv.clone());
        let yuv2 = Yuv::<u16>::try_from((xyb, yuv.config())).unwrap();
        // assert_eq!(yuv.data(), yuv2.data());
        assert_eq!(yuv.width(), yuv2.width());
        assert_eq!(yuv.height(), yuv2.height());
        assert_eq!(yuv.config(), yuv2.config());
    }

    #[interpolate_test(bt709_420, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, (1, 1), false)]
    #[interpolate_test(bt709_422, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, (1, 0), false)]
    #[interpolate_test(bt709_444, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, (0, 0), false)]
    #[interpolate_test(bt709_444_full, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, (0, 0), true)]
    #[interpolate_test(
        bt2020_420,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::BT2020Ten,
        (1, 1),
        false
    )]
    #[interpolate_test(
        bt2020_422,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::BT2020Ten,
        (1, 0),
        false
    )]
    #[interpolate_test(
        bt2020_444,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::BT2020Ten,
        (0, 0),
        false
    )]
    #[interpolate_test(
        bt2020_444_full,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::BT2020Ten,
        (0, 0),
        true
    )]
    #[interpolate_test(
        pq_420,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::PerceptualQuantizer,
        (1, 1),
        false
    )]
    #[interpolate_test(
        pq_422,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::PerceptualQuantizer,
        (1, 0),
        false
    )]
    #[interpolate_test(
        pq_444,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::PerceptualQuantizer,
        (0, 0),
        false
    )]
    #[interpolate_test(
        pq_444_full,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::PerceptualQuantizer,
        (0, 0),
        true
    )]
    #[interpolate_test(
        hlg_420,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::HybridLogGamma,
        (1, 1),
        false
    )]
    #[interpolate_test(
        hlg_422,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::HybridLogGamma,
        (1, 0),
        false
    )]
    #[interpolate_test(
        hlg_444,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::HybridLogGamma,
        (0, 0),
        false
    )]
    #[interpolate_test(
        hlg_444_full,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::HybridLogGamma,
        (0, 0),
        true
    )]
    fn xyb_yuv_xyb_ident_10b(
        mc: MatrixCoefficients,
        tc: TransferCharacteristic,
        ss: (u8, u8),
        full_range: bool,
    ) {
        let dims = (320usize, 240usize);
        let mut data = vec![[0.0f32; 3]; dims.0 * dims.1];
        let mut rng = rand::thread_rng();
        for plane in &mut data {
            for val in plane.iter_mut() {
                *val = rng.gen_range(0.0..=1.0);
            }
        }
        let config = YuvConfig {
            bit_depth: 10,
            subsampling_x: ss.0,
            subsampling_y: ss.1,
            full_range,
            matrix_coefficients: mc,
            transfer_characteristics: tc,
        };
        let xyb = Xyb::new(data, dims.0 as u32, dims.1 as u32).unwrap();
        let yuv = Yuv::<u16>::try_from((xyb.clone(), config)).unwrap();
        dbg!(&yuv.data()[0][0..8]);
        let xyb2 = Xyb::from(yuv);
        assert_eq!(xyb.data().len(), xyb2.data().len());
        // dbg!((&xyb.data()[0..8], &xyb2.data()[0..8]));
        for (pix1, pix2) in xyb.data().iter().zip(xyb2.data().iter()) {
            assert!((pix1[0] - pix2[0]).abs() < f32::EPSILON);
            assert!((pix1[1] - pix2[1]).abs() < f32::EPSILON);
            assert!((pix1[2] - pix2[2]).abs() < f32::EPSILON);
        }
        assert_eq!(xyb.width(), xyb2.width());
        assert_eq!(xyb.height(), xyb2.height());
    }
}
