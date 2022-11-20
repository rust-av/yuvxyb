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
mod fastmath;
#[doc(hidden)]
pub mod rgb_xyb;
#[doc(hidden)]
pub mod yuv_rgb;

#[doc(hidden)]
pub mod hsl;
pub mod xyb;
pub mod yuv;
pub mod rgb;
pub mod linear_rgb;

pub use crate::hsl::Hsl;
pub use crate::linear_rgb::LinearRgb;
pub use crate::rgb::Rgb;
pub use crate::xyb::Xyb;
pub use crate::yuv::{Yuv, YuvConfig};
pub use av_data::pixel::{ColorPrimaries, MatrixCoefficients, TransferCharacteristic};
pub use num_traits::{FromPrimitive, ToPrimitive};
pub use v_frame::{
    frame::Frame,
    plane::Plane,
    prelude::{CastFromPrimitive, Pixel},
};

#[cfg(test)]
mod tests {
    use interpolate_name::interpolate_test;
    use rand::Rng;
    use v_frame::plane::Plane;

    use crate::yuv::YuvConfig;

    use super::*;

    #[interpolate_test(bt601_420, MatrixCoefficients::ST170M, TransferCharacteristic::ST170M, ColorPrimaries::ST170M, (1, 1), false)]
    #[interpolate_test(bt601_422, MatrixCoefficients::ST170M, TransferCharacteristic::ST170M, ColorPrimaries::ST170M, (1, 0), false)]
    #[interpolate_test(bt601_444, MatrixCoefficients::ST170M, TransferCharacteristic::ST170M, ColorPrimaries::ST170M, (0, 0), false)]
    #[interpolate_test(bt601_444_full, MatrixCoefficients::ST170M, TransferCharacteristic::ST170M, ColorPrimaries::ST170M, (0, 0), true)]
    #[interpolate_test(bt709_420, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, ColorPrimaries::BT709, (1, 1), false)]
    #[interpolate_test(bt709_422, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, ColorPrimaries::BT709, (1, 0), false)]
    #[interpolate_test(bt709_444, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, ColorPrimaries::BT709, (0, 0), false)]
    #[interpolate_test(bt709_444_full, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, ColorPrimaries::BT709, (0, 0), true)]
    fn yuv_xyb_yuv_ident_8b(
        mc: MatrixCoefficients,
        tc: TransferCharacteristic,
        cp: ColorPrimaries,
        ss: (u8, u8),
        full_range: bool,
    ) {
        let y_dims = (320usize, 240usize);
        let uv_dims = (y_dims.0 >> ss.0, y_dims.1 >> ss.1);
        let mut data: Frame<u8> = Frame {
            planes: [
                Plane::new(y_dims.0, y_dims.1, 0, 0, 0, 0),
                Plane::new(
                    uv_dims.0,
                    uv_dims.1,
                    usize::from(ss.0),
                    usize::from(ss.1),
                    0,
                    0,
                ),
                Plane::new(
                    uv_dims.0,
                    uv_dims.1,
                    usize::from(ss.0),
                    usize::from(ss.1),
                    0,
                    0,
                ),
            ],
        };
        let mut rng = rand::thread_rng();
        for (i, plane) in data.planes.iter_mut().enumerate() {
            for val in plane.data_origin_mut().iter_mut() {
                *val = rng.gen_range(if full_range {
                    0..=255
                } else if i == 0 {
                    16..=235
                } else {
                    16..=240
                });
            }
        }
        let yuv = Yuv::new(
            data,
            YuvConfig {
                bit_depth: 8,
                subsampling_x: ss.0,
                subsampling_y: ss.1,
                full_range,
                matrix_coefficients: mc,
                transfer_characteristics: tc,
                color_primaries: cp,
            },
        )
        .unwrap();
        let xyb = Xyb::try_from(&yuv).unwrap();
        let yuv2 = Yuv::<u8>::try_from((xyb, yuv.config())).unwrap();
        // assert_eq!(yuv.data(), yuv2.data());
        assert_eq!(yuv.width(), yuv2.width());
        assert_eq!(yuv.height(), yuv2.height());
        assert_eq!(yuv.config(), yuv2.config());
    }

    #[interpolate_test(bt709_420, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, ColorPrimaries::BT709, (1, 1), false)]
    #[interpolate_test(bt709_422, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, ColorPrimaries::BT709, (1, 0), false)]
    #[interpolate_test(bt709_444, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, ColorPrimaries::BT709, (0, 0), false)]
    #[interpolate_test(bt709_444_full, MatrixCoefficients::BT709, TransferCharacteristic::BT1886, ColorPrimaries::BT709, (0, 0), true)]
    #[interpolate_test(
        bt2020_420,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::BT2020Ten,
        ColorPrimaries::BT2020,
        (1, 1),
        false
    )]
    #[interpolate_test(
        bt2020_422,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::BT2020Ten,
        ColorPrimaries::BT2020,
        (1, 0),
        false
    )]
    #[interpolate_test(
        bt2020_444,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::BT2020Ten,
        ColorPrimaries::BT2020,
        (0, 0),
        false
    )]
    #[interpolate_test(
        bt2020_444_full,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::BT2020Ten,
        ColorPrimaries::BT2020,
        (0, 0),
        true
    )]
    #[interpolate_test(
        pq_420,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::PerceptualQuantizer,
        ColorPrimaries::BT2020,
        (1, 1),
        false
    )]
    #[interpolate_test(
        pq_422,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::PerceptualQuantizer,
        ColorPrimaries::BT2020,
        (1, 0),
        false
    )]
    #[interpolate_test(
        pq_444,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::PerceptualQuantizer,
        ColorPrimaries::BT2020,
        (0, 0),
        false
    )]
    #[interpolate_test(
        pq_444_full,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::PerceptualQuantizer,
        ColorPrimaries::BT2020,
        (0, 0),
        true
    )]
    #[interpolate_test(
        hlg_420,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::HybridLogGamma,
        ColorPrimaries::BT2020,
        (1, 1),
        false
    )]
    #[interpolate_test(
        hlg_422,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::HybridLogGamma,
        ColorPrimaries::BT2020,
        (1, 0),
        false
    )]
    #[interpolate_test(
        hlg_444,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::HybridLogGamma,
        ColorPrimaries::BT2020,
        (0, 0),
        false
    )]
    #[interpolate_test(
        hlg_444_full,
        MatrixCoefficients::BT2020NonConstantLuminance,
        TransferCharacteristic::HybridLogGamma,
        ColorPrimaries::BT2020,
        (0, 0),
        true
    )]
    fn yuv_xyb_yuv_ident_10b(
        mc: MatrixCoefficients,
        tc: TransferCharacteristic,
        cp: ColorPrimaries,
        ss: (u8, u8),
        full_range: bool,
    ) {
        let y_dims = (320usize, 240usize);
        let uv_dims = (y_dims.0 >> ss.0, y_dims.1 >> ss.1);
        let mut data: Frame<u16> = Frame {
            planes: [
                Plane::new(y_dims.0, y_dims.1, 0, 0, 0, 0),
                Plane::new(
                    uv_dims.0,
                    uv_dims.1,
                    usize::from(ss.0),
                    usize::from(ss.1),
                    0,
                    0,
                ),
                Plane::new(
                    uv_dims.0,
                    uv_dims.1,
                    usize::from(ss.0),
                    usize::from(ss.1),
                    0,
                    0,
                ),
            ],
        };
        let mut rng = rand::thread_rng();
        for (i, plane) in data.planes.iter_mut().enumerate() {
            for val in plane.data_origin_mut().iter_mut() {
                *val = rng.gen_range(if full_range {
                    0..=1023
                } else if i == 0 {
                    64..=940
                } else {
                    64..=960
                });
            }
        }
        let yuv = Yuv::new(
            data,
            YuvConfig {
                bit_depth: 10,
                subsampling_x: ss.0,
                subsampling_y: ss.1,
                full_range,
                matrix_coefficients: mc,
                transfer_characteristics: tc,
                color_primaries: cp,
            },
        )
        .unwrap();
        let xyb = Xyb::try_from(&yuv).unwrap();
        let yuv2 = Yuv::<u16>::try_from((xyb, yuv.config())).unwrap();
        // assert_eq!(yuv.data(), yuv2.data());
        assert_eq!(yuv.width(), yuv2.width());
        assert_eq!(yuv.height(), yuv2.height());
        assert_eq!(yuv.config(), yuv2.config());
    }
}
