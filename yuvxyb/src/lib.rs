mod rgb_xyb;
mod yuv_rgb;

mod errors;
mod hsl;
mod linear_rgb;
mod rgb;
mod xyb;
mod yuv;

pub use crate::errors::{ConversionError, CreationError};
pub use crate::hsl::Hsl;
pub use crate::linear_rgb::LinearRgb;
pub use crate::rgb::Rgb;
pub use crate::xyb::Xyb;
pub use crate::yuv::{Yuv, YuvConfig, YuvError};
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
        let mut rng = rand::rng();
        for (i, plane) in data.planes.iter_mut().enumerate() {
            for val in plane.data_origin_mut().iter_mut() {
                *val = rng.random_range(if full_range {
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
        let mut rng = rand::rng();
        for (i, plane) in data.planes.iter_mut().enumerate() {
            for val in plane.data_origin_mut().iter_mut() {
                *val = rng.random_range(if full_range {
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
