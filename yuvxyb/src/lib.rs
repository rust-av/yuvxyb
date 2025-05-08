mod rgb_xyb;
mod yuv_rgb;

mod errors;
mod hsl;
mod linear_rgb;
mod rgb;
mod xyb;
mod yuv;

#[cfg(test)]
mod tests;

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
