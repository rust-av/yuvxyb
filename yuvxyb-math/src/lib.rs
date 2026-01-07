mod cbrtf;
mod matrix;
mod mul_add;
mod pow_exp;

#[cfg(feature = "simd")]
pub mod cbrtf_simd;

pub use cbrtf::cbrtf;
pub use matrix::{ColVector, Matrix, RowVector};
pub use mul_add::multiply_add;
pub use pow_exp::{expf, powf};

#[cfg(feature = "simd")]
pub use cbrtf_simd::{cbrtf_x16, cbrtf_x4, cbrtf_x8};
