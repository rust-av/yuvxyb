use std::fmt;

/// Error type for when converting data from one color space to another fails.
///
/// Note that some conversions are infallible. These conversions will be
/// implemented in the [`From<T>`] trait. Check the type's documentation
/// to see which conversions are implemented.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionError {
    UnsupportedMatrixCoefficients,
    UnspecifiedMatrixCoefficients,
    UnsupportedColorPrimaries,
    UnspecifiedColorPrimaries,
    UnsupportedTransferCharacteristic,
    UnspecifiedTransferCharacteristic,
}

impl std::error::Error for ConversionError {}

impl fmt::Display for ConversionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::UnsupportedMatrixCoefficients => write!(f, "Cannot convert between YUV and RGB using these matrix coefficients."),
            Self::UnspecifiedMatrixCoefficients => write!(f, "No matrix coefficients were specified."),
            Self::UnsupportedColorPrimaries => write!(f, "Cannot convert between YUV and RGB using these primaries."),
            Self::UnspecifiedColorPrimaries => write!(f, "No primaries were specified."),
            Self::UnsupportedTransferCharacteristic => write!(f, "Cannot convert between YUV and RGB using this transfer function."),
            Self::UnspecifiedTransferCharacteristic => write!(f, "No transfer function was specified."),
        }
    }
}

/// Error type for when creating one of the colorspace structs fails.
///
/// Note that the [`Yuv`] struct uses a separate Error type, [`YuvError`].
///
/// # Example
/// ```
/// use yuvxyb::{CreationError, LinearRgb};
///
/// // 10 pixels is not enough for the resolution 1920x1080
/// let float_data = vec![[0f32; 3]; 10];
/// let result = LinearRgb::new(float_data, 1920, 1080);
///
/// assert!(result.is_err());
/// assert_eq!(result.unwrap_err(), CreationError::ResolutionMismatch);
/// ```
///
/// [`Yuv`]: crate::yuv::Yuv
/// [`YuvError`]: crate::yuv::YuvError
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreationError {
    /// There is a mismatch between the supplied data and the supplied resolution.
    ///
    /// Generally, data.len() should be equal to width * height.
    ResolutionMismatch,
}

impl std::error::Error for CreationError {}

impl fmt::Display for CreationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::ResolutionMismatch => write!(f, "Data length does not match the specified dimensions."),
        }
    }
}
