use thiserror::Error;

/// Error type for when converting data from one color space to another fails.
///
/// Note that some conversions are infallible. These conversions will be
/// implemented in the [`From<T>`] trait. Check the type's documentation
/// to see which conversions are implemented.
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum ConversionError {
    #[error("Cannot convert between YUV and RGB using these matrix coefficients.")]
    UnsupportedMatrixCoefficients,
    #[error("No matrix coefficients were specified.")]
    UnspecifiedMatrixCoefficients,
    #[error("Cannot convert between YUV and RGB using these primaries.")]
    UnsupportedColorPrimaries,
    #[error("No primaries were specified.")]
    UnspecifiedColorPrimaries,
    #[error("Cannot convert between YUV and RGB using this transfer function.")]
    UnsupportedTransferCharacteristic,
    #[error("No transfer function was specified.")]
    UnspecifiedTransferCharacteristic,
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
#[derive(Debug, Error, Clone, Copy, PartialEq, Eq)]
pub enum CreationError {
    /// There is a mismatch between the supplied data and the supplied resolution.
    ///
    /// Generally, data.len() should be equal to width * height.
    #[error("Data length does not match the specified dimensions.")]
    ResolutionMismatch,
}
