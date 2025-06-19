use thiserror::Error;

/// Error type for color conversion operations
///
/// This error type represents various failure modes that can occur
/// during color space conversions and related operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ConvertColorError {
    /// A value is outside the valid range for the target color type
    ///
    /// # Examples
    ///
    /// This error occurs when converting between color spaces where
    /// the source value cannot be represented in the target space.
    #[error("Value {0} is out of range for the target type")]
    ValueOutOfRange(f32),

    /// Image dimensions do not match expected values
    ///
    /// This error is returned when an operation requires images of
    /// specific dimensions but receives images with different sizes.
    #[error("Image dimensions mismatch: expected {expected_width}x{expected_height}, got {actual_width}x{actual_height}")]
    DimensionMismatch {
        expected_width: u32,
        expected_height: u32,
        actual_width: u32,
        actual_height: u32,
    },

    /// Failed to create an image buffer during processing
    ///
    /// This error indicates that memory allocation or buffer creation
    /// failed during the conversion process.
    #[error("Failed to create image buffer")]
    BufferCreationFailed,
}

/// Error type for alpha mask operations
///
/// This error type covers failures that can occur when applying
/// alpha masks to images or performing alpha-related operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum Error {
    /// Image and mask dimensions do not match
    ///
    /// This error occurs when attempting to apply an alpha mask
    /// to an image where the dimensions don't align properly.
    #[error("Image and mask dimensions do not match: expected {expected:?}, actual {actual:?}")]
    DimensionMismatch {
        /// Expected dimensions (width, height)
        expected: (u32, u32),
        /// Actual dimensions (width, height)  
        actual: (u32, u32),
    },

    /// Failed to create ImageBuffer from processed pixels
    ///
    /// This error indicates that the creation of a new image buffer
    /// failed after processing the pixel data.
    #[error("Failed to create ImageBuffer from processed pixels")]
    ImageBufferCreationFailed,

    /// Invalid parameter provided to the operation
    ///
    /// This error is returned when a parameter value is invalid
    /// or outside the acceptable range for the operation.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// Error type for padding operations
///
/// This error type represents failures that can occur during
/// image padding operations, typically related to size constraints.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PaddingError {
    /// Padding width is smaller than the image width
    ///
    /// This error occurs when the specified padding width is less
    /// than the original image width, which is not a valid operation.
    #[error("Padding width ({pad_width}) must be greater than or equal to image width ({width})")]
    PaddingWidthTooSmall { width: u32, pad_width: u32 },

    /// Padding height is smaller than the image height
    ///
    /// This error occurs when the specified padding height is less
    /// than the original image height, which is not a valid operation.
    #[error(
        "Padding height ({pad_height}) must be greater than or equal to image height ({height})"
    )]
    PaddingHeightTooSmall { height: u32, pad_height: u32 },
}
