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
    #[error("Image dimensions mismatch: expected {expected_width}x{expected_height}, got {actual_width}x{actual_height}"
    )]
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
pub enum AlphaMaskError {
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

/// Error type for clip minimum border operations
///
/// This error type represents failures that can occur during
/// boundary detection and clipping operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ClipBorderError {
    /// Threshold value is invalid
    ///
    /// This error occurs when the provided threshold value
    /// is outside the valid range for the pixel type.
    #[error("Invalid threshold value: {0}")]
    InvalidThreshold(String),

    /// Image is too small for clipping operation
    ///
    /// This error occurs when the image dimensions are insufficient
    /// for the requested clipping operation.
    #[error("Image dimensions ({width}x{height}) are too small for clipping")]
    ImageTooSmall { width: u32, height: u32 },

    /// No content found to clip
    ///
    /// This error occurs when no content is detected within
    /// the specified threshold, resulting in nothing to clip.
    #[error("No content found within threshold")]
    NoContentFound,
}

/// Error type for Non-Local Means operations
///
/// This error type represents failures that can occur during
/// Non-Local Means denoising operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum NLMeansError {
    /// Invalid window size
    ///
    /// This error occurs when the window size is even or zero.
    /// Window sizes must be odd positive integers.
    #[error("Invalid window size: {size}. Window size must be an odd positive integer")]
    InvalidWindowSize { size: u32 },

    /// Invalid filtering parameter
    ///
    /// This error occurs when the filtering parameter h is zero or negative.
    #[error("Invalid filtering parameter h: {h}. Parameter must be positive")]
    InvalidFilteringParameter { h: f32 },

    /// Big window is smaller than or equal to small window
    ///
    /// This error occurs when the search window is not larger than the patch window.
    #[error("Big window ({big_window}) must be larger than small window ({small_window})")]
    InvalidWindowSizes { small_window: u32, big_window: u32 },

    /// Image is too small for the specified window sizes
    ///
    /// This error occurs when the image dimensions are insufficient
    /// for the requested window sizes.
    #[error("Image dimensions ({width}x{height}) are too small for big window size {big_window}")]
    ImageTooSmall {
        width: u32,
        height: u32,
        big_window: u32,
    },
}
