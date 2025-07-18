use thiserror::Error;

/// Error type for color conversion operations.
///
/// This error type represents various failure modes that can occur
/// during color space conversions and related operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ColorConversionError {
    /// A value is outside the valid range for the target color type.
    ///
    /// # Examples
    ///
    /// This error occurs when converting between color spaces where
    /// the source value cannot be represented in the target space.
    #[error("Value {0} is out of range for the target type")]
    ValueOutOfRange(f32),

    /// Image dimensions do not match expected values.
    ///
    /// This error is returned when an operation requires images of
    /// specific dimensions but receives images with different sizes.
    #[error(
        "Image dimensions mismatch: expected {expected_width}x{expected_height}, got {actual_width}x{actual_height}"
    )]
    DimensionMismatch {
        expected_width: u32,
        expected_height: u32,
        actual_width: u32,
        actual_height: u32,
    },

    /// Failed to create an image buffer during processing.
    ///
    /// This error indicates that memory allocation or buffer creation
    /// failed during the conversion process.
    #[error("Failed to create image buffer")]
    BufferCreationFailed,

    /// The input image has zero width or height.
    ///
    /// This error occurs when an operation receives an empty image
    /// (width or height is 0).
    #[error("Image has zero dimensions")]
    EmptyImage,
}

/// Error type for alpha mask operations.
///
/// This error type covers failures that can occur when applying
/// alpha masks to images or performing alpha-related operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AlphaMaskError {
    /// Image and mask dimensions do not match.
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

    /// Failed to create `ImageBuffer` from processed pixels.
    ///
    /// This error indicates that the creation of a new image buffer
    /// failed after processing the pixel data.
    #[error("Failed to create ImageBuffer from processed pixels")]
    ImageBufferCreationFailed,

    /// Invalid parameter provided to the operation.
    ///
    /// This error is returned when a parameter value is invalid
    /// or outside the acceptable range for the operation.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Error during blur fusion operation.
    ///
    /// This error indicates that an issue occurred during the
    /// blur fusion process, which combines foreground and background
    /// images with alpha and beta weights.
    #[error("Blur fusion error: {0}")]
    BlurFusionError(String),
}

/// Error type for padding operations.
///
/// This error type represents failures that can occur during
/// image padding operations, typically related to size constraints.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PaddingError {
    /// Padding width is smaller than the image width.
    ///
    /// This error occurs when the specified padding width is less
    /// than the original image width, which is not a valid operation.
    #[error("Padding width ({pad_width}) must be greater than or equal to image width ({width})")]
    PaddingWidthTooSmall { width: u32, pad_width: u32 },

    /// Padding height is smaller than the image height.
    ///
    /// This error occurs when the specified padding height is less
    /// than the original image height, which is not a valid operation.
    #[error(
        "Padding height ({pad_height}) must be greater than or equal to image height ({height})"
    )]
    PaddingHeightTooSmall { height: u32, pad_height: u32 },
}

/// Error type for clip minimum border operations.
///
/// This error type represents failures that can occur during
/// boundary detection and clipping operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ClipBorderError {
    /// Threshold value is invalid.
    ///
    /// This error occurs when the provided threshold value
    /// is outside the valid range for the pixel type.
    #[error("Invalid threshold value: {0}")]
    InvalidThreshold(String),

    /// Image is too small for clipping operation.
    ///
    /// This error occurs when the image dimensions are insufficient
    /// for the requested clipping operation.
    #[error("Image dimensions ({width}x{height}) are too small for clipping")]
    ImageTooSmall { width: u32, height: u32 },

    /// No content found to clip.
    ///
    /// This error occurs when no content is detected within
    /// the specified threshold, resulting in nothing to clip.
    #[error("No content found within threshold")]
    NoContentFound,
}

/// Error type for Non-Local Means operations.
///
/// This error type represents failures that can occur during
/// Non-Local Means denoising operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum NLMeansError {
    /// Invalid window size.
    ///
    /// This error occurs when the window size is even or zero.
    /// Window sizes must be odd positive integers.
    #[error("Invalid window size: {size}. Window size must be an odd positive integer")]
    InvalidWindowSize { size: u32 },

    /// Invalid filtering parameter.
    ///
    /// This error occurs when the filtering parameter h is zero or negative.
    #[error("Invalid filtering parameter h: {h}. Parameter must be positive")]
    InvalidFilteringParameter { h: f32 },

    /// Big window is smaller than or equal to small window.
    ///
    /// This error occurs when the search window is not larger than the patch window.
    #[error("Big window ({big_window}) must be larger than small window ({small_window})")]
    InvalidWindowSizes { small_window: u32, big_window: u32 },

    /// Image is too small for the specified window sizes.
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

/// Error type for Guided Filter operations.
///
/// This error type represents failures that can occur during
/// Guided Filter operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum GuidedFilterError {
    /// Invalid radius parameter.
    ///
    /// This error occurs when the radius is zero or too large.
    #[error("Invalid radius: {radius}. Radius must be a positive integer")]
    InvalidRadius { radius: u32 },

    /// Invalid epsilon parameter.
    ///
    /// This error occurs when epsilon is negative or zero.
    #[error("Invalid epsilon: {epsilon}. Epsilon must be positive")]
    InvalidEpsilon { epsilon: f32 },

    /// Invalid scale parameter for fast guided filter.
    ///
    /// This error occurs when scale is zero or one.
    #[error("Invalid scale: {scale}. Scale must be greater than 1")]
    InvalidScale { scale: u32 },

    /// Image dimensions do not match between guidance and input images.
    ///
    /// This error occurs when the guidance and input images have different dimensions.
    #[error("Image dimensions mismatch: guidance {guidance_dims:?}, input {input_dims:?}")]
    DimensionMismatch {
        guidance_dims: (u32, u32),
        input_dims: (u32, u32),
    },

    /// Image is too small for the specified parameters.
    ///
    /// This error occurs when the image dimensions are insufficient
    /// for the requested filter parameters.
    #[error("Image dimensions ({width}x{height}) are too small for radius {radius}")]
    ImageTooSmall {
        width: u32,
        height: u32,
        radius: u32,
    },
}

/// Error type for box filter operations.
///
/// This error type represents failures that can occur during
/// box filter operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum BoxFilterError {
    /// Invalid radius parameter.
    ///
    /// This error occurs when the radius is negative.
    #[error("Invalid radius: {radius}. Radius must be non-negative")]
    InvalidRadius { radius: i32 },

    /// Image has zero dimensions.
    ///
    /// This error occurs when the input image has zero width or height.
    #[error("Image has zero dimensions ({width}x{height})")]
    EmptyImage { width: u32, height: u32 },

    /// Image dimensions are too small for the filter radius.
    ///
    /// This error occurs when the image dimensions are insufficient
    /// for the requested filter radius.
    #[error("Image dimensions ({width}x{height}) are too small for radius {radius}")]
    ImageTooSmall {
        width: u32,
        height: u32,
        radius: u32,
    },
}

/// Error type for One-Sided Box Filter operations.
///
/// This error type represents failures that can occur during
/// One-Sided Box Filter operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OSBFilterError {
    /// Invalid radius parameter.
    ///
    /// This error occurs when the radius is zero or negative.
    #[error("Invalid radius: {radius}. Radius must be positive")]
    InvalidRadius { radius: u32 },

    /// Image has zero dimensions.
    ///
    /// This error occurs when the input image has zero width or height.
    #[error("Image has zero dimensions ({width}x{height})")]
    EmptyImage { width: u32, height: u32 },

    /// Image dimensions are too small for the filter radius.
    ///
    /// This error occurs when the image dimensions are insufficient
    /// for the requested filter radius.
    #[error("Image dimensions ({width}x{height}) are too small for radius {radius}")]
    ImageTooSmall {
        width: u32,
        height: u32,
        radius: u32,
    },

    /// Invalid iterations parameter.
    ///
    /// This error occurs when iterations is zero.
    #[error("Invalid iterations: {iterations}. Iterations must be positive")]
    InvalidIterations { iterations: u32 },
}

/// Error type for `INTER_AREA` resize operations.
///
/// This error type represents failures that can occur during
/// `INTER_AREA` resize operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum InterAreaError {
    /// Invalid target dimensions.
    ///
    /// This error occurs when the target width or height is zero.
    #[error("Invalid target dimensions: ({width}x{height}). Dimensions must be positive")]
    InvalidTargetDimensions { width: u32, height: u32 },

    /// Image has zero dimensions.
    ///
    /// This error occurs when the source image has zero width or height.
    #[error("Image has zero dimensions ({width}x{height})")]
    EmptyImage { width: u32, height: u32 },

    /// Upscaling not supported.
    ///
    /// This error occurs when attempting to upscale an image. `INTER_AREA`
    /// is primarily designed for downscaling operations.
    #[error(
        "Upscaling not supported. Source: ({src_width}x{src_height}), Target: ({target_width}x{target_height})"
    )]
    UpscalingNotSupported {
        src_width: u32,
        src_height: u32,
        target_width: u32,
        target_height: u32,
    },

    /// Invalid scale factor.
    ///
    /// This error occurs when the scale factor is invalid or results in
    /// computational issues.
    #[error("Invalid scale factor: {scale}. Scale must be greater than 0")]
    InvalidScale { scale: f64 },
}
