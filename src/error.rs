use thiserror::Error;

/// Error type for box filter operations
#[derive(Error, Debug)]
pub enum BoxFilterError {
    #[error("Filter radius is too large: x_radius={x_radius}, y_radius={y_radius}, image_size=({width}, {height})")]
    RadiusTooLarge {
        x_radius: u32,
        y_radius: u32,
        width: u32,
        height: u32,
    },

    #[error("Cannot apply filter to an empty image")]
    EmptyImage,
}

/// Error type for alpha mask operations
#[derive(Debug, Error)]
pub enum Error {
    #[error("Image and mask dimensions do not match")]
    DimensionMismatch,
    #[error("Failed to create ImageBuffer from processed pixels")]
    ImageBufferCreationFailed,
}

/// Error type for padding operations
#[derive(Error, Debug)]
pub enum PaddingError {
    #[error("Padding width ({pad_width}) must be greater than or equal to image width ({width})")]
    PaddingWidthTooSmall { width: u32, pad_width: u32 },

    #[error(
        "Padding height ({pad_height}) must be greater than or equal to image height ({height})"
    )]
    PaddingHeightTooSmall { height: u32, pad_height: u32 },
}
