//! Internal utility functions for imageops-ai.
//!
//! This module contains common functionality used across different image operations.

mod array_utils;
mod unify;

pub use array_utils::{array3_to_image, image_to_array3};

pub use unify::{unify_gray_images, unify_rgb_images, LargerType, NormalizedFrom};

/// Validates that an image has non-zero dimensions.
///
/// # Arguments
///
/// * `width` - The width of the image
/// * `height` - The height of the image
/// * `context` - A description of the context for error messages
///
/// # Returns
///
/// `Ok(())` if the dimensions are valid, otherwise an error
pub fn validate_non_empty_image(width: u32, height: u32, context: &str) -> Result<(), String> {
    if width == 0 || height == 0 {
        Err(format!("{}: Image dimensions must be non-zero", context))
    } else {
        Ok(())
    }
}

/// Validates that two images have matching dimensions.
///
/// # Arguments
///
/// * `width1` - The width of the first image
/// * `height1` - The height of the first image
/// * `width2` - The width of the second image
/// * `height2` - The height of the second image
/// * `context` - A description of the context for error messages
///
/// # Returns
///
/// `Ok(())` if the dimensions match, otherwise an error
pub fn validate_matching_dimensions(
    width1: u32,
    height1: u32,
    width2: u32,
    height2: u32,
    context: &str,
) -> Result<(), String> {
    if width1 != width2 || height1 != height2 {
        Err(format!(
            "{}: Image dimensions must match. Got {}x{} and {}x{}",
            context, width1, height1, width2, height2
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_non_empty_image() {
        assert!(validate_non_empty_image(100, 100, "test").is_ok());
        assert!(validate_non_empty_image(1, 1, "test").is_ok());
        assert!(validate_non_empty_image(0, 100, "test").is_err());
        assert!(validate_non_empty_image(100, 0, "test").is_err());
        assert!(validate_non_empty_image(0, 0, "test").is_err());
    }

    #[test]
    fn test_validate_matching_dimensions() {
        assert!(validate_matching_dimensions(100, 100, 100, 100, "test").is_ok());
        assert!(validate_matching_dimensions(50, 75, 50, 75, "test").is_ok());
        assert!(validate_matching_dimensions(100, 100, 100, 50, "test").is_err());
        assert!(validate_matching_dimensions(100, 100, 50, 100, "test").is_err());
        assert!(validate_matching_dimensions(100, 100, 50, 50, "test").is_err());
    }
}
