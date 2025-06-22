//! Internal utility functions for imageops-ai.
//!
//! This module contains common functionality used across different image operations.

mod unify;
pub use unify::{unify_gray_images, unify_rgb_images, LargerType, NormalizedFrom};

use image::Primitive;
use imageproc::definitions::Clamp;

/// Clamps a floating-point value to the range of a primitive type.
///
/// This function ensures that the input value is within the valid range
/// of the target primitive type. This version is used when Clamp trait is not available.
///
/// # Arguments
///
/// * `value` - The floating-point value to clamp
///
/// # Returns
///
/// The clamped value as the target primitive type
#[inline]
pub fn clamp_f32_to_primitive<T: Primitive + Clamp<f32>>(value: f32) -> T {
    T::clamp(value)
}

/// Normalizes an alpha value from a subpixel type to a floating-point value in the range [0, 1].
///
/// # Arguments
///
/// * `alpha` - The alpha value to normalize
///
/// # Returns
///
/// The normalized alpha value as a floating-point number between 0 and 1
#[inline]
#[allow(dead_code)]
pub fn normalize_alpha<S>(alpha: S) -> f32
where
    S: Into<f32> + Primitive,
{
    let alpha_f32 = alpha.into();
    let max_value = S::DEFAULT_MAX_VALUE.into();
    alpha_f32 / max_value
}

/// Normalizes an alpha value using a pre-computed max value.
///
/// This is more efficient when processing multiple pixels with the same type.
///
/// # Arguments
///
/// * `alpha` - The alpha value to normalize
/// * `max_value` - The pre-computed maximum value for the type
///
/// # Returns
///
/// The normalized alpha value as a floating-point number between 0 and 1
#[inline]
pub fn normalize_alpha_with_max<S>(alpha: S, max_value: f32) -> f32
where
    S: Into<f32> + Primitive,
{
    alpha.into() / max_value
}

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
    fn test_normalize_alpha_with_max() {
        assert_eq!(normalize_alpha_with_max(0u8, 255.0), 0.0);
        assert_eq!(normalize_alpha_with_max(127u8, 255.0), 127.0 / 255.0);
        assert_eq!(normalize_alpha_with_max(255u8, 255.0), 1.0);
    }

    #[test]
    fn test_clamp_f32_to_primitive() {
        // Test u8 clamping
        assert_eq!(clamp_f32_to_primitive::<u8>(-10.0), 0);
        assert_eq!(clamp_f32_to_primitive::<u8>(0.0), 0);
        assert_eq!(clamp_f32_to_primitive::<u8>(127.5), 127);
        assert_eq!(clamp_f32_to_primitive::<u8>(255.0), 255);
        assert_eq!(clamp_f32_to_primitive::<u8>(300.0), 255);

        // Test u16 clamping
        assert_eq!(clamp_f32_to_primitive::<u16>(-10.0), 0);
        assert_eq!(clamp_f32_to_primitive::<u16>(32767.5), 32767); // imageproc clamp behavior
        assert_eq!(clamp_f32_to_primitive::<u16>(65535.0), 65535);
        assert_eq!(clamp_f32_to_primitive::<u16>(70000.0), 65535);
    }

    #[test]
    fn test_normalize_alpha() {
        // Test u8 normalization
        assert_eq!(normalize_alpha::<u8>(0), 0.0);
        assert_eq!(normalize_alpha::<u8>(255), 1.0);
        assert_eq!(normalize_alpha::<u8>(127), 127.0 / 255.0);

        // Test u16 normalization
        assert_eq!(normalize_alpha::<u16>(0), 0.0);
        assert_eq!(normalize_alpha::<u16>(65535), 1.0);
        assert_eq!(normalize_alpha::<u16>(32767), 32767.0 / 65535.0);
    }

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
