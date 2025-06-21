//! Property-based tests for imageops-ai
//!
//! These tests use proptest to verify mathematical properties and invariants
//! that should hold for all possible inputs to our image operations.

use image::{Luma, Rgb, Rgba};
use imageops_ai::{
    AlphaPremultiply, ApplyAlphaMask, ForegroundEstimator, Image, Padding, Position,
};
use proptest::prelude::*;

/// Strategy for generating small but valid image dimensions
fn image_dimensions() -> impl Strategy<Value = (u32, u32)> {
    (1u32..=20, 1u32..=20)
}

/// Strategy for generating larger image dimensions for less expensive tests
fn larger_image_dimensions() -> impl Strategy<Value = (u32, u32)> {
    (10u32..=50, 10u32..=50)
}

/// Strategy for generating RGB pixel values
fn rgb_pixel() -> impl Strategy<Value = Rgb<u8>> {
    (any::<u8>(), any::<u8>(), any::<u8>()).prop_map(|(r, g, b)| Rgb([r, g, b]))
}

/// Strategy for generating RGBA pixel values
fn rgba_pixel() -> impl Strategy<Value = Rgba<u8>> {
    (any::<u8>(), any::<u8>(), any::<u8>(), any::<u8>()).prop_map(|(r, g, b, a)| Rgba([r, g, b, a]))
}

/// Strategy for generating alpha mask values
fn alpha_pixel() -> impl Strategy<Value = Luma<u8>> {
    any::<u8>().prop_map(|a| Luma([a]))
}

/// Strategy for generating padding positions
fn padding_position() -> impl Strategy<Value = Position> {
    prop_oneof![
        Just(Position::Top),
        Just(Position::Bottom),
        Just(Position::Left),
        Just(Position::Right),
        Just(Position::TopLeft),
        Just(Position::TopRight),
        Just(Position::BottomLeft),
        Just(Position::BottomRight),
        Just(Position::Center),
    ]
}

/// Create a test RGB image with given dimensions and fill pattern
fn create_test_rgb_image_with_pattern(
    width: u32,
    height: u32,
    pattern: impl Fn(u32, u32) -> Rgb<u8>,
) -> Image<Rgb<u8>> {
    let mut image: Image<Rgb<u8>> = Image::new(width, height);
    for y in 0..height {
        for x in 0..width {
            image.put_pixel(x, y, pattern(x, y));
        }
    }
    image
}

/// Create a test RGBA image with given dimensions and fill pattern
fn create_test_rgba_image_with_pattern(
    width: u32,
    height: u32,
    pattern: impl Fn(u32, u32) -> Rgba<u8>,
) -> Image<Rgba<u8>> {
    let mut image: Image<Rgba<u8>> = Image::new(width, height);
    for y in 0..height {
        for x in 0..width {
            image.put_pixel(x, y, pattern(x, y));
        }
    }
    image
}

/// Create a test alpha mask with given dimensions and fill pattern
fn create_test_alpha_mask_with_pattern(
    width: u32,
    height: u32,
    pattern: impl Fn(u32, u32) -> Luma<u8>,
) -> Image<Luma<u8>> {
    let mut mask: Image<Luma<u8>> = Image::new(width, height);
    for y in 0..height {
        for x in 0..width {
            mask.put_pixel(x, y, pattern(x, y));
        }
    }
    mask
}

proptest! {
    /// Property: Alpha premultiplication should preserve dimensions
    #[test]
    fn alpha_premultiply_preserves_dimensions(
        (width, height) in image_dimensions(),
        pixel in rgba_pixel()
    ) {
        let mut image: Image<Rgba<u8>> = Image::new(width, height);
        for y in 0..height {
            for x in 0..width {
                image.put_pixel(x, y, pixel);
            }
        }

        let result = image.premultiply_alpha().unwrap();
        prop_assert_eq!(result.dimensions(), (width, height));
    }

    /// Property: Alpha premultiplication with full opacity should preserve colors
    #[test]
    fn alpha_premultiply_full_opacity_preserves_colors(
        (width, height) in image_dimensions(),
        (r, g, b) in (any::<u8>(), any::<u8>(), any::<u8>())
    ) {
        let original_pixel = Rgba([r, g, b, 255]); // Full opacity
        let mut image: Image<Rgba<u8>> = Image::new(width, height);

        for y in 0..height {
            for x in 0..width {
                image.put_pixel(x, y, original_pixel);
            }
        }

        let result = image.premultiply_alpha().unwrap();

        // With full opacity, RGB values should be preserved
        for y in 0..height {
            for x in 0..width {
                let result_pixel = result.get_pixel(x, y);
                prop_assert_eq!(result_pixel[0], r);
                prop_assert_eq!(result_pixel[1], g);
                prop_assert_eq!(result_pixel[2], b);
            }
        }
    }

    /// Property: Alpha premultiplication with zero opacity should produce black
    #[test]
    fn alpha_premultiply_zero_opacity_produces_black(
        (width, height) in image_dimensions(),
        (r, g, b) in (any::<u8>(), any::<u8>(), any::<u8>())
    ) {
        let transparent_pixel = Rgba([r, g, b, 0]); // Fully transparent
        let mut image: Image<Rgba<u8>> = Image::new(width, height);

        for y in 0..height {
            for x in 0..width {
                image.put_pixel(x, y, transparent_pixel);
            }
        }

        let result = image.premultiply_alpha().unwrap();

        // With zero opacity, all RGB values should be 0 (black)
        for y in 0..height {
            for x in 0..width {
                let result_pixel = result.get_pixel(x, y);
                prop_assert_eq!(result_pixel[0], 0);
                prop_assert_eq!(result_pixel[1], 0);
                prop_assert_eq!(result_pixel[2], 0);
            }
        }
    }

    /// Property: Alpha mask application should preserve dimensions
    #[test]
    fn alpha_mask_preserves_dimensions(
        (width, height) in image_dimensions(),
        rgb_pixel in rgb_pixel(),
        alpha_pixel in alpha_pixel()
    ) {
        let image = create_test_rgb_image_with_pattern(width, height, |_, _| rgb_pixel);
        let mask = create_test_alpha_mask_with_pattern(width, height, |_, _| alpha_pixel);

        let result = image.apply_alpha_mask(&mask).unwrap();
        prop_assert_eq!(result.dimensions(), (width, height));
    }

    /// Property: Alpha mask with full opacity should preserve original colors
    #[test]
    fn alpha_mask_full_opacity_preserves_colors(
        (width, height) in image_dimensions(),
        rgb_pixel in rgb_pixel()
    ) {
        let image = create_test_rgb_image_with_pattern(width, height, |_, _| rgb_pixel);
        let mask = create_test_alpha_mask_with_pattern(width, height, |_, _| Luma([255])); // Full opacity

        let result = image.apply_alpha_mask(&mask).unwrap();

        for y in 0..height {
            for x in 0..width {
                let result_pixel = result.get_pixel(x, y);
                prop_assert_eq!(result_pixel[0], rgb_pixel[0]);
                prop_assert_eq!(result_pixel[1], rgb_pixel[1]);
                prop_assert_eq!(result_pixel[2], rgb_pixel[2]);
                prop_assert_eq!(result_pixel[3], 255); // Alpha should be full
            }
        }
    }

    /// Property: Alpha mask with zero opacity should produce transparent result
    #[test]
    fn alpha_mask_zero_opacity_produces_transparent(
        (width, height) in image_dimensions(),
        rgb_pixel in rgb_pixel()
    ) {
        let image = create_test_rgb_image_with_pattern(width, height, |_, _| rgb_pixel);
        let mask = create_test_alpha_mask_with_pattern(width, height, |_, _| Luma([0])); // Fully transparent

        let result = image.apply_alpha_mask(&mask).unwrap();

        for y in 0..height {
            for x in 0..width {
                let result_pixel = result.get_pixel(x, y);
                prop_assert_eq!(result_pixel[3], 0); // Alpha should be zero
            }
        }
    }

    /// Property: Padding should increase image dimensions correctly
    #[test]
    fn padding_increases_dimensions_correctly(
        (orig_width, orig_height) in image_dimensions(),
        (pad_width, pad_height) in image_dimensions(),
        position in padding_position(),
        fill_color in rgb_pixel()
    ) {
        // Ensure padding size is larger than original
        let pad_width = orig_width + pad_width;
        let pad_height = orig_height + pad_height;

        let image = create_test_rgb_image_with_pattern(orig_width, orig_height, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        });

        let result = image.add_padding((pad_width, pad_height), position, fill_color);
        prop_assert!(result.is_ok());

        if let Ok((padded, _)) = result {
            prop_assert_eq!(padded.dimensions(), (pad_width, pad_height));
        }
    }

    /// Property: Square padding should make non-square images square
    #[test]
    fn square_padding_makes_images_square(
        (width, height) in (1u32..=10, 1u32..=10).prop_filter("not equal", |(w, h)| w != h),
        fill_color in rgb_pixel()
    ) {
        let image = create_test_rgb_image_with_pattern(width, height, |_, _| Rgb([100, 100, 100]));

        let result = image.add_padding_square(fill_color).unwrap();
        let (padded, _) = result;
        let (final_width, final_height) = padded.dimensions();

        prop_assert_eq!(final_width, final_height); // Should be square
        prop_assert!(final_width >= width); // Should be at least as wide as original
        prop_assert!(final_height >= height); // Should be at least as tall as original
        prop_assert_eq!(final_width.max(final_height), width.max(height)); // Should be size of larger dimension
    }

    /// Property: Foreground estimation should preserve dimensions
    #[test]
    fn foreground_estimation_preserves_dimensions(
        (width, height) in (5u32..=15, 5u32..=15), // Larger minimum for blur operations
        rgb_pixel in rgb_pixel(),
        alpha_pixel in alpha_pixel(),
        radius in 1u32..=5 // Small radius for property tests
    ) {
        let image = create_test_rgb_image_with_pattern(width, height, |_, _| rgb_pixel);
        let mask = create_test_alpha_mask_with_pattern(width, height, |_, _| alpha_pixel);

        let result = image.estimate_foreground(&mask, radius);
        prop_assert!(result.is_ok());

        if let Ok(foreground) = result {
            prop_assert_eq!(foreground.dimensions(), (width, height));
        }
    }

    /// Property: Padding position calculation should be within bounds
    #[test]
    fn padding_position_within_bounds(
        (orig_width, orig_height) in image_dimensions(),
        (pad_width, pad_height) in image_dimensions(),
        position in padding_position()
    ) {
        // Ensure padding size is larger than original
        let pad_width = orig_width + pad_width;
        let pad_height = orig_height + pad_height;

        let image = create_test_rgb_image_with_pattern(orig_width, orig_height, |_, _| Rgb([0, 0, 0]));

        if let Ok((_, (pos_x, pos_y))) = image.add_padding((pad_width, pad_height), position, Rgb([255, 255, 255])) {
            prop_assert!(pos_x + orig_width <= pad_width);
            prop_assert!(pos_y + orig_height <= pad_height);
        }
    }

    /// Property: Combining operations should preserve essential properties
    #[test]
    fn combined_operations_preserve_properties(
        (width, height) in (3u32..=8, 3u32..=8), // Small for complex operations
        rgb_pixel in rgb_pixel()
    ) {
        let original_image = create_test_rgb_image_with_pattern(width, height, |_, _| rgb_pixel);
        let mask = create_test_alpha_mask_with_pattern(width, height, |x, y| {
            // Create a simple pattern
            if x < width / 2 && y < height / 2 { Luma([255]) } else { Luma([128]) }
        });

        // Apply sequence: RGB -> RGBA (via alpha mask) -> RGB (via premultiplication)
        let with_alpha = original_image.apply_alpha_mask(&mask);
        prop_assert!(with_alpha.is_ok());

        if let Ok(rgba_image) = with_alpha {
            let premultiplied = rgba_image.premultiply_alpha();
            prop_assert!(premultiplied.is_ok());

            if let Ok(final_image) = premultiplied {
                // Final image should have same dimensions as original
                prop_assert_eq!(final_image.dimensions(), (width, height));
            }
        }
    }

    /// Property: Error conditions should be handled gracefully
    #[test]
    fn padding_errors_handled_gracefully(
        (width, height) in image_dimensions(),
        position in padding_position()
    ) {
        let image = create_test_rgb_image_with_pattern(width, height, |_, _| Rgb([100, 100, 100]));

        // Try to pad to smaller dimensions (should fail)
        // Generate bad dimensions that are definitely smaller
        let bad_width = width.saturating_sub(1);
        let bad_height = height.saturating_sub(1);

        let result = image.add_padding((bad_width, bad_height), position, Rgb([255, 255, 255]));
        prop_assert!(result.is_err());
    }

    /// Property: Dimension mismatch errors should be detected
    #[test]
    fn dimension_mismatch_detected(
        (img_width, img_height) in image_dimensions(),
        (mask_width, mask_height) in image_dimensions()
    ) {
        // Only test when dimensions are actually different
        prop_assume!(mask_width != img_width || mask_height != img_height);
        let image = create_test_rgb_image_with_pattern(img_width, img_height, |_, _| Rgb([100, 100, 100]));
        let mask = create_test_alpha_mask_with_pattern(mask_width, mask_height, |_, _| Luma([128]));

        // Should fail due to dimension mismatch
        let result = image.apply_alpha_mask(&mask);
        prop_assert!(result.is_err());
    }
}
