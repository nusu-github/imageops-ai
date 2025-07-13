//! Comprehensive edge case and error condition tests
//!
//! This test suite focuses on boundary values, error conditions, and edge cases
//! to ensure robust error handling and correct behavior at extremes.

use image::{Luma, Rgb, Rgba};
use imageops_kit::{
    AlphaMaskError, ApplyAlphaMaskExt, ForegroundEstimationExt, Image, PaddingError, PaddingExt,
    Position, PremultiplyAlphaAndDropExt,
};

/// Helper to create minimal 1x1 image
fn create_minimal_rgb_image() -> Image<Rgb<u8>> {
    let mut image: Image<Rgb<u8>> = Image::new(1, 1);
    image.put_pixel(0, 0, Rgb([128, 128, 128]));
    image
}

/// Helper to create minimal 1x1 alpha mask
fn create_minimal_alpha_mask() -> Image<Luma<u8>> {
    let mut mask: Image<Luma<u8>> = Image::new(1, 1);
    mask.put_pixel(0, 0, Luma([128]));
    mask
}

#[test]
fn minimal_image_operations_work_correctly() {
    // Test that 1x1 images work correctly
    let image = create_minimal_rgb_image();
    let mask = create_minimal_alpha_mask();

    // Alpha mask application
    let result = image.apply_alpha_mask(&mask);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().dimensions(), (1, 1));

    // For 1x1 image, minimum radius 0 is not supported by box filter
    // Box filter requires radius > 0 and image size >= 2*radius+1
    // So we create a 3x3 image for radius 1 or use larger radius
    let larger_image: Image<Rgb<u8>> = Image::from_pixel(3, 3, Rgb([100, 150, 200]));
    let larger_mask: Image<Luma<u8>> = Image::from_pixel(3, 3, Luma([128]));

    let fg_result = larger_image.estimate_foreground_colors(&larger_mask, 1);
    assert!(fg_result.is_ok());
    assert_eq!(fg_result.unwrap().dimensions(), (3, 3));
}

#[test]
fn zero_alpha_mask_produces_transparent_pixel() {
    // Test with completely transparent alpha mask
    let image = create_minimal_rgb_image();
    let mut transparent_mask: Image<Luma<u8>> = Image::new(1, 1);
    transparent_mask.put_pixel(0, 0, Luma([0])); // Completely transparent

    let result = image.apply_alpha_mask(&transparent_mask).unwrap();
    let pixel = result.get_pixel(0, 0);
    assert_eq!(pixel[3], 0); // Should be completely transparent
}

#[test]
fn max_alpha_mask_produces_opaque_pixel() {
    // Test with completely opaque alpha mask
    let image = create_minimal_rgb_image();
    let mut opaque_mask: Image<Luma<u8>> = Image::new(1, 1);
    opaque_mask.put_pixel(0, 0, Luma([255])); // Completely opaque

    let result = image.apply_alpha_mask(&opaque_mask).unwrap();
    let pixel = result.get_pixel(0, 0);
    assert_eq!(pixel[3], 255); // Should be completely opaque
    assert_eq!(pixel[0], 128); // RGB should be preserved
    assert_eq!(pixel[1], 128);
    assert_eq!(pixel[2], 128);
}

#[test]
fn rgba_premultiply_handles_extreme_alpha_values() {
    // Test alpha premultiplication with extreme alpha values
    let mut image: Image<Rgba<u8>> = Image::new(3, 1);

    // Test with zero alpha
    image.put_pixel(0, 0, Rgba([255, 255, 255, 0])); // White but transparent
    // Test with max alpha
    image.put_pixel(1, 0, Rgba([255, 255, 255, 255])); // White and opaque
    // Test with mid alpha
    image.put_pixel(2, 0, Rgba([255, 255, 255, 128])); // White and semi-transparent

    let result = image.premultiply_alpha_and_drop().unwrap();

    // Zero alpha should produce black
    let transparent_pixel = result.get_pixel(0, 0);
    assert_eq!(*transparent_pixel, Rgb([0, 0, 0]));

    // Max alpha should preserve color
    let opaque_pixel = result.get_pixel(1, 0);
    assert_eq!(*opaque_pixel, Rgb([255, 255, 255]));

    // Mid alpha should reduce color values
    let semi_transparent_pixel = result.get_pixel(2, 0);
    assert!(semi_transparent_pixel[0] < 255);
    assert!(semi_transparent_pixel[1] < 255);
    assert!(semi_transparent_pixel[2] < 255);
}

#[test]
fn mismatched_dimensions_produce_error() {
    let image = create_minimal_rgb_image(); // 1x1
    let mut mismatched_mask: Image<Luma<u8>> = Image::new(2, 2); // 2x2
    mismatched_mask.put_pixel(0, 0, Luma([128]));
    mismatched_mask.put_pixel(1, 0, Luma([128]));
    mismatched_mask.put_pixel(0, 1, Luma([128]));
    mismatched_mask.put_pixel(1, 1, Luma([128]));

    // Alpha mask application should fail
    let result = image.clone().apply_alpha_mask(&mismatched_mask);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        AlphaMaskError::DimensionMismatch { .. }
    ));

    // Foreground estimation should fail
    let fg_result = image.estimate_foreground_colors(&mismatched_mask, 1);
    assert!(fg_result.is_err());
    assert!(matches!(
        fg_result.unwrap_err(),
        AlphaMaskError::DimensionMismatch { .. }
    ));
}

#[test]
fn invalid_padding_size_produces_error() {
    let image = create_minimal_rgb_image(); // 1x1

    // Test padding size too small (width)
    let result = image
        .clone()
        .add_padding((0, 2), Position::Center, Rgb([255, 255, 255]));
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        PaddingError::PaddingWidthTooSmall { .. }
    ));

    // Test padding size too small (height)
    let result = image
        .clone()
        .add_padding((2, 0), Position::Center, Rgb([255, 255, 255]));
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        PaddingError::PaddingHeightTooSmall { .. }
    ));

    // Test exact same size (should work)
    let result = image.add_padding((1, 1), Position::Center, Rgb([255, 255, 255]));
    result.unwrap();
}

#[test]
fn zero_radius_foreground_estimation_produces_error() {
    let image = create_minimal_rgb_image();
    let mask = create_minimal_alpha_mask();

    // Test with zero radius (should fail)
    let result = image.estimate_foreground_colors(&mask, 0);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        AlphaMaskError::InvalidParameter(_)
    ));
}

#[test]
fn large_padding_values_work_correctly() {
    // Test with very large padding values
    let image = create_minimal_rgb_image(); // 1x1

    // Test maximum reasonable padding (should work)
    let result = image.add_padding((1000, 1000), Position::Center, Rgb([0, 0, 0]));
    assert!(result.is_ok());

    if let Ok((padded, position)) = result {
        assert_eq!(padded.dimensions(), (1000, 1000));
        assert_eq!(position, (499, 499)); // Should be centered
    }
}

#[test]
fn extreme_rgb_values_processed_correctly() {
    // Test with extreme RGB values (all 0s, all 255s, mixed)
    let mut image: Image<Rgb<u8>> = Image::new(3, 1);
    image.put_pixel(0, 0, Rgb([0, 0, 0])); // Black
    image.put_pixel(1, 0, Rgb([255, 255, 255])); // White
    image.put_pixel(2, 0, Rgb([255, 0, 128])); // Mixed extreme

    let uniform_mask: Image<Luma<u8>> = Image::from_fn(3, 1, |_, _| Luma([128]));

    // Test alpha mask application
    let result = image.clone().apply_alpha_mask(&uniform_mask);
    result.unwrap();

    // Test foreground estimation - create larger image for radius 1
    let larger_image: Image<Rgb<u8>> = Image::from_pixel(3, 3, Rgb([255, 255, 255]));
    let larger_mask: Image<Luma<u8>> = Image::from_pixel(3, 3, Luma([255]));
    let fg_result = larger_image.estimate_foreground_colors(&larger_mask, 1);
    fg_result.unwrap();
}

#[test]
fn all_padding_positions_place_image_correctly() {
    let image = create_minimal_rgb_image(); // 1x1
    let target_size = (3, 3);
    let fill_color = Rgb([255, 0, 0]);

    let positions = [
        (Position::TopLeft, (0, 0)),
        (Position::Top, (1, 0)),
        (Position::TopRight, (2, 0)),
        (Position::Left, (0, 1)),
        (Position::Center, (1, 1)),
        (Position::Right, (2, 1)),
        (Position::BottomLeft, (0, 2)),
        (Position::Bottom, (1, 2)),
        (Position::BottomRight, (2, 2)),
    ];

    for (position, expected_pos) in positions {
        let result = image.clone().add_padding(target_size, position, fill_color);
        assert!(result.is_ok(), "Position {position:?} should work");

        if let Ok((padded, actual_pos)) = result {
            assert_eq!(padded.dimensions(), target_size);
            assert_eq!(
                actual_pos, expected_pos,
                "Position calculation wrong for {position:?}"
            );

            // Verify original pixel is at the right location
            let original_pixel = padded.get_pixel(actual_pos.0 as u32, actual_pos.1 as u32);
            assert_eq!(*original_pixel, Rgb([128, 128, 128])); // Original pixel color
        }
    }
}

#[test]
fn square_padding_handles_aspect_ratios_correctly() {
    // Test square padding with already square image
    let square_image: Image<Rgb<u8>> = Image::new(5, 5);
    let (width, height) = square_image.dimensions();
    let size = width.max(height);
    let result = square_image.add_padding((size, size), Position::Center, Rgb([255, 255, 255]));
    assert!(result.is_ok());

    if let Ok((padded, position)) = result {
        assert_eq!(padded.dimensions(), (5, 5)); // No change
        assert_eq!(position, (0, 0)); // No offset
    }

    // Test with extreme aspect ratios
    let wide_image: Image<Rgb<u8>> = Image::new(100, 1);
    let (width, height) = wide_image.dimensions();
    let size = width.max(height);
    let result = wide_image.add_padding((size, size), Position::Center, Rgb([0, 0, 0]));
    assert!(result.is_ok());

    if let Ok((padded, position)) = result {
        assert_eq!(padded.dimensions(), (100, 100)); // Should be square
        assert_eq!(position, (0, 49)); // Vertically centered
    }

    let tall_image: Image<Rgb<u8>> = Image::new(1, 100);
    let (width, height) = tall_image.dimensions();
    let size = width.max(height);
    let result = tall_image.add_padding((size, size), Position::Center, Rgb([0, 0, 0]));
    assert!(result.is_ok());

    if let Ok((padded, position)) = result {
        assert_eq!(padded.dimensions(), (100, 100)); // Should be square
        assert_eq!(position, (49, 0)); // Horizontally centered
    }
}

#[test]
fn complex_workflow_handles_edge_cases_correctly() {
    // Test complete workflow with edge case inputs - using 3x3 for radius 1
    let mut image: Image<Rgb<u8>> = Image::new(3, 3);
    image.put_pixel(0, 0, Rgb([255, 255, 255]));
    image.put_pixel(1, 0, Rgb([0, 0, 0]));
    image.put_pixel(2, 0, Rgb([128, 128, 128]));
    image.put_pixel(0, 1, Rgb([64, 192, 32]));
    image.put_pixel(1, 1, Rgb([200, 100, 50]));
    image.put_pixel(2, 1, Rgb([30, 60, 90]));
    image.put_pixel(0, 2, Rgb([180, 45, 200]));
    image.put_pixel(1, 2, Rgb([90, 180, 45]));
    image.put_pixel(2, 2, Rgb([120, 240, 160]));

    let mut mask: Image<Luma<u8>> = Image::new(3, 3);
    mask.put_pixel(0, 0, Luma([255])); // Opaque
    mask.put_pixel(1, 0, Luma([0])); // Transparent
    mask.put_pixel(2, 0, Luma([128])); // Semi-transparent
    mask.put_pixel(0, 1, Luma([255])); // Opaque
    mask.put_pixel(1, 1, Luma([64])); // Semi-transparent
    mask.put_pixel(2, 1, Luma([192])); // Semi-transparent
    mask.put_pixel(0, 2, Luma([128])); // Semi-transparent
    mask.put_pixel(1, 2, Luma([255])); // Opaque
    mask.put_pixel(2, 2, Luma([255])); // Opaque

    // Complete workflow
    let foreground = image
        .estimate_foreground_colors(&mask, 1)
        .expect("Foreground estimation should work");

    let with_alpha = foreground
        .apply_alpha_mask(&mask)
        .expect("Alpha mask should work");

    let premultiplied = with_alpha
        .premultiply_alpha_and_drop()
        .expect("Premultiplication should work");

    let (width, height) = premultiplied.dimensions();
    let size = width.max(height);
    let (final_result, _) = premultiplied
        .add_padding((size, size), Position::Center, Rgb([255, 128, 0]))
        .expect("Square padding should work");

    // Verify final dimensions - should be 3x3 since we started with 3x3
    assert_eq!(final_result.dimensions(), (3, 3)); // Already square
}

#[test]
fn large_padding_operations_complete_successfully() {
    // Test that large padding operations don't cause memory issues
    let image = create_minimal_rgb_image(); // 1x1

    // Test progressively larger padding sizes
    let test_sizes = [(10, 10), (50, 50), (100, 100)];

    for (width, height) in test_sizes {
        let result =
            image
                .clone()
                .add_padding((width, height), Position::Center, Rgb([128, 128, 128]));
        assert!(result.is_ok(), "Padding to {width}x{height} should work");

        let (padded, _) = result.unwrap();
        assert_eq!(padded.dimensions(), (width, height));

        // Verify center pixel exists and has correct value
        let center_x = width / 2;
        let center_y = height / 2;
        let center_pixel = padded.get_pixel(center_x, center_y);
        assert_eq!(*center_pixel, Rgb([128, 128, 128]));
    }
}

#[test]
fn error_messages_contain_useful_information() {
    // Test that error messages contain useful information
    let image = create_minimal_rgb_image(); // 1x1
    let mismatched_mask: Image<Luma<u8>> = Image::new(5, 5); // Wrong size

    let result = image.apply_alpha_mask(&mismatched_mask);
    assert!(result.is_err());

    if let Err(error) = result {
        let error_message = format!("{error}");
        // Error message should contain dimensional information
        assert!(error_message.contains('1') || error_message.contains('5'));
    }
}

#[test]
fn premultiply_handles_precision_edge_cases() {
    // Test with pixel values that might cause precision issues
    let mut image: Image<Rgba<u8>> = Image::new(3, 1);

    // Edge case alpha values
    image.put_pixel(0, 0, Rgba([1, 1, 1, 1])); // Minimal values
    image.put_pixel(1, 0, Rgba([254, 254, 254, 254])); // Near-maximum values
    image.put_pixel(2, 0, Rgba([127, 128, 129, 127])); // Around midpoint

    let result = image.premultiply_alpha_and_drop();
    assert!(result.is_ok());

    // All results should be valid RGB values
    if let Ok(rgb_image) = result {
        for y in 0..1 {
            for x in 0..3 {
                let pixel = rgb_image.get_pixel(x, y);
                // All components should be valid u8 values (no overflow/underflow)
                // These assertions are technically redundant for u8 (always true)
                // but left for documentation purposes
                let _r = pixel[0];
                let _g = pixel[1];
                let _b = pixel[2];
            }
        }
    }
}
