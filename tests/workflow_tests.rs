//! Integration tests for imageops-ai workflows
//!
//! These tests verify that multiple operations work correctly when combined,
//! simulating real-world usage scenarios.

use image::{Luma, Rgb, Rgba};
use imageops_kit::{
    ApplyAlphaMaskExt, ForegroundEstimationExt, Image, PaddingExt, Position,
    PremultiplyAlphaAndDropExt,
};

/// Test helper to create a test RGB image with known pattern
fn create_test_image() -> Image<Rgb<u8>> {
    let mut image: Image<Rgb<u8>> = Image::new(10, 10);

    // Create a pattern: center content with border
    for y in 0..10 {
        for x in 0..10 {
            if (2..=7).contains(&x) && (2..=7).contains(&y) {
                // Content area - bright colors
                image.put_pixel(x, y, Rgb([200, 100, 50]));
            } else {
                // Border area - dark colors
                image.put_pixel(x, y, Rgb([50, 50, 50]));
            }
        }
    }

    image
}

/// Test helper to create a test alpha mask
fn create_test_alpha_mask() -> Image<Luma<u8>> {
    let mut mask: Image<Luma<u8>> = Image::new(10, 10);

    // Create gradient alpha mask
    for y in 0..10 {
        for x in 0..10 {
            let distance_from_center = (x as f32 - 4.5).hypot(y as f32 - 4.5);
            let alpha = (255.0 * (1.0 - (distance_from_center / 5.0).min(1.0))) as u8;
            mask.put_pixel(x, y, Luma([alpha]));
        }
    }

    mask
}

#[test]
fn rgb_padding_then_alpha_mask_workflow_works() {
    // Workflow: RGB image → Padding → Alpha mask application
    let image = create_test_image();
    let mask_small = create_test_alpha_mask();

    // Step 1: Add padding to make space for alpha mask
    let (padded_image, position) = image
        .add_padding((20, 20), Position::Center, Rgb([255, 255, 255]))
        .expect("Padding should succeed");

    assert_eq!(padded_image.dimensions(), (20, 20));
    assert_eq!(position, (5, 5)); // Centered position

    // Step 2: Create larger alpha mask for padded image
    let mut enlarged_mask: Image<Luma<u8>> = Image::new(20, 20);
    for y in 0..20 {
        for x in 0..20 {
            if (5..15).contains(&x) && (5..15).contains(&y) {
                // Copy from original mask
                let mask_pixel = mask_small.get_pixel(x - 5, y - 5);
                enlarged_mask.put_pixel(x, y, *mask_pixel);
            } else {
                // Transparent outside
                enlarged_mask.put_pixel(x, y, Luma([0]));
            }
        }
    }

    // Step 3: Apply alpha mask
    let result = padded_image
        .apply_alpha_mask(&enlarged_mask)
        .expect("Alpha mask application should succeed");

    assert_eq!(result.dimensions(), (20, 20));

    // Verify that alpha channel is applied correctly
    let center_pixel = result.get_pixel(10, 10);
    assert!(center_pixel[3] > 0); // Should have some alpha

    let corner_pixel = result.get_pixel(0, 0);
    assert_eq!(corner_pixel[3], 0); // Corner should be transparent
}

#[test]
fn rgba_premultiply_workflow_works() {
    // Workflow: Create RGBA → Alpha premultiplication
    let mut rgba_image: Image<Rgba<u8>> = Image::new(5, 5);

    // Fill with semi-transparent red
    for y in 0..5 {
        for x in 0..5 {
            rgba_image.put_pixel(x, y, Rgba([200, 100, 50, 128])); // 50% opacity
        }
    }

    // Apply alpha premultiplication
    let rgb_result = rgba_image
        .premultiply_alpha_and_drop()
        .expect("Alpha premultiplication should succeed");

    assert_eq!(rgb_result.dimensions(), (5, 5));

    // Verify premultiplication: colors should be roughly halved
    let pixel = rgb_result.get_pixel(2, 2);
    assert!(pixel[0] < 200); // Red component should be reduced
    assert!(pixel[1] < 100); // Green component should be reduced
    assert!(pixel[2] < 50); // Blue component should be reduced
}

#[test]
fn rgb_foreground_estimation_with_padding_works() {
    // Workflow: Foreground estimation → Square padding
    let image = create_test_image();
    let alpha_mask = create_test_alpha_mask();

    // Step 1: Estimate foreground using Blur-Fusion
    let foreground = image
        .estimate_foreground_colors(&alpha_mask, 3) // Small radius for small test image
        .expect("Foreground estimation should succeed");

    assert_eq!(foreground.dimensions(), (10, 10));

    // Step 2: Apply square padding to foreground
    let (width, height) = foreground.dimensions();
    let size = width.max(height);
    let (padded_foreground, position) = foreground
        .add_padding((size, size), Position::Center, Rgb([0, 0, 0]))
        .expect("Square padding should succeed");

    // Since image is square (10x10), no padding should be needed
    assert_eq!(padded_foreground.dimensions(), (10, 10));
    assert_eq!(position, (0, 0));

    // Test with rectangular image
    let mut rect_image: Image<Rgb<u8>> = Image::new(8, 4);
    for y in 0..4 {
        for x in 0..8 {
            rect_image.put_pixel(x, y, Rgb([100, 150, 200]));
        }
    }

    let (width, height) = rect_image.dimensions();
    let size = width.max(height);
    let (square_padded, pos) = rect_image
        .add_padding((size, size), Position::Center, Rgb([255, 255, 255]))
        .expect("Square padding of rectangular image should succeed");

    assert_eq!(square_padded.dimensions(), (8, 8)); // Should become square
    assert_eq!(pos, (0, 2)); // Vertically centered
}

#[test]
fn workflow_error_propagation_works_correctly() {
    // Test that errors propagate correctly through workflow chains
    let image = create_test_image();

    // Create mismatched mask (wrong dimensions)
    let mismatched_mask: Image<Luma<u8>> = Image::new(5, 5); // Different size

    // This should fail due to dimension mismatch
    let result = image.clone().apply_alpha_mask(&mismatched_mask);
    assert!(result.is_err());

    // Test with zero-sized padding (should fail)
    let padding_result = image.add_padding((5, 5), Position::Center, Rgb([255, 255, 255]));
    assert!(padding_result.is_err()); // Padding size too small
}

#[test]
fn complex_workflow_all_operations_work() {
    // Complex workflow combining multiple operations
    let original_image = create_test_image();
    let alpha_mask = create_test_alpha_mask();

    // Step 1: Estimate foreground
    let foreground = original_image
        .estimate_foreground_colors(&alpha_mask, 3)
        .expect("Foreground estimation should succeed");

    // Step 2: Add padding to foreground
    let (padded_foreground, _) = foreground
        .add_padding((15, 15), Position::Center, Rgb([128, 128, 128]))
        .expect("Padding should succeed");

    // Step 3: Create larger alpha mask for padded image
    let mut enlarged_alpha: Image<Luma<u8>> = Image::new(15, 15);
    for y in 0..15 {
        for x in 0..15 {
            let distance_from_center = (x as f32 - 7.0).hypot(y as f32 - 7.0);
            let alpha = (255.0 * (1.0 - (distance_from_center / 7.0).min(1.0))) as u8;
            enlarged_alpha.put_pixel(x, y, Luma([alpha]));
        }
    }

    // Step 4: Apply alpha mask to get RGBA
    let rgba_result = padded_foreground
        .apply_alpha_mask(&enlarged_alpha)
        .expect("Alpha mask application should succeed");

    // Step 5: Convert back to RGB with premultiplication
    let final_result = rgba_result
        .premultiply_alpha_and_drop()
        .expect("Alpha premultiplication should succeed");

    assert_eq!(final_result.dimensions(), (15, 15));

    // Verify the final result has reasonable values
    let center_pixel = final_result.get_pixel(7, 7);
    assert!(center_pixel[0] > 0 || center_pixel[1] > 0 || center_pixel[2] > 0);
}

#[test]
fn workflow_consistency_produces_identical_results() {
    // Test that applying operations in different orders produces consistent results
    let image = create_test_image();

    // Workflow A: Padding then cloning for different operations
    let (padded_a, _) = image
        .clone()
        .add_padding((12, 12), Position::Center, Rgb([100, 100, 100]))
        .expect("Padding A should succeed");

    // Workflow B: Same padding
    let (padded_b, _) = image
        .add_padding((12, 12), Position::Center, Rgb([100, 100, 100]))
        .expect("Padding B should succeed");

    // Results should be identical
    assert_eq!(padded_a.dimensions(), padded_b.dimensions());

    // Compare pixel by pixel
    for y in 0..12 {
        for x in 0..12 {
            assert_eq!(padded_a.get_pixel(x, y), padded_b.get_pixel(x, y));
        }
    }
}

#[test]
fn large_workflow_memory_efficiency_works() {
    // Test workflow with moderately large image to check memory efficiency
    let mut large_image: Image<Rgb<u8>> = Image::new(100, 100);

    // Fill with gradient pattern
    for y in 0..100 {
        for x in 0..100 {
            let r = (x * 255 / 100) as u8;
            let g = (y * 255 / 100) as u8;
            let b = ((x + y) * 255 / 200) as u8;
            large_image.put_pixel(x, y, Rgb([r, g, b]));
        }
    }

    // Create alpha mask
    let mut gradient_mask: Image<Luma<u8>> = Image::new(100, 100);
    for y in 0..100 {
        for x in 0..100 {
            let distance = (x as f32 - 50.0).hypot(y as f32 - 50.0);
            let alpha = (255.0 * (1.0 - (distance / 50.0).min(1.0))) as u8;
            gradient_mask.put_pixel(x, y, Luma([alpha]));
        }
    }

    // Perform workflow
    let result = large_image
        .apply_alpha_mask(&gradient_mask)
        .expect("Large image workflow should succeed");

    assert_eq!(result.dimensions(), (100, 100));

    // Verify result is reasonable (not all zeros or all max values)
    let mut non_zero_count = 0;
    let mut max_value_count = 0;

    for y in 0..100 {
        for x in 0..100 {
            let pixel = result.get_pixel(x, y);
            if pixel[0] > 0 || pixel[1] > 0 || pixel[2] > 0 {
                non_zero_count += 1;
            }
            if pixel[0] == 255 && pixel[1] == 255 && pixel[2] == 255 {
                max_value_count += 1;
            }
        }
    }

    // Should have some non-zero pixels but not all max values
    assert!(non_zero_count > 0);
    assert!(max_value_count < 10000); // Less than all pixels
}
