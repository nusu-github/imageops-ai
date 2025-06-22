//! Tests with real image files
//!
//! These tests work with actual PNG and other image files to ensure our library
//! works correctly with real-world image data, including file I/O operations.

use image::{GrayImage, ImageBuffer, Luma, Rgb, RgbImage, Rgba, RgbaImage};
use std::path::{Path, PathBuf};

/// Get the path to test resources directory
#[allow(dead_code)]
fn resources_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("resources")
}

/// Create a test RGB image with realistic content and save it as PNG
#[allow(dead_code)]
fn create_and_save_test_rgb_image(filename: &str, width: u32, height: u32) -> PathBuf {
    let mut image: RgbImage = ImageBuffer::new(width, height);

    // Create a more realistic image with gradients and patterns
    for y in 0..height {
        for x in 0..width {
            let center_x = width as f32 / 2.0;
            let center_y = height as f32 / 2.0;
            let distance = (x as f32 - center_x).hypot(y as f32 - center_y);
            let max_distance = (width.min(height) as f32) / 2.0;

            if distance < max_distance * 0.3 {
                // Center - bright colors (subject)
                image.put_pixel(x, y, Rgb([220, 180, 150])); // Skin-like
            } else if distance < max_distance * 0.6 {
                // Middle ring - medium colors
                let intensity = (1.0 - distance / max_distance) * 255.0;
                image.put_pixel(
                    x,
                    y,
                    Rgb([
                        intensity as u8,
                        (intensity * 0.8) as u8,
                        (intensity * 0.6) as u8,
                    ]),
                );
            } else {
                // Outer ring - background gradient
                let r = ((x * 255) / width) as u8;
                let g = ((y * 255) / height) as u8;
                let b = ((x + y) * 255 / (width + height)) as u8;
                image.put_pixel(x, y, Rgb([r.min(200), g.min(200), b.min(200)]));
            }
        }
    }

    let path = resources_dir().join(filename);
    std::fs::create_dir_all(path.parent().unwrap()).expect("Failed to create resources directory");
    image.save(&path).expect("Failed to save test image");
    path
}

/// Create a test RGBA image with transparency and save it as PNG
#[allow(dead_code)]
fn create_and_save_test_rgba_image(filename: &str, width: u32, height: u32) -> PathBuf {
    let mut image: RgbaImage = ImageBuffer::new(width, height);

    let center_x = width as f32 / 2.0;
    let center_y = height as f32 / 2.0;
    let max_radius = (width.min(height) as f32) / 2.0;

    for y in 0..height {
        for x in 0..width {
            let distance = (x as f32 - center_x).hypot(y as f32 - center_y);

            let (r, g, b, a) = if distance < max_radius * 0.4 {
                // Center - opaque subject
                (200, 150, 100, 255)
            } else if distance < max_radius * 0.8 {
                // Middle - semi-transparent
                let alpha =
                    (255.0 * (1.0 - max_radius.mul_add(-0.4, distance) / (max_radius * 0.4))) as u8;
                (180, 140, 120, alpha)
            } else {
                // Outer - transparent background
                (100, 100, 100, 0)
            };

            image.put_pixel(x, y, Rgba([r, g, b, a]));
        }
    }

    let path = resources_dir().join(filename);
    std::fs::create_dir_all(path.parent().unwrap()).expect("Failed to create resources directory");
    image.save(&path).expect("Failed to save test RGBA image");
    path
}

/// Create a test alpha mask and save it as grayscale PNG
#[allow(dead_code)]
fn create_and_save_test_alpha_mask(filename: &str, width: u32, height: u32) -> PathBuf {
    let mut mask: GrayImage = ImageBuffer::new(width, height);

    let center_x = width as f32 / 2.0;
    let center_y = height as f32 / 2.0;
    let max_radius = (width.min(height) as f32) / 2.0;

    for y in 0..height {
        for x in 0..width {
            let distance = (x as f32 - center_x).hypot(y as f32 - center_y);

            let alpha = if distance <= max_radius * 0.6 {
                255 // Full opacity in center
            } else if distance <= max_radius {
                // Smooth falloff to edges
                (255.0 * (1.0 - max_radius.mul_add(-0.6, distance) / (max_radius * 0.4))) as u8
            } else {
                0 // Transparent outside
            };

            mask.put_pixel(x, y, Luma([alpha]));
        }
    }

    let path = resources_dir().join(filename);
    std::fs::create_dir_all(path.parent().unwrap()).expect("Failed to create resources directory");
    mask.save(&path).expect("Failed to save test alpha mask");
    path
}

#[cfg(feature = "test")]
#[test]
fn test_load_and_process_rgb_image() {
    // Create test image
    let image_path = create_and_save_test_rgb_image("test_rgb.png", 100, 80);

    // Load the image
    let loaded_image = image::open(&image_path).expect("Failed to load test image");
    let rgb_image = loaded_image.to_rgb8();
    let image: Image<Rgb<u8>> = Image::from(rgb_image);

    // Verify dimensions
    assert_eq!(image.dimensions(), (100, 80));

    // Test padding operation
    let (padded, position) = image
        .add_padding((120, 120), Position::Center, Rgb([255, 255, 255]))
        .expect("Padding should succeed");

    assert_eq!(padded.dimensions(), (120, 120));
    assert_eq!(position, (10, 20)); // Centered

    // Verify some pixel values
    let center_pixel = padded.get_pixel(60, 60); // Should be original content
    assert!(center_pixel[0] > 0 || center_pixel[1] > 0 || center_pixel[2] > 0);

    // Check padding area
    let padding_pixel = padded.get_pixel(5, 5); // Should be white padding
    assert_eq!(*padding_pixel, Rgb([255, 255, 255]));
}

#[cfg(feature = "test")]
#[test]
fn test_load_and_process_rgba_image() {
    // Create test RGBA image
    let image_path = create_and_save_test_rgba_image("test_rgba.png", 80, 80);

    // Load the image
    let loaded_image = image::open(&image_path).expect("Failed to load test RGBA image");
    let rgba_image = loaded_image.to_rgba8();
    let image: Image<Rgba<u8>> = Image::from(rgba_image);

    // Verify dimensions
    assert_eq!(image.dimensions(), (80, 80));

    // Test alpha premultiplication
    let rgb_result = image
        .premultiply_alpha()
        .expect("Alpha premultiplication should succeed");

    assert_eq!(rgb_result.dimensions(), (80, 80));

    // Check center pixel (should be opaque, so colors preserved)
    let center_pixel = rgb_result.get_pixel(40, 40);
    assert!(center_pixel[0] > 100); // Should have significant color values

    // Check edge pixel (should be darker due to transparency)
    let edge_pixel = rgb_result.get_pixel(5, 5);
    assert!(edge_pixel[0] < 50 && edge_pixel[1] < 50 && edge_pixel[2] < 50);
}

#[cfg(feature = "test")]
#[test]
fn test_alpha_mask_with_real_images() {
    // Create test images
    let rgb_path = create_and_save_test_rgb_image("test_rgb_for_mask.png", 60, 60);
    let mask_path = create_and_save_test_alpha_mask("test_alpha_mask.png", 60, 60);

    // Load images
    let rgb_loaded = image::open(&rgb_path).expect("Failed to load RGB image");
    let rgb_image: Image<Rgb<u8>> = Image::from(rgb_loaded.to_rgb8());

    let mask_loaded = image::open(&mask_path).expect("Failed to load alpha mask");
    let mask_image: Image<Luma<u8>> = Image::from(mask_loaded.to_luma8());

    // Apply alpha mask
    let result = rgb_image
        .apply_alpha_mask(&mask_image)
        .expect("Alpha mask application should succeed");

    assert_eq!(result.dimensions(), (60, 60));

    // Check center (should be opaque)
    let center_pixel = result.get_pixel(30, 30);
    assert_eq!(center_pixel[3], 255); // Full alpha

    // Check corner (should be transparent)
    let corner_pixel = result.get_pixel(5, 5);
    assert_eq!(corner_pixel[3], 0); // No alpha
}

#[cfg(feature = "test")]
#[test]
fn test_foreground_estimation_with_real_images() {
    // Create test images with smaller size for foreground estimation
    let rgb_path = create_and_save_test_rgb_image("test_rgb_for_fg.png", 40, 40);
    let mask_path = create_and_save_test_alpha_mask("test_mask_for_fg.png", 40, 40);

    // Load images
    let rgb_loaded = image::open(&rgb_path).expect("Failed to load RGB image");
    let rgb_image: Image<Rgb<u8>> = Image::from(rgb_loaded.to_rgb8());

    let mask_loaded = image::open(&mask_path).expect("Failed to load alpha mask");
    let mask_image: Image<Luma<u8>> = Image::from(mask_loaded.to_luma8());

    // Estimate foreground
    let foreground = rgb_image
        .estimate_foreground(&mask_image, 10) // Small radius for test
        .expect("Foreground estimation should succeed");

    assert_eq!(foreground.dimensions(), (40, 40));

    // Verify that foreground estimation produces reasonable results
    let center_pixel = foreground.get_pixel(20, 20);
    assert!(center_pixel[0] > 0 || center_pixel[1] > 0 || center_pixel[2] > 0);
}

#[cfg(feature = "test")]
#[test]
fn test_complete_workflow_with_real_images() {
    // Create test images
    let rgb_path = create_and_save_test_rgb_image("workflow_rgb.png", 50, 40);
    let mask_path = create_and_save_test_alpha_mask("workflow_mask.png", 50, 40);

    // Load images
    let rgb_loaded = image::open(&rgb_path).expect("Failed to load RGB image");
    let rgb_image: Image<Rgb<u8>> = Image::from(rgb_loaded.to_rgb8());

    let mask_loaded = image::open(&mask_path).expect("Failed to load alpha mask");
    let mask_image: Image<Luma<u8>> = Image::from(mask_loaded.to_luma8());

    // Complete workflow: Foreground estimation → Alpha mask → Premultiplication → Square padding
    let foreground = rgb_image
        .estimate_foreground(&mask_image, 8)
        .expect("Foreground estimation should succeed");

    let with_alpha = foreground
        .apply_alpha_mask(&mask_image)
        .expect("Alpha mask application should succeed");

    let premultiplied = with_alpha
        .premultiply_alpha()
        .expect("Alpha premultiplication should succeed");

    let (final_result, position) = premultiplied
        .add_padding_square(Rgb([128, 128, 128]))
        .expect("Square padding should succeed");

    // Verify final result
    assert_eq!(final_result.dimensions(), (50, 50)); // Should be square
    assert_eq!(position, (0, 5)); // Vertically centered (50x40 → 50x50)

    // Verify content preservation in center
    let center_pixel = final_result.get_pixel(25, 20);
    assert!(center_pixel[0] > 0 || center_pixel[1] > 0 || center_pixel[2] > 0);

    // Verify padding area
    let padding_pixel = final_result.get_pixel(25, 2); // Top padding area
    assert_eq!(*padding_pixel, Rgb([128, 128, 128]));
}

#[cfg(feature = "test")]
#[test]
fn test_large_real_image_processing() {
    // Test with a larger image to verify performance and memory handling
    let large_rgb_path = create_and_save_test_rgb_image("large_test.png", 200, 150);
    let large_mask_path = create_and_save_test_alpha_mask("large_mask.png", 200, 150);

    // Load images
    let rgb_loaded = image::open(&large_rgb_path).expect("Failed to load large RGB image");
    let rgb_image: Image<Rgb<u8>> = Image::from(rgb_loaded.to_rgb8());

    let mask_loaded = image::open(&large_mask_path).expect("Failed to load large alpha mask");
    let mask_image: Image<Luma<u8>> = Image::from(mask_loaded.to_luma8());

    // Test alpha mask application with larger image
    let result = rgb_image
        .apply_alpha_mask(&mask_image)
        .expect("Large image alpha mask should succeed");

    assert_eq!(result.dimensions(), (200, 150));

    // Test padding to even larger size
    let (padded, _) = result
        .premultiply_alpha()
        .expect("Large image premultiplication should succeed")
        .add_padding((250, 200), Position::Center, Rgb([64, 64, 64]))
        .expect("Large image padding should succeed");

    assert_eq!(padded.dimensions(), (250, 200));
}

#[cfg(feature = "test")]
#[test]
fn test_edge_case_real_images() {
    // Test with very small real images
    let tiny_rgb_path = create_and_save_test_rgb_image("tiny_test.png", 3, 3);
    let tiny_mask_path = create_and_save_test_alpha_mask("tiny_mask.png", 3, 3);

    // Load tiny images
    let rgb_loaded = image::open(&tiny_rgb_path).expect("Failed to load tiny RGB image");
    let rgb_image: Image<Rgb<u8>> = Image::from(rgb_loaded.to_rgb8());

    let mask_loaded = image::open(&tiny_mask_path).expect("Failed to load tiny alpha mask");
    let mask_image: Image<Luma<u8>> = Image::from(mask_loaded.to_luma8());

    // Test operations with tiny images
    let result = rgb_image
        .apply_alpha_mask(&mask_image)
        .expect("Tiny image processing should succeed");

    assert_eq!(result.dimensions(), (3, 3));

    // Test padding tiny image
    let (padded, position) = result
        .premultiply_alpha()
        .expect("Tiny image premultiplication should succeed")
        .add_padding((10, 10), Position::Center, Rgb([255, 0, 0]))
        .expect("Tiny image padding should succeed");

    assert_eq!(padded.dimensions(), (10, 10));
    assert_eq!(position, (3, 3)); // Should be centered
}
