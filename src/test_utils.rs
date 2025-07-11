//! Test utilities for imageops-ai
//!
//! This module provides common functionality for testing image operations.
//! It is only compiled when running tests.

#[cfg(test)]
use image::{Luma, Pixel, Primitive, Rgb, Rgba};
#[cfg(test)]
use imageproc::definitions::Image;

/// Creates a test RGB image with predefined pixel values for testing.
///
/// This function creates a 2x2 test image with known pixel values:
/// - (0,0): [200, 150, 100]
/// - (1,0): [100, 200, 150]
/// - (0,1): [150, 100, 200]
/// - (1,1): [50, 75, 25]
///
/// # Returns
/// A 2x2 RGB image with u8 subpixels
#[cfg(test)]
pub fn create_test_rgb_image() -> Image<Rgb<u8>> {
    let mut image: Image<Rgb<u8>> = Image::new(2, 2);
    image.put_pixel(0, 0, Rgb([200, 150, 100]));
    image.put_pixel(1, 0, Rgb([100, 200, 150]));
    image.put_pixel(0, 1, Rgb([150, 100, 200]));
    image.put_pixel(1, 1, Rgb([50, 75, 25]));
    image
}

/// Creates a test RGBA image with predefined pixel values for testing.
///
/// This function creates a 2x2 test image with known pixel values including alpha:
/// - (0,0): [200, 150, 100, 255] (opaque)
/// - (1,0): [100, 200, 150, 128] (semi-transparent)
/// - (0,1): [150, 100, 200, 64]  (more transparent)
/// - (1,1): [50, 75, 25, 0]      (fully transparent)
///
/// # Returns
/// A 2x2 RGBA image with u8 subpixels
#[cfg(test)]
pub fn create_test_rgba_image() -> Image<Rgba<u8>> {
    let mut image: Image<Rgba<u8>> = Image::new(2, 2);
    image.put_pixel(0, 0, Rgba([200, 150, 100, 255]));
    image.put_pixel(1, 0, Rgba([100, 200, 150, 128]));
    image.put_pixel(0, 1, Rgba([150, 100, 200, 64]));
    image.put_pixel(1, 1, Rgba([50, 75, 25, 0]));
    image
}

/// Creates a test grayscale (Luma) image with predefined pixel values for testing.
///
/// This function creates a 2x2 test image with known grayscale values:
/// - (0,0): [200]
/// - (1,0): [150]
/// - (0,1): [100]
/// - (1,1): [50]
///
/// # Returns
/// A 2x2 Luma image with u8 subpixels
#[cfg(test)]
#[allow(dead_code)]
pub fn create_test_luma_image() -> Image<Luma<u8>> {
    let mut image: Image<Luma<u8>> = Image::new(2, 2);
    image.put_pixel(0, 0, Luma([200]));
    image.put_pixel(1, 0, Luma([150]));
    image.put_pixel(0, 1, Luma([100]));
    image.put_pixel(1, 1, Luma([50]));
    image
}

/// Creates a test alpha mask image with predefined alpha values for testing.
///
/// This function creates a 2x2 test mask with varying transparency levels:
/// - (0,0): [255] (fully opaque)
/// - (1,0): [192] (mostly opaque)
/// - (0,1): [128] (semi-transparent)
/// - (1,1): [64]  (mostly transparent)
///
/// # Returns
/// A 2x2 Luma image suitable for use as an alpha mask
#[cfg(test)]
#[allow(dead_code)]
pub fn create_test_alpha_mask() -> Image<Luma<u8>> {
    let mut mask: Image<Luma<u8>> = Image::new(2, 2);
    mask.put_pixel(0, 0, Luma([255]));
    mask.put_pixel(1, 0, Luma([192]));
    mask.put_pixel(0, 1, Luma([128]));
    mask.put_pixel(1, 1, Luma([64]));
    mask
}

/// Compares two pixel values with a tolerance for floating-point precision errors.
///
/// This function is useful when comparing the results of operations that involve
/// floating-point calculations, where exact equality may not be achievable due
/// to precision limitations.
///
/// # Arguments
/// * `expected` - The expected pixel value
/// * `actual` - The actual pixel value
/// * `tolerance` - The maximum allowed difference between values
///
/// # Returns
/// `true` if all subpixel values are within the tolerance, `false` otherwise
#[cfg(test)]
pub fn pixels_approx_equal<P>(expected: P, actual: P, tolerance: f32) -> bool
where
    P: Pixel,
    P::Subpixel: Primitive,
    f32: From<P::Subpixel>,
{
    if expected.channels().len() != actual.channels().len() {
        return false;
    }

    for (e, a) in expected.channels().iter().zip(actual.channels().iter()) {
        let diff = (f32::from(*e) - f32::from(*a)).abs();
        if diff > tolerance {
            return false;
        }
    }
    true
}

/// Compares two images pixel by pixel with a tolerance for floating-point errors.
///
/// This function compares entire images, useful for verifying the results of
/// image processing operations that may introduce small numerical errors.
///
/// # Arguments
/// * `expected` - The expected image
/// * `actual` - The actual image
/// * `tolerance` - The maximum allowed difference between pixel values
///
/// # Returns
/// `true` if all pixels are within tolerance and dimensions match, `false` otherwise
#[cfg(test)]
pub fn images_approx_equal<P>(expected: &Image<P>, actual: &Image<P>, tolerance: f32) -> bool
where
    P: Pixel,
    P::Subpixel: Primitive,
    f32: From<P::Subpixel>,
{
    if expected.dimensions() != actual.dimensions() {
        return false;
    }

    for (exp_pixel, act_pixel) in expected.pixels().zip(actual.pixels()) {
        if !pixels_approx_equal(*exp_pixel, *act_pixel, tolerance) {
            return false;
        }
    }
    true
}

/// Creates a large test image for performance testing.
///
/// This function creates a larger image (e.g., 100x100) filled with a pattern
/// that can be used for performance testing or testing with realistic image sizes.
///
/// # Arguments
/// * `width` - Width of the image to create
/// * `height` - Height of the image to create
///
/// # Returns
/// An RGB image filled with a checkerboard pattern
#[cfg(test)]
pub fn create_large_test_image(width: u32, height: u32) -> Image<Rgb<u8>> {
    let mut image: Image<Rgb<u8>> = Image::new(width, height);

    for y in 0..height {
        for x in 0..width {
            // Create a checkerboard pattern
            let color = if (x + y) % 2 == 0 {
                Rgb([200, 150, 100])
            } else {
                Rgb([100, 150, 200])
            };
            image.put_pixel(x, y, color);
        }
    }

    image
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_test_rgb_image_with_valid_input_creates_image() {
        let image = create_test_rgb_image();
        assert_eq!(image.dimensions(), (2, 2));
        assert_eq!(image.get_pixel(0, 0), &Rgb([200, 150, 100]));
        assert_eq!(image.get_pixel(1, 1), &Rgb([50, 75, 25]));
    }

    #[test]
    fn create_test_rgba_image_with_valid_input_creates_image() {
        let image = create_test_rgba_image();
        assert_eq!(image.dimensions(), (2, 2));
        assert_eq!(image.get_pixel(0, 0), &Rgba([200, 150, 100, 255]));
        assert_eq!(image.get_pixel(1, 1), &Rgba([50, 75, 25, 0]));
    }

    #[test]
    fn pixels_approx_equal_with_tolerant_comparison_returns_true() {
        let pixel1 = Rgb([100u8, 150u8, 200u8]);
        let pixel2 = Rgb([101u8, 149u8, 201u8]);
        let pixel3 = Rgb([105u8, 145u8, 205u8]);

        assert!(pixels_approx_equal(pixel1, pixel2, 1.5));
        assert!(!pixels_approx_equal(pixel1, pixel3, 1.5));
    }

    #[test]
    fn images_approx_equal_with_tolerant_comparison_returns_true() {
        let image1 = create_test_rgb_image();
        let mut image2 = create_test_rgb_image();

        // Slightly modify one pixel
        image2.put_pixel(0, 0, Rgb([201, 150, 100]));

        assert!(images_approx_equal(&image1, &image2, 1.5));
        assert!(!images_approx_equal(&image1, &image2, 0.5));
    }

    #[test]
    fn create_large_test_image_with_valid_input_creates_image() {
        let image = create_large_test_image(10, 10);
        assert_eq!(image.dimensions(), (10, 10));

        // Test checkerboard pattern
        assert_eq!(image.get_pixel(0, 0), &Rgb([200, 150, 100]));
        assert_eq!(image.get_pixel(1, 0), &Rgb([100, 150, 200]));
        assert_eq!(image.get_pixel(0, 1), &Rgb([100, 150, 200]));
        assert_eq!(image.get_pixel(1, 1), &Rgb([200, 150, 100]));
    }
}
