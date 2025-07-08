use crate::error::BoxFilterError;
use image::{ImageBuffer, Pixel};
use imageproc::definitions::{Clamp, Image};
use std::marker::PhantomData;

/// Border handling types for box filter operations
/// Corresponds to OpenCV's BorderTypes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BorderType {
    /// Fill with constant value (default: 0)
    Constant = 0,
    /// Replicate edge pixels
    Replicate = 1,
    /// Reflect at edges (abcdefgh -> gfedcba|abcdefgh|hgfedcb)
    Reflect = 2,
    /// Wrap around (abcdefgh -> cdefgh|abcdefgh|abcdef)
    Wrap = 3,
    /// Reflect at edges without repeating edge pixels (abcdefgh -> gfedcb|abcdefgh|gfedcb)
    Reflect101 = 4,
}

impl Default for BorderType {
    fn default() -> Self {
        Self::Reflect101
    }
}

/// Trait for box filter implementations
pub trait BoxFilter<P>
where
    P: Pixel,
{
    /// Apply box filter to the image
    fn filter(&self, image: &Image<P>) -> Result<Image<P>, BoxFilterError>;
}

/// Separable box filter implementation inspired by OpenCV
/// Uses row and column filters for better memory efficiency
pub struct BoxFilterSeparable {
    radius: u32,
    border_type: BorderType,
}

/// Border handling helper functions
mod border_utils {
    use super::BorderType;

    /// Get pixel coordinate based on border type
    pub fn get_border_coord(coord: i32, size: u32, border_type: BorderType) -> u32 {
        match border_type {
            BorderType::Constant => coord.clamp(0, size as i32 - 1) as u32,
            BorderType::Replicate => coord.clamp(0, size as i32 - 1) as u32,
            BorderType::Reflect => {
                if coord < 0 {
                    (-coord - 1) as u32
                } else if coord >= size as i32 {
                    (2 * size as i32 - coord - 1) as u32
                } else {
                    coord as u32
                }
            }
            BorderType::Wrap => {
                if coord < 0 {
                    (size as i32 + coord) as u32
                } else if coord >= size as i32 {
                    (coord - size as i32) as u32
                } else {
                    coord as u32
                }
            }
            BorderType::Reflect101 => {
                if coord < 0 {
                    (-coord) as u32
                } else if coord >= size as i32 {
                    (2 * size as i32 - coord - 2) as u32
                } else {
                    coord as u32
                }
            }
        }
        .min(size - 1)
    }
}

impl BoxFilterSeparable {
    /// Create a new separable box filter with default border type
    pub const fn new(radius: u32) -> Result<Self, BoxFilterError> {
        Ok(Self {
            radius,
            border_type: BorderType::Reflect101,
        })
    }

    /// Create a new separable box filter with specified border type
    pub const fn new_with_border(
        radius: u32,
        border_type: BorderType,
    ) -> Result<Self, BoxFilterError> {
        Ok(Self {
            radius,
            border_type,
        })
    }

    /// Get the kernel size (2 * radius + 1)
    #[inline]
    pub const fn kernel_size(&self) -> u32 {
        2 * self.radius + 1
    }
}

/// Row filter for box filtering - processes horizontal sums
struct RowFilter<P: Pixel> {
    radius: u32,
    border_type: BorderType,
    _phantom: PhantomData<P>,
}

impl<P: Pixel> RowFilter<P> {
    const fn new(radius: u32, border_type: BorderType) -> Self {
        Self {
            radius,
            border_type,
            _phantom: PhantomData,
        }
    }

    /// Apply row filter to a single row
    fn apply_row<S>(&self, row: &[P], output: &mut [f32])
    where
        P: Pixel<Subpixel = S>,
        S: Into<f32> + Copy,
    {
        let width = row.len();
        let channels = P::CHANNEL_COUNT as usize;
        let kernel_size = self.kernel_size() as usize;

        // For each position in the row
        for x in 0..width {
            for c in 0..channels {
                let mut sum = 0.0f32;

                // Sum values in kernel window
                for kx in 0..kernel_size {
                    let src_x = (x as i32) + (kx as i32) - (self.radius as i32);
                    let pixel_idx =
                        border_utils::get_border_coord(src_x, width as u32, self.border_type)
                            as usize;
                    sum += row[pixel_idx].channels()[c].into();
                }

                output[x * channels + c] = sum;
            }
        }
    }

    const fn kernel_size(&self) -> u32 {
        2 * self.radius + 1
    }
}

/// Column filter for box filtering - processes vertical sums
struct ColumnFilter<P: Pixel> {
    radius: u32,
    border_type: BorderType,
    normalize: bool,
    _phantom: PhantomData<P>,
}

impl<P: Pixel> ColumnFilter<P> {
    const fn new(radius: u32, border_type: BorderType, normalize: bool) -> Self {
        Self {
            radius,
            border_type,
            normalize,
            _phantom: PhantomData,
        }
    }

    /// Apply column filter to intermediate buffer
    fn apply_column<S>(&self, buffer: &[Vec<f32>], output: &mut [P])
    where
        P: Pixel<Subpixel = S>,
        S: Clamp<f32> + Copy,
    {
        let height = buffer.len();
        let width = if height > 0 {
            buffer[0].len() / P::CHANNEL_COUNT as usize
        } else {
            0
        };
        let channels = P::CHANNEL_COUNT as usize;

        if width == 0 || height == 0 {
            return;
        }

        let kernel_area = if self.normalize {
            (self.kernel_size() * self.kernel_size()) as f32
        } else {
            1.0f32
        };

        for y in 0..height {
            for x in 0..width {
                let mut pixel_data = vec![S::clamp(0.0f32); channels];

                for c in 0..channels {
                    let mut sum = 0.0f32;

                    // Sum values in kernel window
                    for ky in 0..self.kernel_size() as usize {
                        let src_y = (y as i32) + (ky as i32) - (self.radius as i32);
                        let buffer_y =
                            border_utils::get_border_coord(src_y, height as u32, self.border_type)
                                as usize;
                        sum += buffer[buffer_y][x * channels + c];
                    }

                    let filtered_value = S::clamp(sum / kernel_area);
                    pixel_data[c] = filtered_value;
                }

                output[y * width + x] = *P::from_slice(&pixel_data);
            }
        }
    }

    const fn kernel_size(&self) -> u32 {
        2 * self.radius + 1
    }
}

/// Implement BoxFilter for separable filter
impl<P> BoxFilter<P> for BoxFilterSeparable
where
    P: Pixel,
    P::Subpixel: Clamp<f32> + Into<f32> + Copy,
{
    fn filter(&self, image: &Image<P>) -> Result<Image<P>, BoxFilterError> {
        let (width, height) = image.dimensions();

        if width == 0 || height == 0 {
            return Err(BoxFilterError::EmptyImage { width, height });
        }

        // Check if image is large enough for the filter radius
        let min_dimension = width.min(height);
        if min_dimension < self.kernel_size() {
            return Err(BoxFilterError::ImageTooSmall {
                width,
                height,
                radius: self.radius,
            });
        }

        let channels = P::CHANNEL_COUNT as usize;
        let row_filter = RowFilter::<P>::new(self.radius, self.border_type);
        let col_filter = ColumnFilter::<P>::new(self.radius, self.border_type, true);

        // First pass: apply row filter
        let mut intermediate_buffer = Vec::with_capacity(height as usize);

        for y in 0..height {
            let mut row_output = vec![0.0f32; width as usize * channels];
            let row_pixels: Vec<P> = (0..width).map(|x| *image.get_pixel(x, y)).collect();

            row_filter.apply_row(&row_pixels, &mut row_output);
            intermediate_buffer.push(row_output);
        }

        // Second pass: apply column filter
        let default_subpixel = P::Subpixel::clamp(0.0f32);
        let default_pixel = *P::from_slice(&vec![default_subpixel; channels]);
        let mut output_pixels = vec![default_pixel; (width * height) as usize];
        col_filter.apply_column(&intermediate_buffer, &mut output_pixels);

        // Convert back to ImageBuffer
        let output = ImageBuffer::from_fn(width, height, |x, y| {
            output_pixels[(y * width + x) as usize]
        });

        Ok(output)
    }
}

// No need for specific implementations - generic implementations work for all numeric types

/// Extension trait for ImageBuffer to provide fluent box filter methods
pub trait BoxFilterExt<P>
where
    P: Pixel,
{
    /// Apply box filter with specified border type (default: Reflect101)
    fn box_filter(self, radius: u32) -> Result<Self, BoxFilterError>
    where
        Self: Sized;

    /// Apply box filter with specified border type
    fn box_filter_with_border(
        self,
        radius: u32,
        border_type: BorderType,
    ) -> Result<Self, BoxFilterError>
    where
        Self: Sized;

    /// Apply box filter in-place (not available for this operation)
    fn box_filter_mut(&mut self, radius: u32) -> Result<&mut Self, BoxFilterError>;
}

impl<P> BoxFilterExt<P> for Image<P>
where
    P: Pixel,
    P::Subpixel: Clamp<f32> + Into<f32> + Copy,
{
    fn box_filter(self, radius: u32) -> Result<Self, BoxFilterError> {
        self.box_filter_with_border(radius, BorderType::default())
    }

    fn box_filter_with_border(
        self,
        radius: u32,
        border_type: BorderType,
    ) -> Result<Self, BoxFilterError> {
        let filter = BoxFilterSeparable::new_with_border(radius, border_type)?;
        filter.filter(&self)
    }

    /// Hidden _mut variant that is not available for this operation
    #[doc(hidden)]
    fn box_filter_mut(&mut self, _radius: u32) -> Result<&mut Self, BoxFilterError> {
        unimplemented!(
            "box_filter_mut is not available because the operation requires additional memory allocations equivalent to the owning version"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageBuffer;

    #[test]
    fn test_extension_trait() {
        use image::Rgb;
        let img = ImageBuffer::from_pixel(5, 5, Rgb([100u8, 100, 100]));

        // Test chaining
        let result = img.box_filter(1).unwrap().box_filter(1).unwrap();

        assert_eq!(result.dimensions(), (5, 5));
    }

    #[test]
    fn test_box_filter_separable_basic() {
        use image::Rgb;
        let mut img = ImageBuffer::from_pixel(5, 5, Rgb([100u8, 100, 100]));
        img.put_pixel(2, 2, Rgb([255, 255, 255]));

        let filter = BoxFilterSeparable::new(1).unwrap();
        let result = filter.filter(&img).unwrap();

        // Center pixel should be averaged with neighbors
        let center = result.get_pixel(2, 2);
        assert!(center[0] > 100 && center[0] < 255);
    }

    #[test]
    fn test_border_type_extension_trait() {
        use image::Rgb;
        let img = ImageBuffer::from_pixel(5, 5, Rgb([100u8, 100, 100]));

        // Test with different border types
        let result_default = img.clone().box_filter(1).unwrap();
        let result_replicate = img
            .clone()
            .box_filter_with_border(1, BorderType::Replicate)
            .unwrap();
        let result_reflect = img.box_filter_with_border(1, BorderType::Reflect).unwrap();

        assert_eq!(result_default.dimensions(), (5, 5));
        assert_eq!(result_replicate.dimensions(), (5, 5));
        assert_eq!(result_reflect.dimensions(), (5, 5));
    }

    #[test]
    fn test_edge_handling() {
        use image::Luma;
        let img = ImageBuffer::from_pixel(5, 5, Luma([100u8]));

        let filter = BoxFilterSeparable::new(1).unwrap();
        let result = filter.filter(&img).unwrap();

        // All pixels should remain 100 due to edge replication
        for y in 0..5 {
            for x in 0..5 {
                assert_eq!(result.get_pixel(x, y)[0], 100);
            }
        }
    }

    #[test]
    fn test_empty_image_error() {
        use image::Rgb;
        let img: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(0, 0);
        let filter = BoxFilterSeparable::new(1).unwrap();
        let result = filter.filter(&img);

        assert!(matches!(
            result,
            Err(BoxFilterError::EmptyImage {
                width: 0,
                height: 0
            })
        ));
    }

    #[test]
    fn test_image_too_small_error() {
        use image::Rgb;
        let img = ImageBuffer::from_pixel(2, 2, Rgb([100u8, 100, 100]));
        let filter = BoxFilterSeparable::new(2).unwrap(); // radius 2 needs at least 5x5 image
        let result = filter.filter(&img);

        assert!(matches!(
            result,
            Err(BoxFilterError::ImageTooSmall {
                width: 2,
                height: 2,
                radius: 2
            })
        ));
    }

    #[test]
    fn test_generic_types_u16() {
        use image::Rgb;
        let mut img = ImageBuffer::from_pixel(5, 5, Rgb([1000u16, 2000, 3000]));
        img.put_pixel(2, 2, Rgb([5000, 6000, 7000]));

        let filter = BoxFilterSeparable::new(1).unwrap();
        let result = filter.filter(&img).unwrap();

        // Center pixel should be averaged with neighbors
        let center = result.get_pixel(2, 2);
        assert!(center[0] > 1000 && center[0] < 5000);
        assert!(center[1] > 2000 && center[1] < 6000);
        assert!(center[2] > 3000 && center[2] < 7000);
    }

    #[test]
    fn test_generic_types_f32() {
        use image::Luma;
        let mut img = ImageBuffer::from_pixel(5, 5, Luma([0.5f32]));
        img.put_pixel(2, 2, Luma([1.0]));

        let filter = BoxFilterSeparable::new(1).unwrap();
        let result = filter.filter(&img).unwrap();

        // Center pixel should be averaged with neighbors
        let center = result.get_pixel(2, 2);
        assert!(center[0] > 0.5 && center[0] < 1.0);
    }

    #[test]
    fn test_performance_large_image_separable() {
        use image::Rgb;
        let img = ImageBuffer::from_pixel(500, 500, Rgb([100u8, 100, 100]));

        let start = std::time::Instant::now();
        let filter = BoxFilterSeparable::new(5).unwrap();
        let _result = filter.filter(&img).unwrap();
        let duration = start.elapsed();

        println!(
            "Box filter (500x500, radius=5) Separable took: {:?}",
            duration
        );
        // Should complete in reasonable time
        assert!(duration.as_millis() < 1000);
    }

    #[test]
    fn test_large_image_correctness_gradient() {
        use image::Rgb;
        const WIDTH: u32 = 1200;
        const HEIGHT: u32 = 1000;

        // Create gradient image
        let mut img = ImageBuffer::new(WIDTH, HEIGHT);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let val = ((x as f32 / WIDTH as f32) * 255.0) as u8;
                img.put_pixel(x, y, Rgb([val, val, val]));
            }
        }

        let radius = 10;
        let filter = BoxFilterSeparable::new(radius).unwrap();
        let result = filter.filter(&img).unwrap();

        // Test that the result has the expected dimensions
        assert_eq!(result.dimensions(), (WIDTH, HEIGHT));

        // Test that the gradient is properly smoothed
        let left_avg = result.get_pixel(100, 500)[0] as f32;
        let right_avg = result.get_pixel(WIDTH - 100, 500)[0] as f32;
        assert!(left_avg < right_avg, "Gradient should be preserved");
    }

    #[test]
    fn test_large_image_correctness_checkerboard() {
        use image::Luma;
        const WIDTH: u32 = 1024;
        const HEIGHT: u32 = 1024;

        // Create checkerboard pattern
        let mut img = ImageBuffer::new(WIDTH, HEIGHT);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let val = if ((x / 50) + (y / 50)) % 2 == 0 {
                    0u8
                } else {
                    255u8
                };
                img.put_pixel(x, y, Luma([val]));
            }
        }

        let radius = 15;
        let filter = BoxFilterSeparable::new(radius).unwrap();
        let result = filter.filter(&img).unwrap();

        // Test that the result has the expected dimensions
        assert_eq!(result.dimensions(), (WIDTH, HEIGHT));

        // Test that the checkerboard is properly smoothed
        let smoothed_center = result.get_pixel(WIDTH / 2, HEIGHT / 2)[0];
        assert!(
            smoothed_center > 0 && smoothed_center < 255,
            "Checkerboard should be smoothed"
        );
    }

    #[test]
    fn test_large_image_correctness_random() {
        use image::Rgb;
        use std::num::Wrapping;
        const WIDTH: u32 = 1100;
        const HEIGHT: u32 = 1000;

        // Create pseudo-random image using LCG
        let mut img = ImageBuffer::new(WIDTH, HEIGHT);
        let mut seed = Wrapping(42u32);

        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                // Simple LCG for deterministic randomness
                seed = Wrapping(1664525) * seed + Wrapping(1013904223);
                let val = (seed.0 % 256) as u8;
                img.put_pixel(
                    x,
                    y,
                    Rgb([val, val.wrapping_add(50), val.wrapping_add(100)]),
                );
            }
        }

        let radius = 20;
        let filter = BoxFilterSeparable::new(radius).unwrap();
        let result = filter.filter(&img).unwrap();

        // Test that the result has the expected dimensions
        assert_eq!(result.dimensions(), (WIDTH, HEIGHT));

        // Test that the random noise is properly smoothed
        let smoothed_center = result.get_pixel(WIDTH / 2, HEIGHT / 2);

        // The smoothed version should be within a reasonable range
        assert!(smoothed_center[0] > 0 && smoothed_center[0] < 255);
        assert!(smoothed_center[1] > 0 && smoothed_center[1] < 255);
        assert!(smoothed_center[2] > 0 && smoothed_center[2] < 255);
    }

    #[test]
    fn test_large_image_correctness_edge_cases() {
        use image::Rgba;
        const WIDTH: u32 = 1500;
        const HEIGHT: u32 = 1200;

        // Create image with edge cases (min/max values)
        let mut img = ImageBuffer::new(WIDTH, HEIGHT);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let val = match (x / 500, y / 400) {
                    (0, 0) => [0u8, 0, 0, 255],     // Black
                    (1, 0) => [255, 255, 255, 255], // White
                    (2, 0) => [255, 0, 0, 255],     // Red
                    (0, 1) => [0, 255, 0, 255],     // Green
                    (1, 1) => [0, 0, 255, 255],     // Blue
                    (2, 1) => [128, 128, 128, 255], // Gray
                    _ => [200, 150, 100, 255],      // Default
                };
                img.put_pixel(x, y, Rgba(val));
            }
        }

        let radius = 25;
        let filter = BoxFilterSeparable::new(radius).unwrap();
        let result = filter.filter(&img).unwrap();

        // Test that the result has the expected dimensions
        assert_eq!(result.dimensions(), (WIDTH, HEIGHT));

        // Test that the colors are properly blended
        let center = result.get_pixel(WIDTH / 2, HEIGHT / 2);
        // At least one channel should be less than 255
        assert!(center[0] < 255 || center[1] < 255 || center[2] < 255);
        assert_eq!(center[3], 255); // Alpha should be preserved
    }

    #[test]
    fn test_large_image_correctness_non_square() {
        use image::Rgb;
        const WIDTH: u32 = 2000;
        const HEIGHT: u32 = 1000;

        // Create diagonal gradient
        let mut img = ImageBuffer::new(WIDTH, HEIGHT);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let val =
                    (((x as f32 / WIDTH as f32 + y as f32 / HEIGHT as f32) / 2.0) * 255.0) as u8;
                img.put_pixel(x, y, Rgb([val, 255 - val, val / 2]));
            }
        }

        let radius = 30;
        let filter = BoxFilterSeparable::new(radius).unwrap();
        let result = filter.filter(&img).unwrap();

        // Test that the result has the expected dimensions
        assert_eq!(result.dimensions(), (WIDTH, HEIGHT));

        // Test that the gradient is properly smoothed
        let top_left = result.get_pixel(100, 100);
        let bottom_right = result.get_pixel(WIDTH - 100, HEIGHT - 100);

        // The diagonal gradient should be preserved
        assert!(top_left[0] < bottom_right[0]);
        assert!(top_left[1] > bottom_right[1]);
        assert!(top_left[2] < bottom_right[2]);
    }

    #[test]
    fn test_large_image_correctness_f32() {
        use image::Luma;
        const WIDTH: u32 = 1280;
        const HEIGHT: u32 = 1024;

        // Create gradient with f32 pixels
        let mut img = ImageBuffer::new(WIDTH, HEIGHT);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let val = ((x as f32 / WIDTH as f32) * (y as f32 / HEIGHT as f32)).sqrt();
                img.put_pixel(x, y, Luma([val]));
            }
        }

        let radius = 12;
        let filter = BoxFilterSeparable::new(radius).unwrap();
        let result = filter.filter(&img).unwrap();

        // Test that the result has the expected dimensions
        assert_eq!(result.dimensions(), (WIDTH, HEIGHT));

        // Test that the gradient is properly smoothed
        let top_left = result.get_pixel(100, 100)[0];
        let bottom_right = result.get_pixel(WIDTH - 100, HEIGHT - 100)[0];

        // The gradient should be preserved
        assert!(top_left < bottom_right);
    }
}
