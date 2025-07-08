use crate::error::BoxFilterError;
use image::{ImageBuffer, Pixel, Primitive};
use imageproc::definitions::{Clamp, Image};

/// Trait for box filter implementations
pub trait BoxFilter<P>
where
    P: Pixel,
{
    /// Apply box filter to the image
    fn filter(&self, image: &Image<P>) -> Result<Image<P>, BoxFilterError>;
}

/// Integral image based box filter with O(1) complexity per pixel
pub struct BoxFilterIntegral {
    radius: u32,
}

impl BoxFilterIntegral {
    /// Create a new integral image based box filter
    pub const fn new(radius: u32) -> Result<Self, BoxFilterError> {
        Ok(Self { radius })
    }

    /// Get the kernel size (2 * radius + 1)
    #[inline]
    pub const fn kernel_size(&self) -> u32 {
        2 * self.radius + 1
    }
}

/// One Pass Summed Area Table box filter with O(1) complexity
pub struct BoxFilterOPSAT {
    radius: u32,
}

impl BoxFilterOPSAT {
    /// Create a new OP-SAT box filter
    pub const fn new(radius: u32) -> Result<Self, BoxFilterError> {
        Ok(Self { radius })
    }

    /// Get the kernel size (2 * radius + 1)
    #[inline]
    pub const fn kernel_size(&self) -> u32 {
        2 * self.radius + 1
    }
}

/// Helper function to pad image with edge replication using optimized ImageBuffer::from_fn
fn pad_image<P, S>(image: &ImageBuffer<P, Vec<S>>, radius: u32) -> ImageBuffer<P, Vec<S>>
where
    P: Pixel<Subpixel = S>,
    S: Primitive,
{
    let (width, height) = image.dimensions();
    let new_width = width + 2 * radius;
    let new_height = height + 2 * radius;

    ImageBuffer::from_fn(new_width, new_height, |x, y| {
        // Clamp coordinates to original image bounds for edge replication
        let orig_x = if x < radius {
            0
        } else if x >= width + radius {
            width - 1
        } else {
            x - radius
        };

        let orig_y = if y < radius {
            0
        } else if y >= height + radius {
            height - 1
        } else {
            y - radius
        };

        *image.get_pixel(orig_x, orig_y)
    })
}

/// Implement BoxFilter for pixel types with numeric subpixels
impl<P> BoxFilter<P> for BoxFilterIntegral
where
    P: Pixel,
    P::Subpixel: Clamp<f32> + Into<f32> + Primitive,
{
    fn filter(&self, image: &Image<P>) -> Result<Image<P>, BoxFilterError> {
        let (width, height) = image.dimensions();

        if width == 0 || height == 0 {
            return Err(BoxFilterError::EmptyImage { width, height });
        }

        // Check if image is large enough for the filter radius
        let min_dimension = width.min(height);
        if min_dimension < 2 * self.radius + 1 {
            return Err(BoxFilterError::ImageTooSmall {
                width,
                height,
                radius: self.radius,
            });
        }

        let padded = pad_image(image, self.radius);
        let (pad_width, pad_height) = padded.dimensions();
        let channels = P::CHANNEL_COUNT as usize;

        // Use flat vectors for better cache locality
        let integral_width = (pad_width + 1) as usize;
        let integral_height = (pad_height + 1) as usize;
        let integral_size = integral_width * integral_height;

        // Flat vector per channel for better memory layout
        let mut channel_integrals = vec![vec![0.0f32; integral_size]; channels];

        // Build integral images for each channel with improved indexing
        padded.enumerate_pixels().for_each(|(x, y, pixel)| {
            let pixel_channels = pixel.channels();
            for c in 0..channels {
                let val = pixel_channels[c].into();
                let current_idx = ((y + 1) as usize) * integral_width + ((x + 1) as usize);
                let top_idx = (y as usize) * integral_width + ((x + 1) as usize);
                let left_idx = ((y + 1) as usize) * integral_width + (x as usize);
                let diag_idx = (y as usize) * integral_width + (x as usize);

                channel_integrals[c][current_idx] =
                    val + channel_integrals[c][top_idx] + channel_integrals[c][left_idx]
                        - channel_integrals[c][diag_idx];
            }
        });

        // Create output image with optimized indexing
        let kernel_area = (self.kernel_size() * self.kernel_size()) as f32;
        let inv_kernel_area = 1.0 / kernel_area; // Pre-compute reciprocal for multiplication

        let output = ImageBuffer::from_fn(width, height, |x, y| {
            // Convert to padded coordinates
            let py = y + self.radius;
            let px = x + self.radius;

            // Box boundaries in integral coordinates
            let y1 = py - self.radius;
            let y2 = py + self.radius + 1;
            let x1 = px - self.radius;
            let x2 = px + self.radius + 1;

            // Use fixed-size array instead of Vec for better performance
            let mut pixel_data = [P::Subpixel::DEFAULT_MIN_VALUE; 4];

            for c in 0..channels {
                // Use flat indexing for better cache performance with unsafe for bounds-checked removal
                let y2_idx = (y2 as usize) * integral_width + (x2 as usize);
                let y1_idx = (y1 as usize) * integral_width + (x2 as usize);
                let y2_x1_idx = (y2 as usize) * integral_width + (x1 as usize);
                let y1_x1_idx = (y1 as usize) * integral_width + (x1 as usize);

                // SAFETY: Indices are guaranteed to be within bounds due to padding calculation
                let box_sum = unsafe {
                    *channel_integrals[c].get_unchecked(y2_idx)
                        - *channel_integrals[c].get_unchecked(y1_idx)
                        - *channel_integrals[c].get_unchecked(y2_x1_idx)
                        + *channel_integrals[c].get_unchecked(y1_x1_idx)
                };

                let filtered_value = P::Subpixel::clamp(box_sum * inv_kernel_area);
                pixel_data[c] = filtered_value;
            }

            *P::from_slice(&pixel_data[..channels])
        });

        Ok(output)
    }
}

/// Implement BoxFilter OPSAT for pixel types with numeric subpixels
impl<P> BoxFilter<P> for BoxFilterOPSAT
where
    P: Pixel,
    P::Subpixel: Clamp<f32> + Into<f32> + Primitive,
{
    fn filter(&self, image: &Image<P>) -> Result<Image<P>, BoxFilterError> {
        let (width, height) = image.dimensions();

        if width == 0 || height == 0 {
            return Err(BoxFilterError::EmptyImage { width, height });
        }

        // Check if image is large enough for the filter radius
        let min_dimension = width.min(height);
        if min_dimension < 2 * self.radius + 1 {
            return Err(BoxFilterError::ImageTooSmall {
                width,
                height,
                radius: self.radius,
            });
        }

        let padded = pad_image(image, self.radius);
        let (pad_width, pad_height) = padded.dimensions();
        let channels = P::CHANNEL_COUNT as usize;

        // Use flat vectors for J buffers for better cache locality
        let j_buffer_size = (pad_width as usize) * (pad_height as usize);
        let mut j_buffers: Vec<Vec<f32>> = vec![vec![0.0f32; j_buffer_size]; channels];

        // Initialize J for first kernel_size columns with flat indexing
        let init_cols = self.kernel_size().min(pad_width);
        for x in 0..init_cols {
            for y in self.radius..(height + self.radius) {
                let y_start = y - self.radius;
                let y_end = y + self.radius + 1;

                for c in 0..channels {
                    let mut sum = 0.0f32;
                    for yy in y_start..y_end {
                        let pixel = padded.get_pixel(x, yy);
                        sum += pixel.channels()[c].into();
                    }
                    let idx = (y as usize) * (pad_width as usize) + (x as usize);
                    j_buffers[c][idx] = sum;
                }
            }
        }

        // Create output image
        let mut output = ImageBuffer::new(width, height);
        let kernel_area = (self.kernel_size() * self.kernel_size()) as f32;
        let normalization = 1.0 / kernel_area; // Pre-compute reciprocal

        // Process in raster scan order
        for y in 0..height {
            for x in 0..width {
                let py = y + self.radius;
                let px = x + self.radius;

                // Update J(x+r, y) if needed
                let j_x = px + self.radius;
                if j_x < pad_width {
                    if y == 0 {
                        // First row - compute directly
                        let y_start = py - self.radius;
                        let y_end = py + self.radius + 1;

                        for c in 0..channels {
                            let mut sum = 0.0f32;
                            for yy in y_start..y_end {
                                let pixel = padded.get_pixel(j_x, yy);
                                sum += pixel.channels()[c].into();
                            }
                            let idx = (py as usize) * (pad_width as usize) + (j_x as usize);
                            j_buffers[c][idx] = sum;
                        }
                    } else {
                        // Recursive update
                        let top_pixel = padded.get_pixel(j_x, py + self.radius);
                        let bottom_pixel = padded.get_pixel(j_x, py - self.radius - 1);

                        for c in 0..channels {
                            let current_idx = (py as usize) * (pad_width as usize) + (j_x as usize);
                            let prev_idx =
                                ((py - 1) as usize) * (pad_width as usize) + (j_x as usize);
                            j_buffers[c][current_idx] = j_buffers[c][prev_idx]
                                + top_pixel.channels()[c].into()
                                - bottom_pixel.channels()[c].into();
                        }
                    }
                }

                // Compute output
                let mut pixel_data = [P::Subpixel::DEFAULT_MIN_VALUE; 4]; // Fixed-size array for better performance

                if x == 0 {
                    // First column - compute directly
                    for c in 0..channels {
                        let mut sum = 0.0f32;
                        for i in 0..self.kernel_size() {
                            let idx = (py as usize) * (pad_width as usize) + (i as usize);
                            // SAFETY: Index is guaranteed to be within bounds
                            sum += unsafe { *j_buffers[c].get_unchecked(idx) };
                        }
                        pixel_data[c] = P::Subpixel::clamp(sum * normalization);
                    }
                } else {
                    // Get previous output pixel
                    let prev_pixel: &P = output.get_pixel(x - 1, y);
                    let j_right_idx =
                        (py as usize) * (pad_width as usize) + ((px + self.radius) as usize);
                    let j_left_idx =
                        (py as usize) * (pad_width as usize) + ((px - self.radius - 1) as usize);

                    for c in 0..channels {
                        let prev_val: f32 = prev_pixel.channels()[c].into();
                        // SAFETY: Indices are guaranteed to be within bounds
                        let new_val = unsafe {
                            (*j_buffers[c].get_unchecked(j_right_idx)
                                - *j_buffers[c].get_unchecked(j_left_idx))
                            .mul_add(normalization, prev_val)
                        };
                        pixel_data[c] = P::Subpixel::clamp(new_val);
                    }
                }

                let pixel = *P::from_slice(&pixel_data[..channels]);
                output.put_pixel(x, y, pixel);
            }
        }
        Ok(output)
    }
}

// No need for specific implementations - generic implementations work for all numeric types

/// Extension trait for ImageBuffer to provide fluent box filter methods
pub trait BoxFilterExt<P>
where
    P: Pixel,
{
    /// Apply integral image based box filter
    fn box_filter_integral(self, radius: u32) -> Result<Self, BoxFilterError>
    where
        Self: Sized;

    /// Apply integral image based box filter in-place
    fn box_filter_integral_mut(&mut self, radius: u32) -> Result<&mut Self, BoxFilterError>;

    /// Apply OP-SAT box filter
    fn box_filter_opsat(self, radius: u32) -> Result<Self, BoxFilterError>
    where
        Self: Sized;

    /// Apply OP-SAT box filter in-place
    fn box_filter_opsat_mut(&mut self, radius: u32) -> Result<&mut Self, BoxFilterError>;
}

impl<P> BoxFilterExt<P> for Image<P>
where
    P: Pixel,
    P::Subpixel: Clamp<f32> + Into<f32> + Primitive,
{
    fn box_filter_integral(self, radius: u32) -> Result<Self, BoxFilterError> {
        let filter = BoxFilterIntegral::new(radius)?;
        filter.filter(&self)
    }

    /// Hidden _mut variant that is not available for this operation
    #[doc(hidden)]
    fn box_filter_integral_mut(&mut self, _radius: u32) -> Result<&mut Self, BoxFilterError> {
        unimplemented!(
            "box_filter_integral_mut is not available because the operation requires additional memory allocations equivalent to the owning version"
        )
    }

    fn box_filter_opsat(self, radius: u32) -> Result<Self, BoxFilterError> {
        let filter = BoxFilterOPSAT::new(radius)?;
        filter.filter(&self)
    }

    /// Hidden _mut variant that is not available for this operation
    #[doc(hidden)]
    fn box_filter_opsat_mut(&mut self, _radius: u32) -> Result<&mut Self, BoxFilterError> {
        unimplemented!(
            "box_filter_opsat_mut is not available because the operation requires additional memory allocations equivalent to the owning version"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::ImageBuffer;

    #[test]
    fn test_box_filter_integral_basic() {
        use image::Rgb;
        let mut img = ImageBuffer::from_pixel(5, 5, Rgb([100u8, 100, 100]));
        img.put_pixel(2, 2, Rgb([255, 255, 255]));

        let filter = BoxFilterIntegral::new(1).unwrap();
        let result = filter.filter(&img).unwrap();

        // Center pixel should be averaged with neighbors
        let center = result.get_pixel(2, 2);
        assert!(center[0] > 100 && center[0] < 255);
    }

    #[test]
    fn test_box_filter_opsat_basic() {
        use image::Rgb;
        let mut img = ImageBuffer::from_pixel(5, 5, Rgb([100u8, 100, 100]));
        img.put_pixel(2, 2, Rgb([255, 255, 255]));

        let filter = BoxFilterOPSAT::new(1).unwrap();
        let result = filter.filter(&img).unwrap();

        // Center pixel should be averaged with neighbors
        let center = result.get_pixel(2, 2);
        assert!(center[0] > 100 && center[0] < 255);
    }

    #[test]
    fn test_both_filters_produce_same_result() {
        use image::Rgb;
        let mut img = ImageBuffer::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                let val = ((x + y) * 10) as u8;
                img.put_pixel(x, y, Rgb([val, val, val]));
            }
        }

        let integral_filter = BoxFilterIntegral::new(2).unwrap();
        let opsat_filter = BoxFilterOPSAT::new(2).unwrap();

        let result_integral = integral_filter.filter(&img).unwrap();
        let result_opsat = opsat_filter.filter(&img).unwrap();

        // Results should be identical
        for y in 0..10 {
            for x in 0..10 {
                let p1 = result_integral.get_pixel(x, y);
                let p2 = result_opsat.get_pixel(x, y);
                assert_eq!(p1, p2);
            }
        }
    }

    #[test]
    fn test_extension_trait() {
        use image::Rgb;
        let img = ImageBuffer::from_pixel(5, 5, Rgb([100u8, 100, 100]));

        // Test chaining
        let result = img
            .box_filter_integral(1)
            .unwrap()
            .box_filter_opsat(1)
            .unwrap();

        assert_eq!(result.dimensions(), (5, 5));
    }

    #[test]
    fn test_edge_handling() {
        use image::Luma;
        let img = ImageBuffer::from_pixel(5, 5, Luma([100u8]));

        let filter = BoxFilterIntegral::new(1).unwrap();
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
        let filter = BoxFilterIntegral::new(1).unwrap();
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
        let filter = BoxFilterIntegral::new(2).unwrap(); // radius 2 needs at least 5x5 image
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

        let filter = BoxFilterIntegral::new(1).unwrap();
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

        let filter = BoxFilterIntegral::new(1).unwrap();
        let result = filter.filter(&img).unwrap();

        // Center pixel should be averaged with neighbors
        let center = result.get_pixel(2, 2);
        assert!(center[0] > 0.5 && center[0] < 1.0);
    }

    #[test]
    fn test_performance_large_image_integral() {
        use image::Rgb;
        let img = ImageBuffer::from_pixel(500, 500, Rgb([100u8, 100, 100]));

        let start = std::time::Instant::now();
        let filter = BoxFilterIntegral::new(5).unwrap();
        let _result = filter.filter(&img).unwrap();
        let duration = start.elapsed();

        println!(
            "Box filter (500x500, radius=5) Integral took: {:?}",
            duration
        );
        // Should complete in reasonable time
        assert!(duration.as_millis() < 1000);
    }

    #[test]
    fn test_performance_large_image_opsat() {
        use image::Rgb;
        let img = ImageBuffer::from_pixel(500, 500, Rgb([100u8, 100, 100]));

        let start = std::time::Instant::now();
        let filter = BoxFilterOPSAT::new(5).unwrap();
        let _result = filter.filter(&img).unwrap();
        let duration = start.elapsed();

        println!("Box filter (500x500, radius=5) OPSAT took: {:?}", duration);
        // Should complete in reasonable time
        assert!(duration.as_millis() < 1000);
    }
}
