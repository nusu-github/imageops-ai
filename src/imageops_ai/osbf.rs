use crate::error::OSBFilterError;
use image::{ImageBuffer, Pixel, Primitive};
use imageproc::definitions::{Clamp, Image};

/// Trait for One-Sided Box Filter implementations
pub trait OSBFilter<P>
where
    P: Pixel,
{
    /// Apply OSBF to the image
    fn filter(&self, image: &Image<P>, iterations: u32) -> Result<Image<P>, OSBFilterError>;
}

/// One-Sided Box Filter (OSBF)
///
/// This filter selects the mean value from 8 adjacent regions (4 quarter windows and 4 half windows)
/// that is closest to the current pixel value. It's effective for edge-preserving smoothing.
pub struct OneSidedBoxFilter {
    radius: u32,
}

impl OneSidedBoxFilter {
    /// Create a new One-Sided Box Filter
    pub const fn new(radius: u32) -> Result<Self, OSBFilterError> {
        if radius == 0 {
            return Err(OSBFilterError::InvalidRadius { radius });
        }
        Ok(Self { radius })
    }

    /// Get the kernel size (2 * radius + 1)
    #[inline]
    pub const fn kernel_size(&self) -> u32 {
        2 * self.radius + 1
    }
}

/// Optimized helper function to pad image with edge replication
fn pad_image<P, S>(image: &ImageBuffer<P, Vec<S>>, pad_size: u32) -> ImageBuffer<P, Vec<S>>
where
    P: Pixel<Subpixel = S>,
    S: Primitive,
{
    let (width, height) = image.dimensions();
    let new_width = width + 2 * pad_size;
    let new_height = height + 2 * pad_size;

    // Optimized approach: use from_fn but with reduced condition branches
    ImageBuffer::from_fn(new_width, new_height, |x, y| {
        let orig_x = x.saturating_sub(pad_size).min(width - 1);
        let orig_y = y.saturating_sub(pad_size).min(height - 1);
        *image.get_pixel(orig_x, orig_y)
    })
}

/// Optimized box sum calculation function
#[inline]
fn box_sum(
    integral: &[f32],
    integral_width: usize,
    y1: usize,
    x1: usize,
    y2: usize,
    x2: usize,
) -> f32 {
    integral[y2 * integral_width + x2]
        - integral[y1 * integral_width + x2]
        - integral[y2 * integral_width + x1]
        + integral[y1 * integral_width + x1]
}

/// Pre-computed region coordinates for OSBF
#[derive(Debug, Clone)]
struct OSBFRegions {
    /// Quarter window coordinates: (y1, x1, y2, x2)
    quarters: [(usize, usize, usize, usize); 4],
    /// Half window coordinates: (y1, x1, y2, x2)
    halves: [(usize, usize, usize, usize); 4],
    quarter_area: f32,
    half_area: f32,
}

impl OSBFRegions {
    const fn new(py: usize, px: usize, r: usize) -> Self {
        let py_sub_r = py.saturating_sub(r);
        let px_sub_r = px.saturating_sub(r);

        let quarters = [
            (py, px_sub_r, py + r + 1, px + 1),   // q1
            (py, px, py + r + 1, px + r + 1),     // q2
            (py_sub_r, px, py + 1, px + r + 1),   // q3
            (py_sub_r, px_sub_r, py + 1, px + 1), // q4
        ];

        let halves = [
            (py_sub_r, px_sub_r, py + r + 1, px + 1), // h1
            (py_sub_r, px, py + r + 1, px + r + 1),   // h2
            (py, px_sub_r, py + r + 1, px + r + 1),   // h3
            (py_sub_r, px_sub_r, py + 1, px + r + 1), // h4
        ];

        let quarter_area = ((r + 1) * (r + 1)) as f32;
        let half_area = ((r + 1) * (2 * r + 1)) as f32;

        Self {
            quarters,
            halves,
            quarter_area,
            half_area,
        }
    }
}

/// Perform one OSBF iteration on the entire image
fn osbf_iteration<P>(image: &Image<P>, radius: u32) -> Result<Image<P>, OSBFilterError>
where
    P: Pixel,
    P::Subpixel: Clamp<f32> + Into<f32> + Primitive,
{
    let (width, height) = image.dimensions();
    let channels = P::CHANNEL_COUNT as usize;

    // Pad the image
    let padded = pad_image(image, 2);
    let (pad_width, pad_height) = padded.dimensions();
    let pad_size = 2;

    // Create integral images for each channel using optimized 1D layout
    let integral_width = (pad_width + 1) as usize;
    let integral_height = (pad_height + 1) as usize;
    let integral_size = integral_width * integral_height;
    let mut channel_integrals = vec![0.0f32; channels * integral_size];

    // Build integral images row by row for better cache efficiency
    for y in 0..pad_height {
        for x in 0..pad_width {
            let pixel = padded.get_pixel(x, y);
            let pixel_channels = pixel.channels();

            let current_idx = ((y + 1) as usize) * integral_width + ((x + 1) as usize);
            let top_idx = (y as usize) * integral_width + ((x + 1) as usize);
            let left_idx = ((y + 1) as usize) * integral_width + (x as usize);
            let diag_idx = (y as usize) * integral_width + (x as usize);

            for c in 0..channels {
                let val: f32 = pixel_channels[c].into();
                let base_offset = c * integral_size;

                channel_integrals[base_offset + current_idx] = val
                    + channel_integrals[base_offset + top_idx]
                    + channel_integrals[base_offset + left_idx]
                    - channel_integrals[base_offset + diag_idx];
            }
        }
    }

    // Pre-compute radius and area constants
    let r = radius as usize;

    // Reusable pixel data buffer
    let mut pixel_data = Vec::with_capacity(channels);

    // Process image using optimized pixel mapping
    let output = ImageBuffer::from_fn(width, height, |x, y| {
        // Coordinates in padded space
        let py = (y + pad_size as u32) as usize;
        let px = (x + pad_size as u32) as usize;

        // Get current pixel value
        let current_pixel = image.get_pixel(x, y);
        let current_channels = current_pixel.channels();

        // Pre-compute regions for this pixel
        let regions = OSBFRegions::new(py, px, r);

        pixel_data.clear();

        for c in 0..channels {
            let current_val: f32 = current_channels[c].into();
            let mut min_diff = f32::INFINITY;
            let mut best_val = current_val;

            let integral_base = &channel_integrals[c * integral_size..];

            // Calculate quarter windows
            for &(y1, x1, y2, x2) in &regions.quarters {
                let val =
                    box_sum(integral_base, integral_width, y1, x1, y2, x2) / regions.quarter_area;
                let diff = (val - current_val).abs();
                if diff < min_diff {
                    min_diff = diff;
                    best_val = val;
                }
            }

            // Calculate half windows
            for &(y1, x1, y2, x2) in &regions.halves {
                let val =
                    box_sum(integral_base, integral_width, y1, x1, y2, x2) / regions.half_area;
                let diff = (val - current_val).abs();
                if diff < min_diff {
                    min_diff = diff;
                    best_val = val;
                }
            }

            pixel_data.push(P::Subpixel::clamp(best_val));
        }

        *P::from_slice(&pixel_data)
    });

    Ok(output)
}

impl<P> OSBFilter<P> for OneSidedBoxFilter
where
    P: Pixel,
    P::Subpixel: Clamp<f32> + Into<f32> + Primitive,
{
    fn filter(&self, image: &Image<P>, iterations: u32) -> Result<Image<P>, OSBFilterError> {
        let (width, height) = image.dimensions();

        if width == 0 || height == 0 {
            return Err(OSBFilterError::EmptyImage { width, height });
        }

        if iterations == 0 {
            return Err(OSBFilterError::InvalidIterations { iterations });
        }

        // Check if image is large enough for the filter radius
        let min_dimension = width.min(height);
        if min_dimension < 2 * self.radius + 1 {
            return Err(OSBFilterError::ImageTooSmall {
                width,
                height,
                radius: self.radius,
            });
        }

        let mut result = image.clone();

        // Apply filter iterations
        for _ in 0..iterations {
            result = osbf_iteration(&result, self.radius)?;
        }

        Ok(result)
    }
}

/// Extension trait for ImageBuffer to provide fluent OSBF methods
pub trait OSBFilterExt<P>
where
    P: Pixel,
{
    /// Apply One-Sided Box Filter
    fn osbf(self, radius: u32, iterations: u32) -> Result<Self, OSBFilterError>
    where
        Self: Sized;

    /// Apply One-Sided Box Filter in-place (not available - requires reallocation)
    #[doc(hidden)]
    fn osbf_mut(&mut self, radius: u32, iterations: u32) -> Result<&mut Self, OSBFilterError>;
}

impl<P> OSBFilterExt<P> for Image<P>
where
    P: Pixel,
    P::Subpixel: Clamp<f32> + Into<f32> + Primitive,
{
    fn osbf(self, radius: u32, iterations: u32) -> Result<Self, OSBFilterError> {
        let filter = OneSidedBoxFilter::new(radius)?;
        filter.filter(&self, iterations)
    }

    #[doc(hidden)]
    fn osbf_mut(&mut self, _radius: u32, _iterations: u32) -> Result<&mut Self, OSBFilterError> {
        unimplemented!(
            "osbf_mut is not available because the operation requires additional memory allocations equivalent to the owning version"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma, Rgb};

    #[test]
    fn test_osbf_basic() {
        let mut img = ImageBuffer::from_pixel(5, 5, Luma([100u8]));
        img.put_pixel(2, 2, Luma([255]));

        let filter = OneSidedBoxFilter::new(1).unwrap();
        let result = filter.filter(&img, 1).unwrap();

        // Center pixel should be smoothed
        let center = result.get_pixel(2, 2);
        assert!(center[0] > 100 && center[0] < 255);
    }

    #[test]
    fn test_osbf_rgb() {
        let mut img = ImageBuffer::from_pixel(5, 5, Rgb([100u8, 100, 100]));
        img.put_pixel(2, 2, Rgb([255, 255, 255]));

        let filter = OneSidedBoxFilter::new(1).unwrap();
        let result = filter.filter(&img, 1).unwrap();

        let center = result.get_pixel(2, 2);
        assert!(center[0] > 100 && center[0] < 255);
        assert!(center[1] > 100 && center[1] < 255);
        assert!(center[2] > 100 && center[2] < 255);
    }

    #[test]
    fn test_multiple_iterations() {
        let mut img = ImageBuffer::from_pixel(7, 7, Luma([100u8]));
        img.put_pixel(3, 3, Luma([255]));

        let filter = OneSidedBoxFilter::new(1).unwrap();
        let result1 = filter.filter(&img, 1).unwrap();
        let result5 = filter.filter(&img, 5).unwrap();

        // More iterations should produce more smoothing
        let center1 = result1.get_pixel(3, 3)[0];
        let center5 = result5.get_pixel(3, 3)[0];
        assert!(center5 < center1);
    }

    #[test]
    fn test_edge_preservation() {
        // Create an image with a sharp edge
        let mut img = ImageBuffer::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                if x < 5 {
                    img.put_pixel(x, y, Luma([50u8]));
                } else {
                    img.put_pixel(x, y, Luma([200u8]));
                }
            }
        }

        let filter = OneSidedBoxFilter::new(2).unwrap();
        let result = filter.filter(&img, 3).unwrap();

        // Edge should be somewhat preserved
        let left_side = result.get_pixel(2, 5)[0];
        let right_side = result.get_pixel(7, 5)[0];
        assert!(left_side < 100);
        assert!(right_side > 150);
    }

    #[test]
    fn test_extension_trait() {
        let img = ImageBuffer::from_pixel(5, 5, Rgb([100u8, 100, 100]));

        // Test chaining
        let result = img.osbf(1, 2).unwrap();
        assert_eq!(result.dimensions(), (5, 5));
    }

    #[test]
    fn test_invalid_radius() {
        let result = OneSidedBoxFilter::new(0);
        assert!(matches!(
            result,
            Err(OSBFilterError::InvalidRadius { radius: 0 })
        ));
    }

    #[test]
    fn test_invalid_iterations() {
        let img = ImageBuffer::from_pixel(5, 5, Luma([100u8]));
        let filter = OneSidedBoxFilter::new(1).unwrap();
        let result = filter.filter(&img, 0);
        assert!(matches!(
            result,
            Err(OSBFilterError::InvalidIterations { iterations: 0 })
        ));
    }

    #[test]
    fn test_empty_image() {
        let img: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(0, 0);
        let filter = OneSidedBoxFilter::new(1).unwrap();
        let result = filter.filter(&img, 1);
        assert!(matches!(
            result,
            Err(OSBFilterError::EmptyImage {
                width: 0,
                height: 0
            })
        ));
    }

    #[test]
    fn test_image_too_small() {
        let img = ImageBuffer::from_pixel(2, 2, Luma([100u8]));
        let filter = OneSidedBoxFilter::new(2).unwrap();
        let result = filter.filter(&img, 1);
        assert!(matches!(
            result,
            Err(OSBFilterError::ImageTooSmall {
                width: 2,
                height: 2,
                radius: 2
            })
        ));
    }

    #[test]
    fn test_generic_types_u16() {
        let mut img = ImageBuffer::from_pixel(5, 5, Luma([1000u16]));
        img.put_pixel(2, 2, Luma([5000]));

        let filter = OneSidedBoxFilter::new(1).unwrap();
        let result = filter.filter(&img, 1).unwrap();

        let center = result.get_pixel(2, 2);
        assert!(center[0] > 1000 && center[0] < 5000);
    }

    #[test]
    fn test_generic_types_f32() {
        let mut img = ImageBuffer::from_pixel(5, 5, Luma([0.5f32]));
        img.put_pixel(2, 2, Luma([1.0]));

        let filter = OneSidedBoxFilter::new(1).unwrap();
        let result = filter.filter(&img, 1).unwrap();

        let center = result.get_pixel(2, 2);
        assert!(center[0] > 0.5 && center[0] < 1.0);
    }
}
