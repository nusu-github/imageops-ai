use image::{GenericImageView, ImageBuffer, Pixel, Primitive};
use imageproc::definitions::{Clamp, Image};

use crate::error::InterAreaError;

/// Element of the weight table for area interpolation.
#[derive(Debug, Clone, Copy)]
pub struct InterpolationWeight {
    /// Destination index
    pub destination_index: u32,
    /// Source index
    pub source_index: u32,
    /// Weight value
    pub weight: f32,
}

/// OpenCV INTER_AREA interpolation implementation.
pub struct InterAreaResize {
    /// New width
    pub new_width: u32,
    /// New height
    pub new_height: u32,
}

impl InterAreaResize {
    /// Create a new INTER_AREA resizer.
    pub const fn new(new_width: u32, new_height: u32) -> Result<Self, InterAreaError> {
        if new_width == 0 || new_height == 0 {
            return Err(InterAreaError::InvalidTargetDimensions {
                width: new_width,
                height: new_height,
            });
        }
        Ok(Self {
            new_width,
            new_height,
        })
    }
}

/// Compute resize area decimation table.
///
/// This function computes the weight table for area interpolation based on the
/// source size, destination size, and scale factor.
fn compute_interpolation_weights_impl(
    src_size: u32,
    dst_size: u32,
    scale: f32,
) -> Vec<InterpolationWeight> {
    let mut tab = Vec::new();

    for dx in 0..dst_size {
        let src_x_start = dx as f32 * scale;
        let src_x_end = src_x_start + scale;

        let src_x_start_int = (src_x_start.ceil() as u32).min(src_size);
        let src_x_end_int = (src_x_end.floor() as u32).min(src_size);

        let cell_width = if src_x_end - src_x_start != scale {
            // Handle boundary cases where the footprint extends beyond image bounds
            if src_x_start_int == 0 {
                src_x_end_int as f32
            } else if src_x_end_int == src_size {
                src_size as f32 - src_x_start
            } else {
                scale
            }
        } else {
            scale
        };

        // Left partial overlap
        if src_x_start_int > 0 && (src_x_start_int as f32 - src_x_start) > 1e-3 {
            let alpha = (src_x_start_int as f32 - src_x_start) / cell_width;
            tab.push(InterpolationWeight {
                destination_index: dx,
                source_index: src_x_start_int - 1,
                weight: alpha,
            });
        }

        // Full overlaps
        for sx in src_x_start_int..src_x_end_int {
            let alpha = 1.0 / cell_width;
            tab.push(InterpolationWeight {
                destination_index: dx,
                source_index: sx,
                weight: alpha,
            });
        }

        // Right partial overlap
        if src_x_end_int < src_size && (src_x_end - src_x_end_int as f32) > 1e-3 {
            let alpha = (src_x_end - src_x_end_int as f32) / cell_width;
            tab.push(InterpolationWeight {
                destination_index: dx,
                source_index: src_x_end_int,
                weight: alpha,
            });
        }
    }

    tab
}

/// Check if we can use the integer scale optimization.
fn can_use_integer_scale_impl(src_size: u32, dst_size: u32) -> bool {
    if dst_size >= src_size {
        return false;
    }

    let scale = src_size as f32 / dst_size as f32;
    let int_scale = scale.round() as u32;

    // Check if the scale is close to an integer
    (scale - int_scale as f32).abs() < f32::EPSILON && int_scale >= 2
}

/// Integer scale implementation for optimized performance.
fn resize_area_integer_scale_impl<I, P>(
    src: &I,
    dst_width: u32,
    dst_height: u32,
) -> Result<Image<P>, InterAreaError>
where
    I: GenericImageView<Pixel = P>,
    P: Pixel,
    P::Subpixel: Clamp<f32> + Into<f32> + Primitive,
{
    let (src_width, src_height) = src.dimensions();
    let scale_x = src_width / dst_width;
    let scale_y = src_height / dst_height;
    let area = (scale_x * scale_y) as f32;
    let inv_area = 1.0 / area;

    let result = ImageBuffer::from_fn(dst_width, dst_height, |dx, dy| {
        let mut pixel_sum = vec![0.0f32; P::CHANNEL_COUNT as usize];

        let start_x = dx * scale_x;
        let start_y = dy * scale_y;
        let end_x = start_x + scale_x;
        let end_y = start_y + scale_y;

        for sy in start_y..end_y {
            for sx in start_x..end_x {
                let pixel = src.get_pixel(sx, sy);
                let channels = pixel.channels();

                for c in 0..channels.len() {
                    pixel_sum[c] += channels[c].into();
                }
            }
        }

        let mut output_channels = vec![P::Subpixel::DEFAULT_MIN_VALUE; P::CHANNEL_COUNT as usize];
        for c in 0..pixel_sum.len() {
            output_channels[c] = P::Subpixel::clamp(pixel_sum[c] * inv_area);
        }

        *P::from_slice(&output_channels)
    });

    Ok(result)
}

/// Fractional scale implementation for arbitrary scale factors.
fn resize_area_fractional_scale_impl<I, P>(
    src: &I,
    dst_width: u32,
    dst_height: u32,
) -> Result<Image<P>, InterAreaError>
where
    I: GenericImageView<Pixel = P>,
    P: Pixel,
    P::Subpixel: Clamp<f32> + Into<f32> + Primitive,
{
    let (src_width, src_height) = src.dimensions();
    let scale_x = src_width as f32 / dst_width as f32;
    let scale_y = src_height as f32 / dst_height as f32;

    // Compute X and Y tables
    let x_weights = compute_interpolation_weights_impl(src_width, dst_width, scale_x);
    let y_weights = compute_interpolation_weights_impl(src_height, dst_height, scale_y);

    let channels = P::CHANNEL_COUNT as usize;
    let mut output = ImageBuffer::new(dst_width, dst_height);

    // Intermediate buffers
    let mut buf = vec![0.0f32; dst_width as usize * channels];
    let mut sum = vec![0.0f32; dst_width as usize * channels];

    let mut prev_dy = u32::MAX;

    for y_entry in &y_weights {
        let dy = y_entry.destination_index;
        let sy = y_entry.source_index;
        let beta = y_entry.weight;

        // Clear intermediate buffer
        buf.fill(0.0);

        // Horizontal pass
        for x_entry in &x_weights {
            let dx = x_entry.destination_index;
            let sx = x_entry.source_index;
            let alpha = x_entry.weight;

            let src_pixel = src.get_pixel(sx, sy);
            let src_channels = src_pixel.channels();

            for c in 0..channels {
                let idx = (dx as usize) * channels + c;
                buf[idx] += src_channels[c].into() * alpha;
            }
        }

        // Vertical accumulation
        for dx in 0..dst_width {
            for c in 0..channels {
                let idx = (dx as usize) * channels + c;
                sum[idx] += buf[idx] * beta;
            }
        }

        // Output when destination row changes
        if dy != prev_dy && prev_dy != u32::MAX {
            // Write out the accumulated row
            for dx in 0..dst_width {
                let mut pixel_channels = vec![P::Subpixel::DEFAULT_MIN_VALUE; channels];
                for c in 0..channels {
                    let idx = (dx as usize) * channels + c;
                    pixel_channels[c] = P::Subpixel::clamp(sum[idx]);
                }
                output.put_pixel(dx, prev_dy, *P::from_slice(&pixel_channels));
            }

            // Clear sum for next row
            sum.fill(0.0);
        }

        prev_dy = dy;
    }

    // Write out the last row
    if prev_dy != u32::MAX {
        for dx in 0..dst_width {
            let mut pixel_channels = vec![P::Subpixel::DEFAULT_MIN_VALUE; channels];
            for c in 0..channels {
                let idx = (dx as usize) * channels + c;
                pixel_channels[c] = P::Subpixel::clamp(sum[idx]);
            }
            output.put_pixel(dx, prev_dy, *P::from_slice(&pixel_channels));
        }
    }

    Ok(output)
}

impl InterAreaResize {
    /// Resize image using INTER_AREA interpolation.
    pub fn resize<I, P>(&self, src: &I) -> Result<Image<P>, InterAreaError>
    where
        I: GenericImageView<Pixel = P>,
        P: Pixel,
        P::Subpixel: Clamp<f32> + Into<f32> + Primitive,
    {
        let (src_width, src_height) = src.dimensions();

        if src_width == 0 || src_height == 0 {
            return Err(InterAreaError::EmptyImage {
                width: src_width,
                height: src_height,
            });
        }

        // Handle upscaling (use bilinear interpolation)
        if self.new_width > src_width || self.new_height > src_height {
            // For upscaling, INTER_AREA behaves like INTER_LINEAR
            // For simplicity, we'll return an error for now
            return Err(InterAreaError::UpscalingNotSupported {
                src_width,
                src_height,
                target_width: self.new_width,
                target_height: self.new_height,
            });
        }

        // Check if we can use the integer scale optimization
        if can_use_integer_scale_impl(src_width, self.new_width)
            && can_use_integer_scale_impl(src_height, self.new_height)
        {
            resize_area_integer_scale_impl(src, self.new_width, self.new_height)
        } else {
            resize_area_fractional_scale_impl(src, self.new_width, self.new_height)
        }
    }
}

/// Extension trait for ImageBuffer to provide INTER_AREA resize methods.
pub trait InterAreaResizeExt<P>
where
    P: Pixel,
{
    /// Resize image using INTER_AREA interpolation.
    fn resize_area(self, new_width: u32, new_height: u32) -> Result<Self, InterAreaError>
    where
        Self: Sized;

    /// Resize image using INTER_AREA interpolation in-place.
    fn resize_area_mut(
        &mut self,
        new_width: u32,
        new_height: u32,
    ) -> Result<&mut Self, InterAreaError>;
}

impl<P> InterAreaResizeExt<P> for Image<P>
where
    P: Pixel,
    P::Subpixel: Clamp<f32> + Into<f32> + Primitive,
{
    fn resize_area(self, new_width: u32, new_height: u32) -> Result<Self, InterAreaError> {
        let resizer = InterAreaResize::new(new_width, new_height)?;
        resizer.resize(&self)
    }

    /// Hidden _mut variant that is not available for this operation.
    #[doc(hidden)]
    fn resize_area_mut(
        &mut self,
        _new_width: u32,
        _new_height: u32,
    ) -> Result<&mut Self, InterAreaError> {
        unimplemented!(
            "resize_area_mut is not available because the operation requires creating a new image with different dimensions"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn can_use_integer_scale_impl_with_valid_cases_identifies_optimizable_scales() {
        assert!(can_use_integer_scale_impl(100, 50)); // 2x downscale
        assert!(can_use_integer_scale_impl(150, 50)); // 3x downscale
        assert!(!can_use_integer_scale_impl(100, 67)); // 1.5x downscale (not integer)
        assert!(!can_use_integer_scale_impl(50, 100)); // upscale
    }

    #[test]
    fn compute_interpolation_weights_impl_with_valid_input_produces_normalized_weights() {
        let tab = compute_interpolation_weights_impl(4, 2, 2.0);
        assert!(!tab.is_empty());

        // Check that weights sum to 1.0 for each destination pixel
        let mut weights_sum = [0.0; 2];
        for entry in &tab {
            weights_sum[entry.destination_index as usize] += entry.weight;
        }

        for sum in weights_sum.iter() {
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn resize_area_integer_scale_impl_with_valid_input_preserves_image_structure() {
        let src = ImageBuffer::from_fn(4, 4, |x, y| Rgb([((x + y) * 50) as u8, 100, 150]));

        let result = resize_area_integer_scale_impl(&src, 2, 2).unwrap();
        assert_eq!(result.dimensions(), (2, 2));

        // Check that result is not empty
        assert!(result.get_pixel(0, 0)[0] > 0);
    }

    #[test]
    fn resize_area_fractional_scale_impl_with_valid_input_preserves_image_structure() {
        let src = ImageBuffer::from_fn(6, 6, |x, y| Rgb([((x + y) * 20) as u8, 100, 150]));

        let result = resize_area_fractional_scale_impl(&src, 4, 4).unwrap();
        assert_eq!(result.dimensions(), (4, 4));

        // Check that result is not empty
        assert!(result.get_pixel(0, 0)[0] > 0);
    }

    #[test]
    fn resize_with_valid_input_produces_correct_dimensions() {
        let src = ImageBuffer::from_fn(8, 8, |x, y| Rgb([((x + y) * 16) as u8, 100, 150]));

        let resizer = InterAreaResize::new(4, 4).unwrap();
        let result = resizer.resize(&src).unwrap();

        assert_eq!(result.dimensions(), (4, 4));

        // Check that result is not empty
        assert!(result.get_pixel(0, 0)[0] > 0);
    }

    #[test]
    fn resize_area_ext_with_valid_input_produces_correct_dimensions() {
        let src = ImageBuffer::from_fn(8, 8, |x, y| Rgb([((x + y) * 16) as u8, 100, 150]));

        let result = src.resize_area(4, 4).unwrap();
        assert_eq!(result.dimensions(), (4, 4));
    }

    #[test]
    fn resize_area_with_chained_resizes_produces_correct_dimensions() {
        // Create a gradient image
        let src = ImageBuffer::from_fn(100, 100, |x, y| {
            Rgb([((x + y) as f32 / 200.0 * 255.0) as u8, 128, 192])
        });

        println!("Source image size: {:?}", src.dimensions());

        // Test resizing using the extension trait
        let resized = src.resize_area(50, 50).unwrap();
        assert_eq!(resized.dimensions(), (50, 50));

        // Test resizing using the struct directly
        let resizer = InterAreaResize::new(25, 25).unwrap();
        let resized2 = resizer.resize(&resized).unwrap();
        assert_eq!(resized2.dimensions(), (25, 25));

        // Verify the pixel values are reasonable
        let pixel = resized2.get_pixel(12, 12);
        assert!(pixel[0] > 0 && pixel[0] < 255);
        assert_eq!(pixel[1], 128);
        assert_eq!(pixel[2], 192);
    }
}
