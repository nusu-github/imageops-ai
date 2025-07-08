use crate::error::ResizeAreaError;
use image::{ImageBuffer, Pixel, Primitive};
use imageproc::definitions::{Clamp, Image};

/// Element of the weight table for area interpolation
#[derive(Debug, Clone, Copy)]
pub struct DecimateAlpha {
    /// Destination index
    pub di: u32,
    /// Source index
    pub si: u32,
    /// Alpha value (weight)
    pub alpha: f32,
}

/// OpenCV INTER_AREA interpolation implementation
pub struct InterAreaResize {
    /// New width
    pub new_width: u32,
    /// New height
    pub new_height: u32,
}

impl InterAreaResize {
    /// Create a new INTER_AREA resizer
    pub const fn new(new_width: u32, new_height: u32) -> Result<Self, ResizeAreaError> {
        if new_width == 0 || new_height == 0 {
            return Err(ResizeAreaError::InvalidTargetDimensions {
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

/// Compute resize area decimation table
///
/// This function computes the weight table for area interpolation based on the
/// source size, destination size, and scale factor.
fn compute_resize_area_tab(ssize: u32, dsize: u32, scale: f32) -> Vec<DecimateAlpha> {
    let mut tab = Vec::new();

    for dx in 0..dsize {
        let fsx1 = dx as f32 * scale;
        let fsx2 = fsx1 + scale;

        let sx1 = (fsx1.ceil() as u32).min(ssize);
        let sx2 = (fsx2.floor() as u32).min(ssize);

        let cell_width = if fsx2 - fsx1 != scale {
            // Handle boundary cases where the footprint extends beyond image bounds
            if sx1 == 0 {
                sx2 as f32
            } else if sx2 == ssize {
                ssize as f32 - fsx1
            } else {
                scale
            }
        } else {
            scale
        };

        // Left partial overlap
        if sx1 > 0 && (sx1 as f32 - fsx1) > 1e-3 {
            let alpha = (sx1 as f32 - fsx1) / cell_width;
            tab.push(DecimateAlpha {
                di: dx,
                si: sx1 - 1,
                alpha,
            });
        }

        // Full overlaps
        for sx in sx1..sx2 {
            let alpha = 1.0 / cell_width;
            tab.push(DecimateAlpha {
                di: dx,
                si: sx,
                alpha,
            });
        }

        // Right partial overlap
        if sx2 < ssize && (fsx2 - sx2 as f32) > 1e-3 {
            let alpha = (fsx2 - sx2 as f32) / cell_width;
            tab.push(DecimateAlpha {
                di: dx,
                si: sx2,
                alpha,
            });
        }
    }

    tab
}

/// Check if we can use the fast path (integer scale)
fn is_area_fast(src_size: u32, dst_size: u32) -> bool {
    if dst_size >= src_size {
        return false;
    }

    let scale = src_size as f32 / dst_size as f32;
    let int_scale = scale.round() as u32;

    // Check if the scale is close to an integer
    (scale - int_scale as f32).abs() < f32::EPSILON && int_scale >= 2
}

/// Fast path implementation for integer scale factors
fn resize_area_fast<P>(
    src: &Image<P>,
    dst_width: u32,
    dst_height: u32,
) -> Result<Image<P>, ResizeAreaError>
where
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

/// General path implementation for non-integer scale factors
fn resize_area_general<P>(
    src: &Image<P>,
    dst_width: u32,
    dst_height: u32,
) -> Result<Image<P>, ResizeAreaError>
where
    P: Pixel,
    P::Subpixel: Clamp<f32> + Into<f32> + Primitive,
{
    let (src_width, src_height) = src.dimensions();
    let scale_x = src_width as f32 / dst_width as f32;
    let scale_y = src_height as f32 / dst_height as f32;

    // Compute X and Y tables
    let xtab = compute_resize_area_tab(src_width, dst_width, scale_x);
    let ytab = compute_resize_area_tab(src_height, dst_height, scale_y);

    let channels = P::CHANNEL_COUNT as usize;
    let mut output = ImageBuffer::new(dst_width, dst_height);

    // Intermediate buffers
    let mut buf = vec![0.0f32; dst_width as usize * channels];
    let mut sum = vec![0.0f32; dst_width as usize * channels];

    let mut prev_dy = u32::MAX;

    for y_entry in &ytab {
        let dy = y_entry.di;
        let sy = y_entry.si;
        let beta = y_entry.alpha;

        // Clear intermediate buffer
        buf.fill(0.0);

        // Horizontal pass
        for x_entry in &xtab {
            let dx = x_entry.di;
            let sx = x_entry.si;
            let alpha = x_entry.alpha;

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
    /// Resize image using INTER_AREA interpolation
    pub fn resize<P>(&self, src: &Image<P>) -> Result<Image<P>, ResizeAreaError>
    where
        P: Pixel,
        P::Subpixel: Clamp<f32> + Into<f32> + Primitive,
    {
        let (src_width, src_height) = src.dimensions();

        if src_width == 0 || src_height == 0 {
            return Err(ResizeAreaError::EmptyImage {
                width: src_width,
                height: src_height,
            });
        }

        // Handle upscaling (use bilinear interpolation)
        if self.new_width > src_width || self.new_height > src_height {
            // For upscaling, INTER_AREA behaves like INTER_LINEAR
            // For simplicity, we'll return an error for now
            return Err(ResizeAreaError::UpscalingNotSupported {
                src_width,
                src_height,
                target_width: self.new_width,
                target_height: self.new_height,
            });
        }

        // Check if we can use the fast path
        if is_area_fast(src_width, self.new_width) && is_area_fast(src_height, self.new_height) {
            resize_area_fast(src, self.new_width, self.new_height)
        } else {
            resize_area_general(src, self.new_width, self.new_height)
        }
    }
}

/// Extension trait for ImageBuffer to provide INTER_AREA resize methods
pub trait InterAreaExt<P>
where
    P: Pixel,
{
    /// Resize image using INTER_AREA interpolation
    fn resize_area(self, new_width: u32, new_height: u32) -> Result<Self, ResizeAreaError>
    where
        Self: Sized;

    /// Resize image using INTER_AREA interpolation in-place
    fn resize_area_mut(
        &mut self,
        new_width: u32,
        new_height: u32,
    ) -> Result<&mut Self, ResizeAreaError>;
}

impl<P> InterAreaExt<P> for Image<P>
where
    P: Pixel,
    P::Subpixel: Clamp<f32> + Into<f32> + Primitive,
{
    fn resize_area(self, new_width: u32, new_height: u32) -> Result<Self, ResizeAreaError> {
        let resizer = InterAreaResize::new(new_width, new_height)?;
        resizer.resize(&self)
    }

    /// Hidden _mut variant that is not available for this operation
    #[doc(hidden)]
    fn resize_area_mut(
        &mut self,
        _new_width: u32,
        _new_height: u32,
    ) -> Result<&mut Self, ResizeAreaError> {
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
    fn test_is_area_fast() {
        assert!(is_area_fast(100, 50)); // 2x downscale
        assert!(is_area_fast(150, 50)); // 3x downscale
        assert!(!is_area_fast(100, 67)); // 1.5x downscale (not integer)
        assert!(!is_area_fast(50, 100)); // upscale
    }

    #[test]
    fn test_compute_resize_area_tab() {
        let tab = compute_resize_area_tab(4, 2, 2.0);
        assert!(!tab.is_empty());

        // Check that weights sum to 1.0 for each destination pixel
        let mut weights_sum = [0.0; 2];
        for entry in &tab {
            weights_sum[entry.di as usize] += entry.alpha;
        }

        for sum in weights_sum.iter() {
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_resize_area_fast() {
        let src = ImageBuffer::from_fn(4, 4, |x, y| Rgb([((x + y) * 50) as u8, 100, 150]));

        let result = resize_area_fast(&src, 2, 2).unwrap();
        assert_eq!(result.dimensions(), (2, 2));

        // Check that result is not empty
        assert!(result.get_pixel(0, 0)[0] > 0);
    }

    #[test]
    fn test_resize_area_general() {
        let src = ImageBuffer::from_fn(6, 6, |x, y| Rgb([((x + y) * 20) as u8, 100, 150]));

        let result = resize_area_general(&src, 4, 4).unwrap();
        assert_eq!(result.dimensions(), (4, 4));

        // Check that result is not empty
        assert!(result.get_pixel(0, 0)[0] > 0);
    }

    #[test]
    fn test_inter_area_resize() {
        let src = ImageBuffer::from_fn(8, 8, |x, y| Rgb([((x + y) * 16) as u8, 100, 150]));

        let resizer = InterAreaResize::new(4, 4).unwrap();
        let result = resizer.resize(&src).unwrap();

        assert_eq!(result.dimensions(), (4, 4));

        // Check that result is not empty
        assert!(result.get_pixel(0, 0)[0] > 0);
    }

    #[test]
    fn test_extension_trait() {
        let src = ImageBuffer::from_fn(8, 8, |x, y| Rgb([((x + y) * 16) as u8, 100, 150]));

        let result = src.resize_area(4, 4).unwrap();
        assert_eq!(result.dimensions(), (4, 4));
    }

    #[test]
    fn test_comprehensive_workflow() {
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
