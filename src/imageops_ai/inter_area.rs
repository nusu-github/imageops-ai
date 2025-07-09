use image::{Pixel, Primitive};
use imageproc::definitions::{Clamp, Image};
use ndarray::prelude::*;

use crate::{
    error::ResizeAreaError,
    utils::{array3_to_image, image_to_array3},
};

/// Weight information for area interpolation
#[derive(Debug, Clone, Copy)]
pub struct AreaWeight {
    /// Source pixel range for this destination pixel
    #[allow(dead_code)]
    pub src_start: f32,
    #[allow(dead_code)]
    pub src_end: f32,
    /// Weights for each source pixel contributing to this destination pixel
    pub weights: WeightSegments,
}

/// Segments of weights for a single destination pixel
#[derive(Debug, Clone, Copy)]
pub struct WeightSegments {
    /// Left partial overlap weight and index
    pub left: Option<(f32, u32)>,
    /// Full overlap indices (start, end) with their weight
    pub full_range: Option<(u32, u32, f32)>,
    /// Right partial overlap weight and index
    pub right: Option<(f32, u32)>,
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

/// Compute area weights for resize operation
///
/// This function computes the weight segments for area interpolation.
/// Each destination pixel gets weight contributions from source pixels it overlaps.
fn compute_area_weights(src_size: u32, dst_size: u32) -> Vec<AreaWeight> {
    let scale = src_size as f32 / dst_size as f32;

    (0..dst_size)
        .map(|dst_idx| {
            let src_start = dst_idx as f32 * scale;
            let src_end = src_start + scale;

            let weights = compute_weight_segments(src_start, src_end, src_size, scale);

            AreaWeight {
                src_start,
                src_end,
                weights,
            }
        })
        .collect()
}

/// Compute weight segments for a single destination pixel
fn compute_weight_segments(
    src_start: f32,
    src_end: f32,
    src_size: u32,
    scale: f32,
) -> WeightSegments {
    let sx1 = (src_start.ceil() as u32).min(src_size);
    let sx2 = (src_end.floor() as u32).min(src_size);

    let cell_width = calculate_cell_width(src_start, src_end, sx1, sx2, src_size, scale);
    let inv_width = 1.0 / cell_width;

    WeightSegments {
        left: compute_left_weight(src_start, sx1, inv_width),
        full_range: compute_full_range(sx1, sx2, inv_width),
        right: compute_right_weight(src_end, sx2, src_size, inv_width),
    }
}

/// Calculate the effective cell width for weight normalization
#[inline]
fn calculate_cell_width(
    src_start: f32,
    src_end: f32,
    sx1: u32,
    sx2: u32,
    src_size: u32,
    scale: f32,
) -> f32 {
    if (src_end - src_start - scale).abs() > f32::EPSILON {
        if sx1 == 0 {
            sx2 as f32
        } else if sx2 == src_size {
            src_size as f32 - src_start
        } else {
            scale
        }
    } else {
        scale
    }
}

/// Compute left partial overlap weight
#[inline]
fn compute_left_weight(src_start: f32, sx1: u32, inv_width: f32) -> Option<(f32, u32)> {
    if sx1 > 0 {
        let overlap = sx1 as f32 - src_start;
        if overlap > 1e-3 {
            return Some((overlap * inv_width, sx1 - 1));
        }
    }
    None
}

/// Compute full pixel range with weight
#[inline]
const fn compute_full_range(sx1: u32, sx2: u32, inv_width: f32) -> Option<(u32, u32, f32)> {
    if sx2 > sx1 {
        Some((sx1, sx2, inv_width))
    } else {
        None
    }
}

/// Compute right partial overlap weight
#[inline]
fn compute_right_weight(
    src_end: f32,
    sx2: u32,
    src_size: u32,
    inv_width: f32,
) -> Option<(f32, u32)> {
    if sx2 < src_size {
        let overlap = src_end - sx2 as f32;
        if overlap > 1e-3 {
            return Some((overlap * inv_width, sx2));
        }
    }
    None
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

/// Fast path implementation for integer scale factors using ndarray
fn resize_area_fast<P, S>(
    src_array: &ArrayView3<f32>,
    dst_width: u32,
    dst_height: u32,
) -> Result<Array3<f32>, ResizeAreaError>
where
    P: Pixel<Subpixel = S>,
    S: Primitive,
{
    let (src_height, src_width, channels) = src_array.dim();
    let scale_x = src_width / dst_width as usize;
    let scale_y = src_height / dst_height as usize;
    let inv_area = 1.0 / (scale_x * scale_y) as f32;

    // Create output array
    let mut output = Array3::<f32>::zeros((dst_height as usize, dst_width as usize, channels));

    // Process each destination pixel
    for dy in 0..dst_height as usize {
        for dx in 0..dst_width as usize {
            let sy_start = dy * scale_y;
            let sx_start = dx * scale_x;

            // Extract the source block and compute mean
            let src_block = src_array.slice(s![
                sy_start..sy_start + scale_y,
                sx_start..sx_start + scale_x,
                ..
            ]);

            // Sum all values in the block and normalize
            let sum = src_block.sum_axis(Axis(0)).sum_axis(Axis(0));
            output.slice_mut(s![dy, dx, ..]).assign(&(&sum * inv_area));
        }
    }

    Ok(output)
}

/// General path implementation for non-integer scale factors using ndarray
fn resize_area_general<P, S>(
    src_array: &ArrayView3<f32>,
    dst_width: u32,
    dst_height: u32,
) -> Result<Array3<f32>, ResizeAreaError>
where
    P: Pixel<Subpixel = S>,
    S: Primitive,
{
    let (src_height, src_width, channels) = src_array.dim();

    // Compute weight tables
    let x_weights = compute_area_weights(src_width as u32, dst_width);
    let y_weights = compute_area_weights(src_height as u32, dst_height);

    // Create output array
    let mut output = Array3::<f32>::zeros((dst_height as usize, dst_width as usize, channels));

    // Process each destination pixel
    for (dst_y, y_weight) in y_weights.iter().enumerate() {
        for (dst_x, x_weight) in x_weights.iter().enumerate() {
            // Accumulate weighted sum for this destination pixel
            let mut pixel_sum = Array1::<f32>::zeros(channels);

            // Apply y weights
            accumulate_weighted_pixels(
                src_array,
                &y_weight.weights,
                &x_weight.weights,
                &mut pixel_sum,
            );

            // Assign to output
            output.slice_mut(s![dst_y, dst_x, ..]).assign(&pixel_sum);
        }
    }

    Ok(output)
}

/// Accumulate weighted pixels from source based on weight segments
fn accumulate_weighted_pixels(
    src_array: &ArrayView3<f32>,
    y_weights: &WeightSegments,
    x_weights: &WeightSegments,
    pixel_sum: &mut Array1<f32>,
) {
    // Helper to process a single y index with weight
    let mut process_y = |y_idx: u32, y_weight: f32| {
        let src_row = src_array.index_axis(Axis(0), y_idx as usize);

        // Process x weights for this row
        if let Some((x_weight, x_idx)) = x_weights.left {
            let weighted_pixel =
                &src_row.index_axis(Axis(0), x_idx as usize) * (y_weight * x_weight);
            *pixel_sum += &weighted_pixel;
        }

        if let Some((x_start, x_end, x_weight)) = x_weights.full_range {
            for x_idx in x_start..x_end {
                let weighted_pixel =
                    &src_row.index_axis(Axis(0), x_idx as usize) * (y_weight * x_weight);
                *pixel_sum += &weighted_pixel;
            }
        }

        if let Some((x_weight, x_idx)) = x_weights.right {
            let weighted_pixel =
                &src_row.index_axis(Axis(0), x_idx as usize) * (y_weight * x_weight);
            *pixel_sum += &weighted_pixel;
        }
    };

    // Process y weights
    if let Some((weight, idx)) = y_weights.left {
        process_y(idx, weight);
    }

    if let Some((start, end, weight)) = y_weights.full_range {
        for idx in start..end {
            process_y(idx, weight);
        }
    }

    if let Some((weight, idx)) = y_weights.right {
        process_y(idx, weight);
    }
}

impl InterAreaResize {
    /// Resize image using INTER_AREA interpolation
    pub fn resize<P>(&self, src: &Image<P>) -> Result<Image<P>, ResizeAreaError>
    where
        P: Pixel,
        P::Subpixel: Clamp<f32> + Primitive,
        f32: From<P::Subpixel>,
    {
        let (src_width, src_height) = src.dimensions();

        validate_dimensions(src_width, src_height, self.new_width, self.new_height)?;

        // Convert to ndarray for processing
        let src_array = image_to_array3(src).mapv(f32::from);

        // Choose appropriate algorithm
        let result_array = if is_area_fast(src_width, self.new_width)
            && is_area_fast(src_height, self.new_height)
        {
            resize_area_fast::<P, P::Subpixel>(&src_array.view(), self.new_width, self.new_height)?
        } else {
            resize_area_general::<P, P::Subpixel>(
                &src_array.view(),
                self.new_width,
                self.new_height,
            )?
        };

        // Convert back to image
        Ok(array3_to_image(
            &result_array.mapv(P::Subpixel::clamp).view(),
        ))
    }
}

/// Validate input dimensions for resize operation
const fn validate_dimensions(
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
) -> Result<(), ResizeAreaError> {
    if src_width == 0 || src_height == 0 {
        return Err(ResizeAreaError::EmptyImage {
            width: src_width,
            height: src_height,
        });
    }

    if dst_width > src_width || dst_height > src_height {
        return Err(ResizeAreaError::UpscalingNotSupported {
            src_width,
            src_height,
            target_width: dst_width,
            target_height: dst_height,
        });
    }

    Ok(())
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
    P::Subpixel: Clamp<f32> + Primitive,
    f32: From<P::Subpixel>,
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
    use crate::utils::image_to_array3;
    use image::Rgb;

    #[test]
    fn test_is_area_fast() {
        assert!(is_area_fast(100, 50)); // 2x downscale
        assert!(is_area_fast(150, 50)); // 3x downscale
        assert!(!is_area_fast(100, 67)); // 1.5x downscale (not integer)
        assert!(!is_area_fast(50, 100)); // upscale
    }

    #[test]
    fn test_compute_area_weights() {
        let weights = compute_area_weights(4, 2);
        assert_eq!(weights.len(), 2);

        // Check that weights cover the full source range
        for (i, weight) in weights.iter().enumerate() {
            let expected_start = i as f32 * 2.0;
            let expected_end = expected_start + 2.0;
            assert!((weight.src_start - expected_start).abs() < f32::EPSILON);
            assert!((weight.src_end - expected_end).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_resize_area_fast() {
        let src: Image<Rgb<u8>> =
            Image::from_fn(4, 4, |x, y| Rgb([((x + y) * 50) as u8, 100, 150]));
        let src_array = image_to_array3(&src).mapv(f32::from);

        let result = resize_area_fast::<Rgb<u8>, u8>(&src_array.view(), 2, 2).unwrap();
        assert_eq!(result.dim(), (2, 2, 3));

        // Check that result is not empty
        assert!(result[[0, 0, 0]] > 0.0);
    }

    #[test]
    fn test_resize_area_general() {
        let src: Image<Rgb<u8>> =
            Image::from_fn(6, 6, |x, y| Rgb([((x + y) * 20) as u8, 100, 150]));
        let src_array = image_to_array3(&src).mapv(f32::from);

        let result = resize_area_general::<Rgb<u8>, u8>(&src_array.view(), 4, 4).unwrap();
        assert_eq!(result.dim(), (4, 4, 3));

        // Check that result is not empty
        assert!(result[[0, 0, 0]] > 0.0);
    }

    #[test]
    fn test_inter_area_resize() {
        let src: Image<Rgb<u8>> =
            Image::from_fn(8, 8, |x, y| Rgb([((x + y) * 16) as u8, 100, 150]));

        let resizer = InterAreaResize::new(4, 4).unwrap();
        let result = resizer.resize(&src).unwrap();

        assert_eq!(result.dimensions(), (4, 4));

        // Check that result is not empty
        assert!(result.get_pixel(0, 0)[0] > 0);
    }

    #[test]
    fn test_extension_trait() {
        let src: Image<Rgb<u8>> =
            Image::from_fn(8, 8, |x, y| Rgb([((x + y) * 16) as u8, 100, 150]));

        let result = src.resize_area(4, 4).unwrap();
        assert_eq!(result.dimensions(), (4, 4));
    }

    #[test]
    fn test_comprehensive_workflow() {
        // Create a gradient image
        let src: Image<Rgb<u8>> = Image::from_fn(100, 100, |x, y| {
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
