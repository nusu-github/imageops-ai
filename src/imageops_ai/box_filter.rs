use std::marker::{Send, Sync};

use image::{Pixel, Primitive};
use imageproc::definitions::{Clamp, Image};

#[cfg(feature = "rayon")]
use ndarray::parallel::prelude::*;

use ndarray::prelude::*;

use crate::{
    error::BoxFilterError,
    utils::{array3_to_image, image_to_array3},
};

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
    P: Pixel + Send + Sync,
{
    /// Apply box filter to the image
    fn filter(&self, image: &Image<P>) -> Result<Image<P>, BoxFilterError>;
}

/// Separable box filter implementation inspired by OpenCV
/// Uses row and column filters for better memory efficiency
pub struct BoxFilterSeparable {
    radius: u32,
    border_type: BorderType,
    use_parallel: bool,
}

/// Border handling helper functions
mod border_utils {
    use super::BorderType;

    /// Get pixel coordinate based on border type
    pub fn get_border_coord(coord: i32, size: u32, border_type: BorderType) -> u32 {
        let size_i32 = size as i32;
        let coord = match border_type {
            BorderType::Constant => coord.clamp(0, size_i32 - 1),
            BorderType::Replicate => coord.clamp(0, size_i32 - 1),
            BorderType::Reflect => {
                if coord < 0 {
                    -coord - 1
                } else if coord >= size_i32 {
                    2 * size_i32 - coord - 1
                } else {
                    coord
                }
            }
            BorderType::Wrap => {
                if coord < 0 {
                    size_i32 + coord
                } else if coord >= size_i32 {
                    coord - size_i32
                } else {
                    coord
                }
            }
            BorderType::Reflect101 => {
                if coord < 0 {
                    -coord
                } else if coord >= size_i32 {
                    2 * size_i32 - coord - 2
                } else {
                    coord
                }
            }
        };
        coord.clamp(0, size_i32 - 1) as u32
    }
}

impl BoxFilterSeparable {
    /// Create a new separable box filter with default border type
    pub const fn new(radius: u32) -> Result<Self, BoxFilterError> {
        Ok(Self {
            radius,
            border_type: BorderType::Reflect101,
            use_parallel: true,
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
            use_parallel: true,
        })
    }

    /// Create a new separable box filter with parallel processing control
    pub const fn new_with_options(
        radius: u32,
        border_type: BorderType,
        use_parallel: bool,
    ) -> Result<Self, BoxFilterError> {
        Ok(Self {
            radius,
            border_type,
            use_parallel,
        })
    }

    /// Get the kernel size (2 * radius + 1)
    #[inline]
    pub const fn kernel_size(&self) -> u32 {
        2 * self.radius + 1
    }

    /// Applies the box filter to a 3D ndarray.
    pub fn filter_array(&self, arr: &ArrayView3<f32>) -> Result<Array3<f32>, BoxFilterError> {
        let (height, width, _) = arr.dim();
        let min_dimension = (width as u32).min(height as u32);
        if min_dimension < self.kernel_size() {
            return Err(BoxFilterError::ImageTooSmall {
                width: width as u32,
                height: height as u32,
                radius: self.radius,
            });
        }

        let radius = self.radius as usize;

        // --- Horizontal pass ---
        let mut intermediate = Array3::<f32>::zeros(arr.dim());
        if self.use_parallel {
            #[cfg(feature = "rayon")]
            par_azip!((out_row in intermediate.axis_iter_mut(Axis(0)), in_row in arr.axis_iter(Axis(0))) {
                apply_sliding_window_1d(in_row, out_row, radius, self.border_type);
            });
            #[cfg(not(feature = "rayon"))]
            azip!((out_row in intermediate.axis_iter_mut(Axis(0)), in_row in arr.axis_iter(Axis(0))) {
                apply_sliding_window_1d(in_row, out_row, radius, self.border_type);
            });
        } else {
            azip!((out_row in intermediate.axis_iter_mut(Axis(0)), in_row in arr.axis_iter(Axis(0))) {
                apply_sliding_window_1d(in_row, out_row, radius, self.border_type);
            });
        }

        // --- Vertical pass ---
        let mut final_sum = Array3::<f32>::zeros(arr.dim());
        let intermediate_t = intermediate.permuted_axes([1, 0, 2]);
        let mut final_sum_t = final_sum.view_mut().permuted_axes([1, 0, 2]);

        if self.use_parallel {
            #[cfg(feature = "rayon")]
            par_azip!((out_col in final_sum_t.axis_iter_mut(Axis(0)), in_col in intermediate_t.axis_iter(Axis(0))) {
                apply_sliding_window_1d(in_col, out_col, radius, self.border_type);
            });
            #[cfg(not(feature = "rayon"))]
            azip!((out_col in final_sum_t.axis_iter_mut(Axis(0)), in_col in intermediate_t.axis_iter(Axis(0))) {
                apply_sliding_window_1d(in_col, out_col, radius, self.border_type);
            });
        } else {
            azip!((out_col in final_sum_t.axis_iter_mut(Axis(0)), in_col in intermediate_t.axis_iter(Axis(0))) {
                apply_sliding_window_1d(in_col, out_col, radius, self.border_type);
            });
        }

        let kernel_area = (self.kernel_size() * self.kernel_size()) as f32;
        Ok(final_sum / kernel_area)
    }
}

/// Applies a 1D sliding window sum to an axis of an array.
fn apply_sliding_window_1d(
    input: ArrayView2<f32>,
    mut output: ArrayViewMut2<f32>,
    radius: usize,
    border_type: BorderType,
) {
    let len = input.len_of(Axis(0));
    let channels = input.len_of(Axis(1));
    let kernel_size = 2 * radius + 1;

    for c in 0..channels {
        let mut sum = 0.0f32;
        // Initialize sum for the first position
        for k in 0..kernel_size {
            let src_idx =
                border_utils::get_border_coord(k as i32 - radius as i32, len as u32, border_type)
                    as usize;
            sum += input[[src_idx, c]];
        }
        output[[0, c]] = sum;

        // Slide the window for remaining positions
        for i in 1..len {
            let remove_idx = border_utils::get_border_coord(
                i as i32 - radius as i32 - 1,
                len as u32,
                border_type,
            ) as usize;
            let add_idx =
                border_utils::get_border_coord(i as i32 + radius as i32, len as u32, border_type)
                    as usize;
            sum = sum - input[[remove_idx, c]] + input[[add_idx, c]];
            output[[i, c]] = sum;
        }
    }
}

/// Implement BoxFilter for separable filter
impl<P> BoxFilter<P> for BoxFilterSeparable
where
    P: Pixel + Send + Sync,
    P::Subpixel: Clamp<f32> + Primitive + Send + Sync,
    f32: From<P::Subpixel>,
{
    fn filter(&self, image: &Image<P>) -> Result<Image<P>, BoxFilterError> {
        let (width, height) = image.dimensions();

        if width == 0 || height == 0 {
            return Err(BoxFilterError::EmptyImage { width, height });
        }

        // Convert image to ndarray of f32
        let arr = image_to_array3(image).mapv(f32::from);

        // Apply filter on the ndarray
        let filtered_arr = self.filter_array(&arr.view())?;

        // Convert back to ImageBuffer
        let output = array3_to_image(&filtered_arr.mapv(P::Subpixel::clamp).view());

        Ok(output)
    }
}

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

    /// Apply box filter with parallel processing control
    fn box_filter_with_options(
        self,
        radius: u32,
        border_type: BorderType,
        use_parallel: bool,
    ) -> Result<Self, BoxFilterError>
    where
        Self: Sized;

    /// Apply box filter in-place (not available for this operation)
    fn box_filter_mut(&mut self, radius: u32) -> Result<&mut Self, BoxFilterError>;
}

impl<P> BoxFilterExt<P> for Image<P>
where
    P: Pixel + Send + Sync,
    P::Subpixel: Clamp<f32> + Primitive + Send + Sync,
    f32: From<P::Subpixel>,
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

    fn box_filter_with_options(
        self,
        radius: u32,
        border_type: BorderType,
        use_parallel: bool,
    ) -> Result<Self, BoxFilterError> {
        let filter = BoxFilterSeparable::new_with_options(radius, border_type, use_parallel)?;
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

        let filter = BoxFilterSeparable::new_with_border(1, BorderType::Replicate).unwrap();
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

        assert!(matches!(result, Err(BoxFilterError::EmptyImage { .. })));
    }

    #[test]
    fn test_image_too_small_error() {
        use image::Rgb;
        let img = ImageBuffer::from_pixel(2, 2, Rgb([100u8, 100, 100]));
        let filter = BoxFilterSeparable::new(2).unwrap(); // radius 2 needs at least 5x5 image
        let result = filter.filter(&img);

        assert!(matches!(result, Err(BoxFilterError::ImageTooSmall { .. })));
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

    #[cfg(feature = "rayon")]
    #[test]
    fn test_parallel_vs_sequential() {
        use image::Rgb;
        let img = ImageBuffer::from_pixel(800, 600, Rgb([100u8, 100, 100]));

        // Test parallel version
        let start = std::time::Instant::now();
        let filter_parallel =
            BoxFilterSeparable::new_with_options(3, BorderType::Reflect101, true).unwrap();
        let result_parallel = filter_parallel.filter(&img).unwrap();
        let duration_parallel = start.elapsed();

        // Test sequential version
        let start = std::time::Instant::now();
        let filter_sequential =
            BoxFilterSeparable::new_with_options(3, BorderType::Reflect101, false).unwrap();
        let result_sequential = filter_sequential.filter(&img).unwrap();
        let duration_sequential = start.elapsed();

        println!(
            "Parallel: {:?}, Sequential: {:?}",
            duration_parallel, duration_sequential
        );

        // Results should be identical
        for (p1, p2) in result_parallel.pixels().zip(result_sequential.pixels()) {
            assert_eq!(p1, p2);
        }

        // For large images, parallel should be faster or comparable
        println!(
            "Parallel speedup: {:.2}x",
            duration_sequential.as_secs_f64() / duration_parallel.as_secs_f64()
        );
    }
}
