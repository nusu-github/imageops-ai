//! Implementation of the Blur-Fusion foreground color estimation algorithm
//!
//! This module implements the Blur-Fusion algorithm proposed in the following research paper:
//!
//! **"Approximate Fast Foreground Colour Estimation"**  
//! IEEE International Conference on Image Processing (ICIP) 2021  
//! Date of Conference: 19-22 September 2021  
//! DOI: 10.1109/ICIP42928.2021.9506164  
//! Publisher: IEEE  
//! Conference Location: Anchorage, AK, USA  
//!
//! ## Overview
//!
//! When compositing objects extracted through alpha matting onto new backgrounds,
//! estimating foreground colors in transparent regions becomes a critical problem.
//! Simply using the original image as the foreground causes color bleeding from
//! the original background.
//!
//! This Blur-Fusion algorithm was developed as an approximation of Germer et al.'s
//! multi-level foreground estimation method. Despite being implementable in just
//! 11 lines of Python code, it achieves results comparable to state-of-the-art
//! methods with high speed.
//!
//! ## Mathematical Background of the Algorithm
//!
//! ### Compositing Equation
//!
//! The fundamental model of alpha matting is expressed by the following compositing equation:
//!
//! ```text
//! I_i = α_i * F_i + (1 - α_i) * B_i
//! ```
//!
//! Where:
//! - `I_i`: Observed color value at pixel position i
//! - `F_i`: Foreground color
//! - `B_i`: Background color  
//! - `α_i`: Mixing level between foreground and background (0=transparent, 1=opaque)
//!
//! ### Blur-Fusion Cost Function
//!
//! This implementation uses the following modified cost function:
//!
//! ```text
//! cost_local(F_i, B_i) = (α_i * F_i + (1-α_i) * B_i - I_i)² +
//!                        Σ[α_j * (F_i - F_j)² + (1-α_j) * (B_i - B_j)²]
//! ```
//!
//! ### Smoothed Estimation (Equations 4, 5)
//!
//! By minimizing the spatial smoothness term, we obtain the following estimates:
//!
//! ```text
//! F̂_i = Σ(F_j * α_j) / Σ(α_j)      (Equation 4)
//! B̂_i = Σ(B_j * (1-α_j)) / Σ(1-α_j) (Equation 5)
//! ```
//!
//! ### Final Foreground Estimation (Equation 7)
//!
//! The final foreground color is calculated using the following formula:
//!
//! ```text
//! F_i = F̂_i + α_i * (I_i - α_i * F̂_i - (1-α_i) * B̂_i)
//! ```
//!
//! ## Algorithm Features
//!
//! ### Performance
//! - **High-speed processing**: Efficient computation through box filter optimization
//! - **Memory efficiency**: Significantly reduced memory usage compared to Germer et al.'s method
//! - **Parallel processing**: Acceleration through channel separation for parallelization
//!
//! ### Implementation Details
//! - **Neighborhood radius**: Uses r=90 by default (paper recommended value)
//! - **Iterative processing**: Blur-Fusion x2 performs 2 iterations (r=90, r=6)
//! - **Type safety**: Safety and performance through Rust's type system
//! - **image::imageops integration**: Leverages standard library functions for better performance
//!
//! ## Usage Examples
//!
//! ### Basic Usage
//!
//! ```rust
//! use imageops_ai::{ForegroundEstimator};
//! use imageproc::definitions::Image;
//! use image::{Rgb, Luma};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Load input image and alpha matte
//! let image: Image<Rgb<u8>> = Image::new(640, 480);
//! let alpha: Image<Luma<u8>> = Image::new(640, 480);
//!
//! // Or use trait method
//! let foreground = image.estimate_foreground(&alpha, 90)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Parameter Tuning Guidelines
//!
//! - **radius**: 90 provides the most accurate average results
//! - **Large uncertain regions**: Use larger radius values
//! - **Local improvements**: Use r=6 for second iteration
//! - **iterations**: 1 (standard) or 2 (Blur-Fusion x2)
//!
//! ## Error Handling
//!
//! The function returns errors under the following conditions:
//! - Image and alpha matte dimensions do not match
//! - Radius is 0
//! - Iteration count is 0 or exceeds 2
//!
//! ## Optimization Techniques
//!
//! This implementation adopts the following optimizations:
//!
//! 1. **Channel separation**: Uses `imageproc` channel separation functions
//! 2. **Parallel box filtering**: Processes each color channel independently
//! 3. **Pre-computation**: Efficiency through pre-computed weight images
//! 4. **Inline functions**: Inlining of critical path functions
//! 5. **Type specialization**: Balancing flexibility and performance through generic types
//!
//! ## References
//!
//! [1] Germer, T., Uelwer, T., Conrad, S., & Harmeling, S. "Fast Multi-Level Foreground Estimation."
//!     Proceedings of the 28th ACM International Conference on Multimedia, 2020.
//!
//! [2] Porter, T., & Duff, T. "Compositing digital images."
//!     ACM SIGGRAPH Computer Graphics, 1984.

use image::{Luma, Pixel, Primitive, Rgb};
use imageproc::definitions::{Clamp, Image};
use ndarray::prelude::*;

use crate::{
    error::AlphaMaskError,
    imageops_ai::box_filter::BoxFilterSeparable,
    utils::{array3_to_image, image_to_array3, validate_matching_dimensions},
};

/// Trait for performing Blur-Fusion foreground estimation on RGB images
pub trait ForegroundEstimator<S>
where
    Rgb<S>: Pixel<Subpixel = S>,
    S: Clamp<f32> + Primitive,
    f32: From<S>,
{
    /// Performs Blur-Fusion foreground estimation on the image
    fn estimate_foreground(
        self,
        alpha: &Image<Luma<S>>,
        radius: u32,
    ) -> Result<Image<Rgb<S>>, AlphaMaskError>;

    /// Hidden _mut variant that is not available for this operation
    #[doc(hidden)]
    fn estimate_foreground_mut(
        &mut self,
        _alpha: &Image<Luma<S>>,
        _radius: u32,
    ) -> Result<&mut Self, AlphaMaskError> {
        unimplemented!("estimate_foreground_mut is not available because the algorithm requires new buffer allocations")
    }
}

impl<S> ForegroundEstimator<S> for Image<Rgb<S>>
where
    Rgb<S>: Pixel<Subpixel = S>,
    S: Clamp<f32> + Primitive + Send + Sync,
    f32: From<S>,
{
    fn estimate_foreground(
        self,
        alpha: &Image<Luma<S>>,
        radius: u32,
    ) -> Result<Self, AlphaMaskError> {
        estimate_foreground(&self, alpha, radius, 1)
    }
}

/// Estimates foreground colors using the Blur-Fusion algorithm
pub fn estimate_foreground<S>(
    image: &Image<Rgb<S>>,
    alpha: &Image<Luma<S>>,
    radius: u32,
    iterations: u8,
) -> Result<Image<Rgb<S>>, AlphaMaskError>
where
    Rgb<S>: Pixel<Subpixel = S>,
    S: Clamp<f32> + Primitive + Send + Sync,
    f32: From<S>,
{
    validate_inputs(image, alpha, radius, iterations)?;

    let radii = match iterations {
        1 => vec![radius],
        2 => vec![90, 6], // Standard Blur-Fusion x2 radii
        _ => {
            return Err(AlphaMaskError::InvalidParameter(
                "iterations must be 1 or 2".to_string(),
            ))
        }
    };

    // --- Single conversion from ImageBuffer to ndarray ---
    let max_val = f32::from(S::DEFAULT_MAX_VALUE);
    let inv_max_val = 1.0 / max_val;

    let image_arr = image_to_array3(image).mapv(f32::from);
    let alpha_arr = image_to_array3(alpha).mapv(|x| f32::from(x) * inv_max_val);

    let mut foreground_arr = image_arr.clone();

    for r in radii {
        apply_blur_fusion_step_ndarray(
            &image_arr.view(),
            &alpha_arr.view(),
            &mut foreground_arr,
            r,
        )?;
    }

    // --- Single conversion from ndarray to ImageBuffer ---
    Ok(array3_to_image(&foreground_arr.mapv(S::clamp).view()))
}

/// Applies one step of the Blur-Fusion algorithm entirely on ndarrays.
fn apply_blur_fusion_step_ndarray(
    image: &ArrayView3<f32>,
    alpha: &ArrayView3<f32>,
    foreground: &mut Array3<f32>,
    radius: u32,
) -> Result<(), AlphaMaskError> {
    // Phase 1: Compute blurred estimates using the ndarray-based function
    let (f_hat, b_hat) = compute_optimized_smoothed_estimates_ndarray(
        &foreground.view(),
        image, // The original image is used as the initial background estimate
        alpha,
        radius,
    )?;

    // Phase 2: Apply final foreground estimation using ndarray vectorized operations
    let beta = 1.0 - alpha;
    let correction = image - (alpha * &f_hat) - (&beta * &b_hat);
    *foreground = &f_hat + alpha * &correction;

    Ok(())
}

/// Optimized computation of smoothed estimates using direct f32 box filtering on ndarrays.
fn compute_optimized_smoothed_estimates_ndarray(
    foreground: &ArrayView3<f32>,
    background: &ArrayView3<f32>,
    alpha: &ArrayView3<f32>,
    radius: u32,
) -> Result<(Array3<f32>, Array3<f32>), AlphaMaskError> {
    let beta = 1.0 - alpha;

    // Create weighted arrays
    let fg_weighted = foreground * alpha;
    let bg_weighted = background * &beta;

    // Apply box filter to all weighted images
    let filter = BoxFilterSeparable::new(radius)?;
    let fg_blurred = filter.filter_array(&fg_weighted.view())?;
    let bg_blurred = filter.filter_array(&bg_weighted.view())?;
    let alpha_blurred = filter.filter_array(&alpha.view())?;
    let beta_blurred = filter.filter_array(&beta.view())?;

    // Reconstruct final averaged images
    let mut f_hat = foreground.to_owned();
    let mut b_hat = background.to_owned();

    // Avoid division by zero where weights are zero
    let alpha_weight_mask = &alpha_blurred.mapv(|x| x > 1e-6);
    let beta_weight_mask = &beta_blurred.mapv(|x| x > 1e-6);

    f_hat.zip_mut_with(alpha_weight_mask, |f, &mask| {
        if !mask {
            *f = 0.0
        }
    });
    b_hat.zip_mut_with(beta_weight_mask, |b, &mask| {
        if !mask {
            *b = 0.0
        }
    });

    let f_hat_normalized = fg_blurred / &alpha_blurred.mapv(|x| if x > 1e-6 { x } else { 1.0 });
    let b_hat_normalized = bg_blurred / &beta_blurred.mapv(|x| if x > 1e-6 { x } else { 1.0 });

    f_hat.zip_mut_with(&f_hat_normalized, |f, &n| *f = n);
    b_hat.zip_mut_with(&b_hat_normalized, |b, &n| *b = n);

    Ok((f_hat, b_hat))
}

/// Validates input parameters
fn validate_inputs<T>(
    image: &Image<Rgb<T>>,
    alpha: &Image<Luma<T>>,
    radius: u32,
    iterations: u8,
) -> Result<(), AlphaMaskError>
where
    Rgb<T>: Pixel<Subpixel = T>,
    T: Primitive,
{
    let (img_w, img_h) = image.dimensions();
    let (alpha_w, alpha_h) = alpha.dimensions();

    validate_matching_dimensions(img_w, img_h, alpha_w, alpha_h, "ForegroundEstimator").map_err(
        |_| AlphaMaskError::DimensionMismatch {
            expected: (img_w, img_h),
            actual: (alpha_w, alpha_h),
        },
    )?;

    if radius == 0 {
        return Err(AlphaMaskError::InvalidParameter(
            "radius must be > 0".to_string(),
        ));
    }

    let min_dimension_required = 2 * radius + 1;
    let min_dimension = img_w.min(img_h);
    if min_dimension < min_dimension_required {
        return Err(AlphaMaskError::InvalidParameter(format!(
            "Image dimensions ({img_w}x{img_h}) are too small for radius {radius}. Minimum required: {min_dimension_required}x{min_dimension_required}"
        )));
    }

    if iterations == 0 || iterations > 2 {
        return Err(AlphaMaskError::InvalidParameter(
            "iterations must be 1 or 2".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::image_to_array3;
    use image::{ImageBuffer, Luma, Rgb};

    #[test]
    fn test_validate_inputs() {
        let image: Image<Rgb<u8>> = Image::new(12, 12);
        let alpha: Image<Luma<u8>> = Image::new(12, 12);

        assert!(validate_inputs(&image, &alpha, 5, 1).is_ok());

        let alpha_wrong_size: Image<Luma<u8>> = Image::new(5, 5);
        assert!(validate_inputs(&image, &alpha_wrong_size, 5, 1).is_err());

        assert!(validate_inputs(&image, &alpha, 0, 1).is_err());
        assert!(validate_inputs(&image, &alpha, 5, 0).is_err());
        assert!(validate_inputs(&image, &alpha, 5, 3).is_err());

        let small_image: Image<Rgb<u8>> = Image::new(3, 3);
        let small_alpha: Image<Luma<u8>> = Image::new(3, 3);
        assert!(validate_inputs(&small_image, &small_alpha, 5, 1).is_err());
    }

    #[test]
    fn test_blur_fusion_ndarray_logic() {
        let image: Image<Rgb<f32>> = ImageBuffer::from_pixel(10, 10, Rgb([0.5, 0.2, 0.8]));
        let alpha: Image<Luma<f32>> = ImageBuffer::from_pixel(10, 10, Luma([0.5]));

        let result_img = estimate_foreground(&image, &alpha, 1, 1).unwrap();
        let result_arr = image_to_array3(&result_img);

        // In a uniform area, the foreground should be close to the original image color.
        let pixel = result_arr.slice(s![5usize, 5usize, ..]);
        assert!((pixel[0] - 0.5).abs() < 1e-5);
        assert!((pixel[1] - 0.2).abs() < 1e-5);
        assert!((pixel[2] - 0.8).abs() < 1e-5);
    }
}
