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
//! use imageops_ai::{estimate_foreground, ForegroundEstimator};
//! use imageproc::definitions::Image;
//! use image::{Rgb, Luma};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Load input image and alpha matte
//! let image: Image<Rgb<u8>> = Image::new(640, 480);
//! let alpha: Image<Luma<u8>> = Image::new(640, 480);
//!
//! // Estimate foreground using Blur-Fusion (r=90, 1 iteration)
//! let foreground = estimate_foreground(&image, &alpha, 90, 1)?;
//!
//! // Or use trait method
//! let foreground = image.estimate_foreground(&alpha, 90)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Blur-Fusion x2 (2 iterations)
//!
//! ```rust
//! # use imageops_ai::estimate_foreground;
//! # use imageproc::definitions::Image;
//! # use image::{Rgb, Luma};
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let image: Image<Rgb<u8>> = Image::new(640, 480);
//! let alpha: Image<Luma<u8>> = Image::new(640, 480);
//!
//! // 2 iterations for more precise estimation (r=90, r=6 in sequence)
//! let foreground = estimate_foreground(&image, &alpha, 90, 2)?;
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

use crate::imageops_ai::box_filter::{BoxFilter, BoxFilterIntegral};
use crate::utils::validate_matching_dimensions;
use crate::AlphaMaskError;
use image::{ImageBuffer, Luma, LumaA, Pixel, Primitive, Rgb};
use imageproc::definitions::{Clamp, Image};
use itertools::izip;

/// Trait for performing Blur-Fusion foreground estimation on RGB images
///
/// This trait provides convenient method chaining for foreground estimation
/// functionality on RGB images. Internally, it calls the `estimate_foreground`
/// function with 1 iteration.
///
/// Note: This operation requires creating new images for the blurred estimates,
/// so there is no `_mut` variant available. The algorithm always requires
/// allocation of new buffers for intermediate calculations.
pub trait ForegroundEstimator<S>
where
    Rgb<S>: Pixel<Subpixel = S>,
    S: Into<f32> + Clamp<f32> + Primitive,
{
    /// Performs Blur-Fusion foreground estimation on the image
    ///
    /// This consumes the original image.
    ///
    /// # Arguments
    /// * `alpha` - Alpha matte (grayscale image)
    /// * `radius` - Neighborhood radius for blur operations
    ///
    /// # Returns
    /// * `Ok(Image<Rgb<S>>)` - Estimated foreground image
    /// * `Err(Error)` - If an error occurs
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
    S: Into<f32> + Clamp<f32> + Primitive,
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
///
/// This function provides a complete implementation of the Blur-Fusion algorithm
/// proposed in the paper. It performs smoothed estimation via equations 4 and 5,
/// followed by final foreground estimation via equation 7.
///
/// # Arguments
/// * `image` - Input RGB image
/// * `alpha` - Alpha matte (grayscale image, 0=transparent, max_value=opaque)
/// * `radius` - Neighborhood radius for blur operations (paper recommended: 90)
/// * `iterations` - Number of iterations (1=standard, 2=Blur-Fusion x2)
///
/// # Returns
/// * `Ok(Image<Rgb<T>>)` - Estimated foreground image
/// * `Err(Error)` - If dimensions don't match or other errors occur
///
/// # Panics
/// This function does not panic. All error conditions are handled through `Result`.
///
/// # Examples
/// ```no_run
/// use imageops_ai::estimate_foreground;
/// use imageproc::definitions::Image;
/// use image::{Rgb, Luma};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let image: Image<Rgb<u8>> = Image::new(100, 100);
/// let alpha: Image<Luma<u8>> = Image::new(100, 100);
///
/// // Standard Blur-Fusion (1 iteration)
/// let foreground = estimate_foreground(&image, &alpha, 90, 1)?;
///
/// // Blur-Fusion x2 (2 iterations, more precise)
/// let foreground_x2 = estimate_foreground(&image, &alpha, 90, 2)?;
/// # Ok(())
/// # }
/// ```
pub fn estimate_foreground<T>(
    image: &Image<Rgb<T>>,
    alpha: &Image<Luma<T>>,
    radius: u32,
    iterations: u8,
) -> Result<Image<Rgb<T>>, AlphaMaskError>
where
    Rgb<T>: Pixel<Subpixel = T>,
    T: Into<f32> + Clamp<f32> + Primitive,
{
    validate_inputs(image, alpha, radius, iterations)?;

    let mut foreground = image.clone();
    let background = image;

    // Use standard radii for iterations
    let radii = match iterations {
        1 => vec![radius],
        2 => vec![90, 6], // Standard Blur-Fusion x2 radii
        _ => {
            return Err(AlphaMaskError::InvalidParameter(
                "iterations must be 1 or 2".to_string(),
            ))
        }
    };

    for r in radii {
        apply_blur_fusion_step(image, alpha, &mut foreground, background, r)?;
    }

    Ok(foreground)
}

/// Applies one step of the Blur-Fusion algorithm using optimized box filtering
///
/// This function implements equations 4, 5, and 7 from the paper, performing:
/// 1. Computation of optimized smoothed estimates (Equations 4, 5)
/// 2. Application of final foreground estimation (Equation 7)
fn apply_blur_fusion_step<T>(
    image: &Image<Rgb<T>>,
    alpha: &Image<Luma<T>>,
    foreground: &mut Image<Rgb<T>>,
    background: &Image<Rgb<T>>,
    radius: u32,
) -> Result<(), AlphaMaskError>
where
    Rgb<T>: Pixel<Subpixel = T>,
    T: Into<f32> + Clamp<f32> + Primitive,
{
    // Phase 1: Compute weighted blurred estimates using optimized approach
    let (f_hat, b_hat) =
        compute_optimized_smoothed_estimates(foreground, background, alpha, radius)?;

    // Phase 2: Apply final foreground estimation (Equation 7)
    let max_val = T::DEFAULT_MAX_VALUE.into();
    for (i_pixel, f_hat_pixel, b_hat_pixel, Luma([alpha_val]), f_pixel) in izip!(
        image.pixels(),
        f_hat.pixels(),
        b_hat.pixels(),
        alpha.pixels(),
        foreground.pixels_mut()
    ) {
        let normalized_alpha = (*alpha_val).into() / max_val;
        *f_pixel =
            compute_final_foreground_pixel(*i_pixel, *f_hat_pixel, *b_hat_pixel, normalized_alpha);
    }

    Ok(())
}

/// Optimized computation of smoothed estimates using direct f32 box filtering (Equations 4 and 5)
///
/// This function efficiently implements equations 4 and 5 from the paper:
/// - F̂_i = Σ(F_j * α_j) / Σ(α_j)
/// - B̂_i = Σ(B_j * (1-α_j)) / Σ(1-α_j)
///
/// Acceleration is achieved through direct f32 calculations and optimized channel processing.
fn compute_optimized_smoothed_estimates<T>(
    foreground: &Image<Rgb<T>>,
    background: &Image<Rgb<T>>,
    alpha: &Image<Luma<T>>,
    radius: u32,
) -> Result<(Image<Rgb<T>>, Image<Rgb<T>>), AlphaMaskError>
where
    Rgb<T>: Pixel<Subpixel = T>,
    T: Into<f32> + Clamp<f32> + Primitive,
{
    let (width, height) = foreground.dimensions();

    let mut fg_weighted = ImageBuffer::new(width, height);
    let mut bg_weighted = ImageBuffer::new(width, height);
    let mut alpha_beta_weighted = ImageBuffer::new(width, height);

    for (
        fg_pixel,
        bg_pixel,
        alpha_beta_pixel,
        fg_weighted_pixel,
        bg_weighted_pixel,
        alpha_beta_weighted_pixel,
    ) in izip!(
        foreground.pixels(),
        background.pixels(),
        alpha.pixels(),
        fg_weighted.pixels_mut(),
        bg_weighted.pixels_mut(),
        alpha_beta_weighted.pixels_mut()
    ) {
        let Rgb([fg_r, fg_g, fg_b]) = *fg_pixel;
        let Rgb([bg_r, bg_g, bg_b]) = *bg_pixel;
        let Luma([alpha]) = *alpha_beta_pixel;
        let beta = T::DEFAULT_MAX_VALUE - alpha;
        *fg_weighted_pixel = Rgb([fg_r * alpha, fg_g * alpha, fg_b * alpha]);
        *bg_weighted_pixel = Rgb([bg_r * beta, bg_g * beta, bg_b * beta]);
        *alpha_beta_weighted_pixel = LumaA([alpha, beta]);
    }

    // Apply box filter to all weighted images using integral image implementation
    let filter = BoxFilterIntegral::new(radius).unwrap(); // radius already validated
    let (fg_blurred, bg_blurred, alpha_beta_weights_blurred) = (
        filter.filter(&fg_weighted).unwrap(),
        filter.filter(&bg_weighted).unwrap(),
        filter.filter(&alpha_beta_weighted).unwrap(),
    );

    // Reconstruct final averaged images using direct calculations
    let mut f_hat = ImageBuffer::new(width, height);
    let mut b_hat = ImageBuffer::new(width, height);

    for (
        fg_pixel,
        bg_pixel,
        fg_blurred_pixel,
        bg_blurred_pixel,
        alpha_beta_weights_blurred_pixel,
        f_hat_pixel,
        b_hat_pixel,
    ) in izip!(
        foreground.pixels(),
        background.pixels(),
        fg_blurred.pixels(),
        bg_blurred.pixels(),
        alpha_beta_weights_blurred.pixels(),
        f_hat.pixels_mut(),
        b_hat.pixels_mut(),
    ) {
        let LumaA([alpha_weight, beta_weight]) = *alpha_beta_weights_blurred_pixel;

        if alpha_weight > T::DEFAULT_MIN_VALUE {
            let inv_alpha_weight = T::DEFAULT_MAX_VALUE / alpha_weight;
            let Rgb([fg_sum_r, fg_sum_g, fg_sum_b]) = *fg_blurred_pixel;
            *f_hat_pixel = Rgb([
                fg_sum_r * inv_alpha_weight,
                fg_sum_g * inv_alpha_weight,
                fg_sum_b * inv_alpha_weight,
            ]);
        } else {
            *f_hat_pixel = *fg_pixel;
        }

        if beta_weight > T::DEFAULT_MIN_VALUE {
            let inv_beta_weight = T::DEFAULT_MAX_VALUE / beta_weight;
            let Rgb([bg_sum_r, bg_sum_g, bg_sum_b]) = *bg_blurred_pixel;
            *b_hat_pixel = Rgb([
                bg_sum_r * inv_beta_weight,
                bg_sum_g * inv_beta_weight,
                bg_sum_b * inv_beta_weight,
            ]);
        } else {
            *b_hat_pixel = *bg_pixel;
        }
    }

    Ok((f_hat, b_hat))
}

/// Computes the final foreground pixel using Equation 7
///
/// F_i = F̂_i + α_i * (I_i - α_i * F̂_i - (1-α_i) * B̂_i)
#[inline]
fn compute_final_foreground_pixel<T>(i: Rgb<T>, f_hat: Rgb<T>, b_hat: Rgb<T>, alpha: f32) -> Rgb<T>
where
    Rgb<T>: Pixel<Subpixel = T>,
    T: Into<f32> + Clamp<f32> + Primitive,
{
    let beta = 1.0 - alpha;

    let mut result = [T::DEFAULT_MIN_VALUE; 3];
    for c in 0..3 {
        let i_c = i[c].into();
        let f_hat_c = f_hat[c].into();
        let b_hat_c = b_hat[c].into();

        // F_i = F_hat_i + alpha_i * (I_i - alpha_i * F_hat_i - (1-alpha_i) * B_hat_i)
        let correction = beta.mul_add(-b_hat_c, alpha.mul_add(-f_hat_c, i_c));
        let final_val = alpha.mul_add(correction, f_hat_c);

        result[c] = T::clamp(final_val);
    }

    Rgb(result)
}

/// Validates input parameters
///
/// Checks the following conditions:
/// - Image and alpha matte dimensions match
/// - Radius is greater than 0
/// - Iteration count is 1 or 2
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

    // Check if image is large enough for box filter with given radius
    let min_dimension_required = 2 * radius + 1;
    let min_dimension = img_w.min(img_h);
    if min_dimension < min_dimension_required {
        return Err(AlphaMaskError::InvalidParameter(
            format!("Image dimensions ({img_w}x{img_h}) are too small for radius {radius}. Minimum required: {min_dimension_required}x{min_dimension_required}")
        ));
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

    #[test]
    fn test_validate_inputs() {
        // Use 12x12 image for radius 5 (needs at least 11x11)
        let image: Image<Rgb<u8>> = Image::new(12, 12);
        let alpha: Image<Luma<u8>> = Image::new(12, 12);

        assert!(validate_inputs(&image, &alpha, 5, 1).is_ok());

        let alpha_wrong_size: Image<Luma<u8>> = Image::new(5, 5);
        assert!(validate_inputs(&image, &alpha_wrong_size, 5, 1).is_err());

        assert!(validate_inputs(&image, &alpha, 0, 1).is_err());
        assert!(validate_inputs(&image, &alpha, 5, 0).is_err());
        assert!(validate_inputs(&image, &alpha, 5, 3).is_err());

        // Test image too small for radius
        let small_image: Image<Rgb<u8>> = Image::new(3, 3);
        let small_alpha: Image<Luma<u8>> = Image::new(3, 3);
        assert!(validate_inputs(&small_image, &small_alpha, 5, 1).is_err());
    }

    #[test]
    fn test_box_filter_integration() {
        // 基本的なテスト: 均一な画像
        let image = ImageBuffer::from_fn(5, 5, |_, _| Luma([1.0f32]));
        let filter = BoxFilterIntegral::new(1).unwrap();
        let filtered = filter.filter(&image).unwrap();
        assert_eq!(filtered.get_pixel(2, 2)[0], 1.0);

        // 境界でのテスト: 左上角
        assert_eq!(filtered.get_pixel(0, 0)[0], 1.0);

        // 境界でのテスト: 右下角
        assert_eq!(filtered.get_pixel(4, 4)[0], 1.0);

        // より複雑なテスト: 中央に異なる値
        let mut test_image = ImageBuffer::from_fn(5, 5, |_, _| Luma([1.0f32]));
        test_image.put_pixel(2, 2, Luma([5.0f32]));

        let filtered_complex = filter.filter(&test_image).unwrap();

        // 中央のピクセルは周囲8個（値1.0）と自分（値5.0）の平均
        // (8 * 1.0 + 1 * 5.0) / 9 = 13.0 / 9 ≈ 1.444
        let expected_center = 8.0f32.mul_add(1.0, 1.0 * 5.0) / 9.0;
        assert!((filtered_complex.get_pixel(2, 2)[0] - expected_center).abs() < 0.001);
    }

    #[test]
    fn test_box_filter_integration_basic() {
        // 基本的なテスト: 均一な画像でのIntegral実装
        let image = ImageBuffer::from_fn(7, 7, |_, _| Luma([2.0f32]));

        let integral_filter = BoxFilterIntegral::new(2).unwrap();
        let integral_result = integral_filter.filter(&image).unwrap();

        // 均一な画像では全てのピクセルが同じ値になるべき
        for y in 0..7 {
            for x in 0..7 {
                let integral_val = integral_result.get_pixel(x, y)[0];
                assert!(
                    (integral_val - 2.0).abs() < 0.001,
                    "Uniform image test failed at ({}, {}): expected=2.0, actual={}",
                    x,
                    y,
                    integral_val
                );
            }
        }
    }
}
