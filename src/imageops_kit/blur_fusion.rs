//! Implementation of the Blur-Fusion foreground color estimation algorithm
//!
//! This module implements the Blur-Fusion algorithm proposed in the following research paper:
//!
//! **"Approximate Fast Foreground Colour Estimation"**\
//! IEEE International Conference on Image Processing (ICIP) 2021\
//! Date of Conference: 19-22 September 2021\
//! DOI: 10.1109/ICIP42928.2021.9506164\
//! Publisher: IEEE\
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
//! - **Neighborhood radius**: Uses r=91 by default (adjusted from paper's r=90 to satisfy odd requirement)
//! - **Iterative processing**: Blur-Fusion x2 performs 2 iterations (r=91, r=7)
//! - **Type safety**: Safety and performance through Rust's type system
//!
//! ## Usage Examples
//!
//! ### Basic Usage
//!
//! ```rust
//! use imageops_kit::{estimate_foreground_colors, ForegroundEstimationExt};
//! use imageproc::definitions::Image;
//! use image::{Rgb, Luma};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Load input image and alpha matte
//! let image: Image<Rgb<u8>> = Image::new(640, 480);
//! let alpha: Image<Luma<u8>> = Image::new(640, 480);
//!
//! // Estimate foreground using Blur-Fusion (r=91, 1 iteration)
//! let foreground = estimate_foreground_colors(&image, &alpha, 91, 1)?;
//!
//! // Or use trait method
//! let foreground = image.estimate_foreground_colors(&alpha, 91)?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Blur-Fusion x2 (2 iterations)
//!
//! ```rust
//! # use imageops_kit::estimate_foreground_colors;
//! # use imageproc::definitions::Image;
//! # use image::{Rgb, Luma};
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let image: Image<Rgb<u8>> = Image::new(640, 480);
//! let alpha: Image<Luma<u8>> = Image::new(640, 480);
//!
//! // 2 iterations for more precise estimation (r=91, r=7 in sequence)
//! let foreground = estimate_foreground_colors(&image, &alpha, 91, 2)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Parameter Tuning Guidelines
//!
//! - **radius**: Must be odd. 91 provides the most accurate average results (paper recommends 90, adjusted to 91)
//! - **Large uncertain regions**: Use larger radius values (must be odd)
//! - **Local improvements**: Use r=7 for second iteration (adjusted from paper's r=6)
//! - **iterations**: 1 (standard) or 2 (Blur-Fusion x2)
//!
//! ## Error Handling
//!
//! The function returns errors under the following conditions:
//! - Image and alpha matte dimensions do not match
//! - Radius is 0 or even (must be odd)
//! - Image dimensions are too small for the given radius
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

use image::{ImageBuffer, Luma, Pixel, Primitive, Rgb};
use imageproc::definitions::{Clamp, Image};
use itertools::izip;
use libblur::{
    BlurImage, BlurImageMut, BoxBlurParameters, FastBlurChannels, ThreadingPolicy, box_blur_f32,
};

use crate::{error::AlphaMaskError, utils::validate_matching_dimensions};

type SmoothingResult<T> = Result<(Image<Rgb<T>>, Image<Rgb<T>>), AlphaMaskError>;

/// Trait for performing Blur-Fusion foreground estimation on RGB images.
///
/// This trait provides convenient method chaining for foreground estimation
/// functionality on RGB images. Internally, it calls the `estimate_foreground_colors`
/// function with 1 iteration.
///
/// Note: This operation requires creating new images for the blurred estimates,
/// so there is no `_mut` variant available. The algorithm always requires
/// allocation of new buffers for intermediate calculations.
pub trait ForegroundEstimationExt<S>
where
    Rgb<S>: Pixel<Subpixel = S>,
    S: Clamp<f32> + Primitive,
    f32: From<S>,
{
    /// Performs Blur-Fusion foreground estimation on the image.
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
    fn estimate_foreground_colors(
        self,
        alpha: &Image<Luma<S>>,
        radius: u32,
    ) -> Result<Image<Rgb<S>>, AlphaMaskError>;

    /// Hidden _mut variant that is not available for this operation.
    #[doc(hidden)]
    fn estimate_foreground_colors_mut(
        &mut self,
        _alpha: &Image<Luma<S>>,
        _radius: u32,
    ) -> Result<&mut Self, AlphaMaskError> {
        unimplemented!(
            "estimate_foreground_colors_mut is not available because the algorithm requires new buffer allocations"
        )
    }
}

impl<S> ForegroundEstimationExt<S> for Image<Rgb<S>>
where
    Rgb<S>: Pixel<Subpixel = S>,
    S: Clamp<f32> + Primitive,
    f32: From<S>,
{
    fn estimate_foreground_colors(
        self,
        alpha: &Image<Luma<S>>,
        radius: u32,
    ) -> Result<Self, AlphaMaskError> {
        estimate_foreground_colors(&self, alpha, radius, 1)
    }
}

/// Estimates foreground colors using the Blur-Fusion algorithm.
///
/// This function provides a complete implementation of the Blur-Fusion algorithm
/// proposed in the paper. It performs smoothed estimation via equations 4 and 5,
/// followed by final foreground estimation via equation 7.
///
/// # Arguments
/// * `image` - Input RGB image
/// * `alpha` - Alpha matte (grayscale image, 0=transparent, `max_value=opaque`)
/// * `radius` - Neighborhood radius for blur operations (must be odd, paper recommended: 91)
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
/// use imageops_kit::estimate_foreground_colors;
/// use imageproc::definitions::Image;
/// use image::{Rgb, Luma};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let image: Image<Rgb<u8>> = Image::new(100, 100);
/// let alpha: Image<Luma<u8>> = Image::new(100, 100);
///
/// // Standard Blur-Fusion (1 iteration)
/// let foreground = estimate_foreground_colors(&image, &alpha, 91, 1)?;
///
/// // Blur-Fusion x2 (2 iterations, more precise)
/// let foreground_x2 = estimate_foreground_colors(&image, &alpha, 91, 2)?;
/// # Ok(())
/// # }
/// ```
pub fn estimate_foreground_colors<T>(
    image: &Image<Rgb<T>>,
    alpha: &Image<Luma<T>>,
    radius: u32,
    iterations: u8,
) -> Result<Image<Rgb<T>>, AlphaMaskError>
where
    Rgb<T>: Pixel<Subpixel = T>,
    T: Clamp<f32> + Primitive,
    f32: From<T>,
{
    validate_inputs_impl(image, alpha, radius, iterations)?;

    let mut foreground = image.clone();
    let background = image;

    // Use standard radii for iterations
    let radii = match iterations {
        1 => vec![radius],
        2 => vec![91, 7], // Standard Blur-Fusion x2 radii (adjusted to odd numbers)
        _ => {
            return Err(AlphaMaskError::InvalidParameter(
                "iterations must be 1 or 2".to_owned(),
            ));
        }
    };

    for r in radii {
        apply_blur_fusion_step_impl(image, alpha, &mut foreground, background, r)?;
    }

    Ok(foreground)
}

/// Applies one step of the Blur-Fusion algorithm using optimized box filtering.
///
/// This function implements equations 4, 5, and 7 from the paper, performing:
/// 1. Computation of optimized smoothed estimates (Equations 4, 5)
/// 2. Application of final foreground estimation (Equation 7)
fn apply_blur_fusion_step_impl<T>(
    image: &Image<Rgb<T>>,
    alpha: &Image<Luma<T>>,
    foreground: &mut Image<Rgb<T>>,
    background: &Image<Rgb<T>>,
    radius: u32,
) -> Result<(), AlphaMaskError>
where
    Rgb<T>: Pixel<Subpixel = T>,
    T: Clamp<f32> + Primitive,
    f32: From<T>,
{
    // Phase 1: Compute weighted blurred estimates using optimized approach
    let (f_hat, b_hat) = compute_smoothed_estimates_impl(foreground, background, alpha, radius)?;

    // Phase 2: Apply final foreground estimation (Equation 7) using iterators
    let max_val = f32::from(T::DEFAULT_MAX_VALUE);
    let inv_max_val = 1.0 / max_val;

    izip!(
        foreground.pixels_mut(),
        image.pixels(),
        f_hat.pixels(),
        b_hat.pixels(),
        alpha.pixels()
    )
    .for_each(
        |(fg_pixel, i_pixel, f_hat_pixel, b_hat_pixel, alpha_pixel)| {
            let normalized_alpha = f32::from(alpha_pixel[0]) * inv_max_val;
            let beta = 1.0 - normalized_alpha;

            for c in 0..3 {
                let i_c = f32::from(i_pixel[c]);
                let f_hat_c = f32::from(f_hat_pixel[c]);
                let b_hat_c = f32::from(b_hat_pixel[c]);

                let correction = beta.mul_add(-b_hat_c, normalized_alpha.mul_add(-f_hat_c, i_c));
                let final_val = normalized_alpha.mul_add(correction, f_hat_c);

                fg_pixel[c] = T::clamp(final_val);
            }
        },
    );

    Ok(())
}

/// Computation of smoothed estimates using direct f32 box filtering (Equations 4 and 5).
///
/// This function efficiently implements equations 4 and 5 from the paper:
/// - F̂_i = Σ(F_j * α_j) / Σ(α_j)
/// - B̂_i = Σ(B_j * (1-α_j)) / Σ(1-α_j)
///
/// Acceleration is achieved through direct f32 calculations and optimized channel processing.
fn compute_smoothed_estimates_impl<T>(
    foreground: &Image<Rgb<T>>,
    background: &Image<Rgb<T>>,
    alpha: &Image<Luma<T>>,
    radius: u32,
) -> SmoothingResult<T>
where
    Rgb<T>: Pixel<Subpixel = T>,
    T: Clamp<f32> + Primitive,
    f32: From<T>,
{
    let (width, height) = foreground.dimensions();

    let mut fg_weighted: Image<Rgb<f32>> = ImageBuffer::new(width, height);
    let mut bg_weighted: Image<Rgb<f32>> = ImageBuffer::new(width, height);
    let mut alpha_weighted: Image<Luma<f32>> = ImageBuffer::new(width, height);
    let mut beta_weighted: Image<Luma<f32>> = ImageBuffer::new(width, height);

    // Pre-compute weighted images using iterators for improved safety and readability
    let max_val = f32::from(T::DEFAULT_MAX_VALUE);

    izip!(
        fg_weighted.pixels_mut(),
        bg_weighted.pixels_mut(),
        alpha_weighted.pixels_mut(),
        beta_weighted.pixels_mut(),
        foreground.pixels(),
        background.pixels(),
        alpha.pixels()
    )
    .for_each(
        |(fg_w_pixel, bg_w_pixel, alpha_w_pixel, beta_w_pixel, fg_pixel, bg_pixel, alpha_pixel)| {
            let alpha_f32 = f32::from(alpha_pixel[0]);
            let beta_f32 = max_val - alpha_f32;

            // Update weighted foreground and background
            for c in 0..3 {
                fg_w_pixel[c] = f32::from(fg_pixel[c]) * alpha_f32;
                bg_w_pixel[c] = f32::from(bg_pixel[c]) * beta_f32;
            }

            // Update weighted alpha and beta
            alpha_w_pixel[0] = alpha_f32;
            beta_w_pixel[0] = beta_f32;
        },
    );

    // Apply box filter to all weighted images using integral image implementation
    let converted_fg_weighted = BlurImage::borrow(
        fg_weighted.as_raw(),
        fg_weighted.width(),
        fg_weighted.height(),
        FastBlurChannels::Channels3,
    );
    let mut blurred_fg_weighted = BlurImageMut::default();
    let converted_bg_weighted = BlurImage::borrow(
        bg_weighted.as_raw(),
        bg_weighted.width(),
        bg_weighted.height(),
        FastBlurChannels::Channels3,
    );
    let mut blurred_bg_weighted = BlurImageMut::default();
    let converted_alpha_weighted = BlurImage::borrow(
        alpha_weighted.as_raw(),
        alpha_weighted.width(),
        alpha_weighted.height(),
        FastBlurChannels::Plane,
    );
    let mut blurred_alpha_weighted = BlurImageMut::default();
    let converted_beta_weighted = BlurImage::borrow(
        beta_weighted.as_raw(),
        beta_weighted.width(),
        beta_weighted.height(),
        FastBlurChannels::Plane,
    );
    let mut blurred_beta_weighted = BlurImageMut::default();

    let box_params = BoxBlurParameters::new(radius);

    // Use Sequential threading policy to adhere to project guidelines
    box_blur_f32(
        &converted_fg_weighted,
        &mut blurred_fg_weighted,
        box_params,
        ThreadingPolicy::Single,
    )
    .map_err(|e| AlphaMaskError::BlurFusionError(e.to_string()))?;
    box_blur_f32(
        &converted_bg_weighted,
        &mut blurred_bg_weighted,
        box_params,
        ThreadingPolicy::Single,
    )
    .map_err(|e| AlphaMaskError::BlurFusionError(e.to_string()))?;
    box_blur_f32(
        &converted_alpha_weighted,
        &mut blurred_alpha_weighted,
        box_params,
        ThreadingPolicy::Single,
    )
    .map_err(|e| AlphaMaskError::BlurFusionError(e.to_string()))?;
    box_blur_f32(
        &converted_beta_weighted,
        &mut blurred_beta_weighted,
        box_params,
        ThreadingPolicy::Single,
    )
    .map_err(|e| AlphaMaskError::BlurFusionError(e.to_string()))?;

    // Convert blurred images back to original types
    let fg_blurred: Image<Rgb<f32>> = ImageBuffer::from_raw(
        blurred_fg_weighted.width,
        blurred_fg_weighted.height,
        blurred_fg_weighted.data.borrow().as_ref().to_vec(),
    )
    .ok_or_else(|| {
        AlphaMaskError::BlurFusionError("Failed to create blurred foreground image".to_owned())
    })?;
    let bg_blurred: Image<Rgb<f32>> = ImageBuffer::from_raw(
        blurred_bg_weighted.width,
        blurred_bg_weighted.height,
        blurred_bg_weighted.data.borrow().as_ref().to_vec(),
    )
    .ok_or_else(|| {
        AlphaMaskError::BlurFusionError("Failed to create blurred background image".to_owned())
    })?;
    let alpha_weights_blurred: Image<Luma<f32>> = ImageBuffer::from_raw(
        blurred_alpha_weighted.width,
        blurred_alpha_weighted.height,
        blurred_alpha_weighted.data.borrow().as_ref().to_vec(),
    )
    .ok_or_else(|| {
        AlphaMaskError::BlurFusionError("Failed to create blurred alpha weights image".to_owned())
    })?;
    let beta_weights_blurred: Image<Luma<f32>> = ImageBuffer::from_raw(
        blurred_beta_weighted.width,
        blurred_beta_weighted.height,
        blurred_beta_weighted.data.borrow().as_ref().to_vec(),
    )
    .ok_or_else(|| {
        AlphaMaskError::BlurFusionError("Failed to create blurred beta weights image".to_owned())
    })?;

    // Reconstruct final averaged images using iterators
    let mut f_hat: Image<Rgb<T>> = ImageBuffer::new(width, height);
    let mut b_hat: Image<Rgb<T>> = ImageBuffer::new(width, height);

    izip!(
        f_hat.pixels_mut(),
        b_hat.pixels_mut(),
        foreground.pixels(),
        background.pixels(),
        fg_blurred.pixels(),
        bg_blurred.pixels(),
        alpha_weights_blurred.pixels(),
        beta_weights_blurred.pixels()
    )
    .for_each(
        |(f_hat_pixel, b_hat_pixel, fg_pixel, bg_pixel, fg_b, bg_b, alpha_w, beta_w)| {
            let alpha_weight = alpha_w[0];
            if alpha_weight > 0.0 {
                let inv_alpha_weight = 1.0 / alpha_weight;
                for c in 0..3 {
                    f_hat_pixel[c] = T::clamp(fg_b[c] * inv_alpha_weight);
                }
            } else {
                *f_hat_pixel = *fg_pixel;
            }

            let beta_weight = beta_w[0];
            if beta_weight > 0.0 {
                let inv_beta_weight = 1.0 / beta_weight;
                for c in 0..3 {
                    b_hat_pixel[c] = T::clamp(bg_b[c] * inv_beta_weight);
                }
            } else {
                *b_hat_pixel = *bg_pixel;
            }
        },
    );

    Ok((f_hat, b_hat))
}

/// Validates input parameters.
///
/// Checks the following conditions:
/// - Image and alpha matte dimensions match
/// - Radius is greater than 0 and odd
/// - Image is large enough for the given radius
/// - Iteration count is 1 or 2
fn validate_inputs_impl<T>(
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
            "radius must be > 0".to_owned(),
        ));
    }

    if radius % 2 == 0 {
        return Err(AlphaMaskError::InvalidParameter(
            "radius must be odd".to_owned(),
        ));
    }

    // Check if image is large enough for box filter with given radius
    // Note: The libblur crate requires stricter bounds than the theoretical minimum
    // For safety, the image should be larger than 2*radius+1, not just equal to it
    let min_dimension_required = 2 * radius + 1;
    let min_dimension = img_w.min(img_h);
    if min_dimension < min_dimension_required {
        return Err(AlphaMaskError::InvalidParameter(format!(
            "Image dimensions ({img_w}x{img_h}) are too small for radius {radius}. Minimum required: {min_dimension_required}x{min_dimension_required}"
        )));
    }

    if iterations == 0 || iterations > 2 {
        return Err(AlphaMaskError::InvalidParameter(
            "iterations must be 1 or 2".to_owned(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;
    use image::{Luma, Rgb};
    use imageproc::definitions::Image;

    // --- Test Setup ---
    fn create_basic_4x4_image_and_alpha() -> (Image<Rgb<u8>>, Image<Luma<u8>>) {
        let image = create_large_test_image(4, 4);
        let mut alpha: Image<Luma<u8>> = Image::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                let alpha_val = if x < 2 { 255 } else { 128 };
                alpha.put_pixel(x, y, Luma([alpha_val]));
            }
        }
        (image, alpha)
    }

    // --- Success Cases ---

    #[test]
    fn estimate_foreground_colors_succeeds_on_valid_input() {
        let (image, alpha) = create_basic_4x4_image_and_alpha();
        let result = estimate_foreground_colors(&image, &alpha, 1, 1);
        result.unwrap();
    }

    #[test]
    fn estimate_foreground_colors_returns_image_with_correct_dimensions() {
        let (image, alpha) = create_basic_4x4_image_and_alpha();
        let foreground = estimate_foreground_colors(&image, &alpha, 1, 1).unwrap();
        assert_eq!(foreground.dimensions(), image.dimensions());
    }

    #[test]
    fn estimate_foreground_colors_ext_trait_succeeds() {
        let image = create_large_test_image(4, 4);
        let alpha = Image::from_pixel(4, 4, Luma([200]));
        let result = image.estimate_foreground_colors(&alpha, 1);
        result.unwrap();
    }

    #[test]
    fn estimate_foreground_colors_ext_trait_returns_correct_dimensions() {
        let image = create_large_test_image(4, 4);
        let alpha = Image::from_pixel(4, 4, Luma([200]));
        let foreground = image.estimate_foreground_colors(&alpha, 1).unwrap();
        assert_eq!(foreground.dimensions(), (4, 4));
    }

    #[test]
    fn estimate_foreground_colors_succeeds_with_two_iterations() {
        let image = create_large_test_image(193, 193);
        let alpha = Image::from_fn(193, 193, |x, y| Luma([((x + y) * 255 / 384) as u8]));
        let result = estimate_foreground_colors(&image, &alpha, 91, 2);
        result.unwrap();
    }

    #[test]
    fn estimate_foreground_colors_with_fully_opaque_alpha_is_close_to_original() {
        let image = create_large_test_image(4, 4);
        let alpha = Image::from_pixel(4, 4, Luma([255]));
        let foreground = estimate_foreground_colors(&image, &alpha, 1, 1).unwrap();
        assert!(images_approx_equal(&image, &foreground, 10.0));
    }

    #[test]
    fn estimate_foreground_colors_succeeds_with_fully_transparent_alpha() {
        let image = create_large_test_image(4, 4);
        let alpha = Image::from_pixel(4, 4, Luma([0]));
        let result = estimate_foreground_colors(&image, &alpha, 1, 1);
        result.unwrap();
    }

    #[test]
    fn estimate_foreground_colors_succeeds_with_f32_type() {
        let image: Image<Rgb<f32>> = Image::from_pixel(4, 4, Rgb([0.5, 0.7, 0.9]));
        let alpha: Image<Luma<f32>> = Image::from_pixel(4, 4, Luma([0.5]));
        let result = estimate_foreground_colors(&image, &alpha, 1, 1);
        result.unwrap();
    }

    // --- Error Cases ---

    #[test]
    fn estimate_foreground_colors_errs_on_mismatched_dimensions() {
        let image = create_test_rgb_image();
        let alpha: Image<Luma<u8>> = Image::new(3, 3);
        let result = estimate_foreground_colors(&image, &alpha, 1, 1);
        match result {
            Err(AlphaMaskError::DimensionMismatch { expected, actual }) => {
                assert_eq!(expected, (2, 2));
                assert_eq!(actual, (3, 3));
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }

    #[test]
    fn estimate_foreground_colors_errs_on_zero_radius() {
        let image = create_test_rgb_image();
        let alpha = create_test_alpha_mask();
        let result = estimate_foreground_colors(&image, &alpha, 0, 1);
        assert!(matches!(
            result,
            Err(AlphaMaskError::InvalidParameter(msg)) if msg.contains("radius must be > 0")
        ));
    }

    #[test]
    fn estimate_foreground_colors_errs_on_even_radius() {
        let image = create_large_test_image(10, 10);
        let alpha = Image::from_pixel(10, 10, Luma([128]));
        let result = estimate_foreground_colors(&image, &alpha, 4, 1);
        assert!(matches!(
            result,
            Err(AlphaMaskError::InvalidParameter(msg)) if msg.contains("radius must be odd")
        ));
    }

    #[test]
    fn estimate_foreground_colors_errs_on_zero_iterations() {
        let (image, alpha) = create_basic_4x4_image_and_alpha();
        let result = estimate_foreground_colors(&image, &alpha, 1, 0);
        assert!(matches!(
            result,
            Err(AlphaMaskError::InvalidParameter(msg)) if msg.contains("iterations must be 1 or 2")
        ));
    }

    #[test]
    fn estimate_foreground_colors_errs_on_too_many_iterations() {
        let (image, alpha) = create_basic_4x4_image_and_alpha();
        let result = estimate_foreground_colors(&image, &alpha, 1, 3);
        assert!(matches!(
            result,
            Err(AlphaMaskError::InvalidParameter(msg)) if msg.contains("iterations must be 1 or 2")
        ));
    }

    #[test]
    fn estimate_foreground_colors_errs_on_radius_too_large_for_image() {
        let image = create_test_rgb_image(); // 2x2 image
        let alpha = create_test_alpha_mask();
        let result = estimate_foreground_colors(&image, &alpha, 3, 1);
        assert!(matches!(
            result,
            Err(AlphaMaskError::InvalidParameter(msg)) if msg.contains("too small for radius")
        ));
    }
}
