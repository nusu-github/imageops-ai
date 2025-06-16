use crate::imageops_ai::box_filter::BoxFilter;
use crate::Image;
use image::{ImageBuffer, Luma, LumaA, Pixel, Primitive, Rgb, Rgb32FImage, Rgba};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConvertColorError {
    #[error("Value {0} is out of range for the target type")]
    ValueOutOfRange(f32),
    #[error("Image dimensions mismatch: expected {expected_width}x{expected_height}, got {actual_width}x{actual_height}")]
    DimensionMismatch {
        expected_width: u32,
        expected_height: u32,
        actual_width: u32,
        actual_height: u32,
    },
    #[error("Failed to create image buffer")]
    BufferCreationFailed,
    #[error("Box filter operation failed: {0}")]
    BoxFilterError(#[from] crate::BoxFilterError),
}

/// Trait for merging (premultiplying) alpha channel into color channels.
///
/// This operation multiplies each color channel by the alpha value,
/// effectively creating a premultiplied alpha image. The alpha channel
/// is discarded in the output.
pub trait AlphaPremultiply {
    type Output;

    /// Premultiplies color channels by alpha and returns an image without alpha channel.
    ///
    /// # Errors
    ///
    /// Returns an error if the conversion results in values outside the valid range
    /// for the pixel type.
    fn premultiply_alpha(&self) -> Result<Self::Output, ConvertColorError>;
}

/// Safe implementation for LumaA -> Luma conversion with alpha premultiplication
impl<S> AlphaPremultiply for Image<LumaA<S>>
where
    LumaA<S>: Pixel<Subpixel = S>,
    Luma<S>: Pixel<Subpixel = S>,
    S: Primitive + 'static,
    f32: From<S>,
{
    type Output = Image<Luma<S>>;

    fn premultiply_alpha(&self) -> Result<Self::Output, ConvertColorError> {
        let max_value = f32::from(S::DEFAULT_MAX_VALUE);
        let mut output = ImageBuffer::new(self.width(), self.height());

        for (x, y, pixel) in self.enumerate_pixels() {
            let LumaA([luminance, alpha]) = *pixel;
            let alpha_normalized = f32::from(alpha) / max_value;
            let luminance_f32 = f32::from(luminance);

            // Clamp the result to ensure it's within valid range
            let merged_f32 = (luminance_f32 * alpha_normalized).clamp(0.0, max_value);

            // Safe conversion with fallback
            let merged = S::from(merged_f32).unwrap_or_else(|| {
                // If exact conversion fails, use the nearest valid value
                if merged_f32 >= max_value {
                    S::DEFAULT_MAX_VALUE
                } else {
                    S::DEFAULT_MIN_VALUE
                }
            });

            output.put_pixel(x, y, Luma([merged]));
        }

        Ok(output)
    }
}

/// Safe implementation for Rgba -> Rgb conversion with alpha premultiplication
impl<S> AlphaPremultiply for Image<Rgba<S>>
where
    Rgba<S>: Pixel<Subpixel = S>,
    Rgb<S>: Pixel<Subpixel = S>,
    S: Primitive + 'static,
    f32: From<S>,
{
    type Output = Image<Rgb<S>>;

    fn premultiply_alpha(&self) -> Result<Self::Output, ConvertColorError> {
        let max_value = f32::from(S::DEFAULT_MAX_VALUE);
        let mut output = ImageBuffer::new(self.width(), self.height());

        for (x, y, pixel) in self.enumerate_pixels() {
            let Rgba([red, green, blue, alpha]) = *pixel;
            let alpha_normalized = f32::from(alpha) / max_value;

            // Convert to f32 and premultiply
            let red_f32 = f32::from(red) * alpha_normalized;
            let green_f32 = f32::from(green) * alpha_normalized;
            let blue_f32 = f32::from(blue) * alpha_normalized;

            // Clamp and convert back safely
            let clamp_and_convert = |value: f32| -> S {
                let clamped = value.clamp(0.0, max_value);
                S::from(clamped).unwrap_or_else(|| {
                    if clamped >= max_value {
                        S::DEFAULT_MAX_VALUE
                    } else {
                        S::DEFAULT_MIN_VALUE
                    }
                })
            };

            output.put_pixel(
                x,
                y,
                Rgb([
                    clamp_and_convert(red_f32),
                    clamp_and_convert(green_f32),
                    clamp_and_convert(blue_f32),
                ]),
            );
        }

        Ok(output)
    }
}

/// Trait for estimating foreground colors from images with alpha masks.
///
/// This implements a foreground color estimation algorithm that uses
/// guided filtering to separate foreground and background colors.
pub trait ForegroundEstimator<S>
where
    Rgb<S>: Pixel<Subpixel = S>,
    Luma<S>: Pixel<Subpixel = S>,
    S: Primitive + 'static,
    f32: From<S>,
{
    /// Estimates foreground colors using the provided alpha mask.
    ///
    /// # Parameters
    ///
    /// * `mask` - Alpha mask indicating foreground (high values) and background (low values)
    /// * `r` - Radius for the box filter used in the estimation process
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Image and mask dimensions don't match
    /// - Box filter operations fail
    /// - Image buffer creation fails
    fn estimate_foreground<SM>(
        self,
        mask: &Image<Luma<SM>>,
        r: u32,
    ) -> Result<Image<Rgb<S>>, ConvertColorError>
    where
        SM: Primitive + 'static,
        f32: From<SM>;
}

impl<S> ForegroundEstimator<S> for Image<Rgb<S>>
where
    Rgb<S>: Pixel<Subpixel = S>,
    Luma<S>: Pixel<Subpixel = S>,
    S: Primitive + 'static,
    f32: From<S>,
{
    fn estimate_foreground<SM>(
        self,
        mask: &Image<Luma<SM>>,
        r: u32,
    ) -> Result<Self, ConvertColorError>
    where
        SM: Primitive + 'static,
        f32: From<SM>,
    {
        // Validate dimensions
        if self.width() != mask.width() || self.height() != mask.height() {
            return Err(ConvertColorError::DimensionMismatch {
                expected_width: self.width(),
                expected_height: self.height(),
                actual_width: mask.width(),
                actual_height: mask.height(),
            });
        }

        let max = f32::from(S::DEFAULT_MAX_VALUE);

        // Convert image to normalized f32
        let image_f32: Vec<f32> = self.iter().map(|x| f32::from(*x) / max).collect();

        let image = ImageBuffer::from_raw(self.width(), self.height(), image_f32)
            .ok_or(ConvertColorError::BufferCreationFailed)?;

        // Convert mask to normalized f32
        let mask_max = f32::from(SM::DEFAULT_MAX_VALUE);
        let alpha_f32: Vec<f32> = mask.iter().map(|x| f32::from(*x) / mask_max).collect();

        let alpha = ImageBuffer::from_raw(mask.width(), mask.height(), alpha_f32)
            .ok_or(ConvertColorError::BufferCreationFailed)?;

        // Perform estimation
        let estimated = estimate(&image, &alpha, r)?;

        // Convert back to original type
        let result_data: Vec<S> = estimated
            .iter()
            .map(|&x| {
                let scaled = (x * max).clamp(0.0, max);
                S::from(scaled).unwrap_or_else(|| {
                    if scaled >= max {
                        S::DEFAULT_MAX_VALUE
                    } else {
                        S::DEFAULT_MIN_VALUE
                    }
                })
            })
            .collect();

        ImageBuffer::from_raw(estimated.width(), estimated.height(), result_data)
            .ok_or(ConvertColorError::BufferCreationFailed)
    }
}

/// Performs foreground estimation using iterative guided filtering.
///
/// This uses a two-pass approach:
/// 1. Initial estimation with radius `r`
/// 2. Refinement pass with radius 6
const REFINEMENT_RADIUS: u32 = 6;

fn estimate(
    image: &Rgb32FImage,
    alpha: &Image<Luma<f32>>,
    r: u32,
) -> Result<Rgb32FImage, ConvertColorError> {
    let (f, blur_b) = blur_fusion_estimator(image, image, image, alpha, r)?;
    let (f, _) = blur_fusion_estimator(image, &f, &blur_b, alpha, REFINEMENT_RADIUS)?;
    Ok(f)
}

/// Guided filter-based foreground/background separation.
///
/// This implements a blur-based fusion estimator that separates
/// foreground and background components using the alpha mask as guidance.
const MIN_DENOMINATOR: f32 = 1e-5;

fn blur_fusion_estimator(
    image: &Rgb32FImage,
    f: &Rgb32FImage,
    b: &Rgb32FImage,
    alpha: &Image<Luma<f32>>,
    r: u32,
) -> Result<(Rgb32FImage, Rgb32FImage), ConvertColorError> {
    // Blur the alpha channel
    let blurred_alpha = alpha.box_filter(r, r)?;

    // Compute f * alpha and blur it
    let fa_data: Vec<f32> = f
        .pixels()
        .zip(alpha.pixels())
        .flat_map(|(f_pixel, alpha_pixel)| {
            let Rgb([f_r, f_g, f_b]) = f_pixel;
            let Luma([a]) = alpha_pixel;
            [f_r * a, f_g * a, f_b * a]
        })
        .collect();

    let blurred_fa = ImageBuffer::<Rgb<f32>, Vec<f32>>::from_raw(f.width(), f.height(), fa_data)
        .ok_or(ConvertColorError::BufferCreationFailed)?
        .box_filter(r, r)?;

    // Compute blurred foreground by dividing by blurred alpha
    let blurred_f_data: Vec<f32> = blurred_fa
        .pixels()
        .zip(blurred_alpha.pixels())
        .flat_map(|(fa_pixel, alpha_pixel)| {
            let Rgb([fa_r, fa_g, fa_b]) = fa_pixel;
            let Luma([a]) = alpha_pixel;
            let denominator = a + MIN_DENOMINATOR;
            [fa_r / denominator, fa_g / denominator, fa_b / denominator]
        })
        .collect();

    let blurred_f = ImageBuffer::from_raw(blurred_fa.width(), blurred_fa.height(), blurred_f_data)
        .ok_or(ConvertColorError::BufferCreationFailed)?;

    // Compute b * (1 - alpha) and blur it
    let b1a_data: Vec<f32> = b
        .pixels()
        .zip(alpha.pixels())
        .flat_map(|(b_pixel, alpha_pixel)| {
            let Rgb([b_r, b_g, b_b]) = b_pixel;
            let Luma([a]) = alpha_pixel;
            let one_minus_a = 1.0 - a;
            [b_r * one_minus_a, b_g * one_minus_a, b_b * one_minus_a]
        })
        .collect();

    let blurred_b1a: Image<Rgb<f32>> = ImageBuffer::<Rgb<f32>, Vec<f32>>::from_raw(b.width(), b.height(), b1a_data)
        .ok_or(ConvertColorError::BufferCreationFailed)?
        .box_filter(r, r)?;

    // Compute blurred background by dividing by (1 - blurred_alpha)
    let blurred_b_data: Vec<f32> = blurred_b1a
        .pixels()
        .zip(blurred_alpha.pixels())
        .flat_map(|(b1a_pixel, alpha_pixel)| {
            let Rgb([b_r, b_g, b_b]) = b1a_pixel;
            let Luma([a]) = alpha_pixel;
            let denominator = (1.0 - a) + MIN_DENOMINATOR;
            [b_r / denominator, b_g / denominator, b_b / denominator]
        })
        .collect();

    let blurred_b =
        ImageBuffer::from_raw(blurred_b1a.width(), blurred_b1a.height(), blurred_b_data)
            .ok_or(ConvertColorError::BufferCreationFailed)?;

    // Update foreground estimation
    let updated_f_data: Vec<f32> = f
        .pixels()
        .zip(image.pixels())
        .zip(alpha.pixels())
        .zip(blurred_f.pixels())
        .zip(blurred_b.pixels())
        .flat_map(
            |((((f_pixel, image_pixel), alpha_pixel), f_blurred), b_blurred)| {
                let Rgb([f_r, f_g, f_b]) = *f_pixel;
                let Rgb([i_r, i_g, i_b]) = *image_pixel;
                let Luma([a]) = alpha_pixel;
                let Rgb([fb_r, fb_g, fb_b]) = *f_blurred;
                let Rgb([bb_r, bb_g, bb_b]) = *b_blurred;

                let a_safe = a + MIN_DENOMINATOR;

                // Update formula: f_new = f + a * (I - a * f_blur - (1-a) * b_blur)
                let update_channel = |f: f32, i: f32, fb: f32, bb: f32| -> f32 {
                    let updated = f + a_safe * (i - a * fb - (1.0 - a) * bb);
                    updated.clamp(0.0, 1.0)
                };

                [
                    update_channel(f_r, i_r, fb_r, bb_r),
                    update_channel(f_g, i_g, fb_g, bb_g),
                    update_channel(f_b, i_b, fb_b, bb_b),
                ]
            },
        )
        .collect();

    let updated_f = ImageBuffer::from_raw(blurred_f.width(), blurred_f.height(), updated_f_data)
        .ok_or(ConvertColorError::BufferCreationFailed)?;

    Ok((updated_f, blurred_b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Luma, LumaA, Rgb, Rgba};

    #[test]
    fn test_merge_alpha_luma() {
        let mut img = ImageBuffer::new(2, 2);
        img.put_pixel(0, 0, LumaA([128u8, 255])); // Full alpha
        img.put_pixel(1, 0, LumaA([128u8, 128])); // Half alpha
        img.put_pixel(0, 1, LumaA([255u8, 0])); // Zero alpha
        img.put_pixel(1, 1, LumaA([200u8, 200])); // Partial alpha

        let result = img.premultiply_alpha().unwrap();

        assert_eq!(*result.get_pixel(0, 0), Luma([128u8])); // 128 * 1.0
        assert_eq!(*result.get_pixel(1, 0), Luma([64u8])); // 128 * 0.5
        assert_eq!(*result.get_pixel(0, 1), Luma([0u8])); // 255 * 0.0

        // 200 * (200/255) ≈ 156.86, rounded to 157
        let expected = (200.0 * (200.0 / 255.0)) as u8;
        assert!((result.get_pixel(1, 1).0[0] as i32 - expected as i32).abs() <= 1);
    }

    #[test]
    fn test_merge_alpha_rgba() {
        let mut img = ImageBuffer::new(1, 2);
        img.put_pixel(0, 0, Rgba([255u8, 128, 64, 255])); // Full alpha
        img.put_pixel(0, 1, Rgba([255u8, 128, 64, 128])); // Half alpha

        let result = img.premultiply_alpha().unwrap();

        assert_eq!(*result.get_pixel(0, 0), Rgb([255u8, 128, 64]));
        // Half alpha: each channel * 0.5
        assert_eq!(*result.get_pixel(0, 1), Rgb([128u8, 64, 32]));
    }

    #[test]
    fn test_dimension_mismatch() {
        let image: Image<Rgb<u8>> = ImageBuffer::new(10, 10);
        let mask: Image<Luma<u8>> = ImageBuffer::new(5, 5);

        let result = image.estimate_foreground(&mask, 3);
        assert!(matches!(
            result,
            Err(ConvertColorError::DimensionMismatch { .. })
        ));
    }
}
