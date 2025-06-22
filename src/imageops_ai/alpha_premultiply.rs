use crate::error::ConvertColorError;
use crate::utils::{clamp_f32_to_primitive, normalize_alpha_with_max, validate_non_empty_image};
use image::{Luma, LumaA, Pixel, Primitive, Rgb, Rgba};
use imageproc::definitions::{Clamp, Image};
use imageproc::map::map_colors;

/// Trait for merging (premultiplying) alpha channel into color channels.
///
/// This operation multiplies each color channel by the alpha value,
/// effectively creating a premultiplied alpha image. The alpha channel
/// is discarded in the output.
///
/// # Alpha Premultiplication
///
/// Alpha premultiplication is the process of multiplying the color channels
/// by the alpha value, resulting in:
/// - Red' = Red × Alpha
/// - Green' = Green × Alpha  
/// - Blue' = Blue × Alpha
/// - Luminance' = Luminance × Alpha
///
/// This is commonly used in compositing operations and can help reduce
/// artifacts in image processing pipelines.
///
/// Note: This trait performs type conversion (e.g., Rgba -> Rgb). For in-place
/// premultiplication while keeping the alpha channel, use the `PremultiplyAlphaInPlace` trait.
pub trait AlphaPremultiply {
    type Output;

    /// Premultiplies color channels by alpha and returns an image without alpha channel.
    ///
    /// This consumes the original image.
    ///
    /// # Returns
    /// * `Ok(Self::Output)` - Successfully premultiplied image
    /// * `Err(ConvertColorError)` - If conversion fails
    ///
    /// # Panics
    /// This function does not panic. All error conditions are handled through `Result`.
    ///
    /// # Examples
    /// ```no_run
    /// use imageops_ai::AlphaPremultiply;
    /// use imageproc::definitions::Image;
    /// use image::Rgba;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let rgba_image: Image<Rgba<u8>> = Image::new(100, 100);
    /// let rgb_image = rgba_image.premultiply_alpha()?;
    /// # Ok(())
    /// # }
    /// ```
    fn premultiply_alpha(self) -> Result<Self::Output, ConvertColorError>;
}

/// Trait for in-place alpha premultiplication (keeps alpha channel).
///
/// This trait provides functionality to premultiply color channels with alpha
/// while preserving the alpha channel in the output.
pub trait PremultiplyAlphaInPlace {
    /// Premultiplies color channels by alpha in-place, keeping the alpha channel.
    ///
    /// This consumes the original image and returns a premultiplied version
    /// with the same pixel type.
    ///
    /// # Returns
    /// * `Ok(Self)` - Successfully premultiplied image with alpha
    /// * `Err(ConvertColorError)` - If conversion fails
    ///
    /// # Examples
    /// ```no_run
    /// use imageops_ai::PremultiplyAlphaInPlace;
    /// use imageproc::definitions::Image;
    /// use image::Rgba;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let rgba_image: Image<Rgba<u8>> = Image::new(100, 100);
    /// let premultiplied = rgba_image.premultiply_alpha_keep()?;
    /// # Ok(())
    /// # }
    /// ```
    fn premultiply_alpha_keep(self) -> Result<Self, ConvertColorError>
    where
        Self: Sized;

    /// Premultiplies color channels by alpha in-place, modifying the image.
    ///
    /// # Returns
    /// * `Ok(&mut Self)` - Successfully modified image
    /// * `Err(ConvertColorError)` - If conversion fails
    ///
    /// # Examples
    /// ```no_run
    /// use imageops_ai::PremultiplyAlphaInPlace;
    /// use imageproc::definitions::Image;
    /// use image::Rgba;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut rgba_image: Image<Rgba<u8>> = Image::new(100, 100);
    /// rgba_image.premultiply_alpha_keep_mut()?;
    /// # Ok(())
    /// # }
    /// ```
    fn premultiply_alpha_keep_mut(&mut self) -> Result<&mut Self, ConvertColorError>
    where
        Self: Sized;
}

/// Safe implementation for LumaA -> Luma conversion with alpha premultiplication
impl<S> AlphaPremultiply for Image<LumaA<S>>
where
    LumaA<S>: Pixel<Subpixel = S>,
    Luma<S>: Pixel<Subpixel = S>,
    S: Into<f32> + Clamp<f32> + Primitive,
{
    type Output = Image<Luma<S>>;

    fn premultiply_alpha(self) -> Result<Self::Output, ConvertColorError> {
        validate_image_dimensions(&self)?;

        let max_value = S::DEFAULT_MAX_VALUE.into();

        Ok(map_colors(&self, |pixel| {
            let LumaA([luminance, alpha]) = pixel;
            let alpha_normalized = normalize_alpha_with_max(alpha, max_value);
            let luminance_f32 = luminance.into();

            // Apply premultiplication with proper clamping
            let merged_f32 = luminance_f32 * alpha_normalized;
            let merged = clamp_f32_to_primitive(merged_f32);

            Luma([merged])
        }))
    }
}

/// Safe implementation for Rgba -> Rgb conversion with alpha premultiplication
impl<S> AlphaPremultiply for Image<Rgba<S>>
where
    Rgba<S>: Pixel<Subpixel = S>,
    Rgb<S>: Pixel<Subpixel = S>,
    S: Into<f32> + Clamp<f32> + Primitive,
{
    type Output = Image<Rgb<S>>;

    fn premultiply_alpha(self) -> Result<Self::Output, ConvertColorError> {
        validate_image_dimensions(&self)?;

        let max_value = S::DEFAULT_MAX_VALUE.into();

        Ok(map_colors(&self, |pixel| {
            let Rgba([red, green, blue, alpha]) = pixel;
            let alpha_normalized = normalize_alpha_with_max(alpha, max_value);

            // Convert to f32 and premultiply with optimized computation
            compute_premultiplied_rgb_pixel([red, green, blue], alpha_normalized)
        }))
    }
}

/// Implementation for LumaA images to premultiply while keeping alpha
impl<S> PremultiplyAlphaInPlace for Image<LumaA<S>>
where
    LumaA<S>: Pixel<Subpixel = S>,
    S: Into<f32> + Clamp<f32> + Primitive,
{
    fn premultiply_alpha_keep(self) -> Result<Self, ConvertColorError> {
        validate_image_dimensions(&self)?;

        let max_value = S::DEFAULT_MAX_VALUE.into();

        Ok(map_colors(&self, |pixel| {
            let LumaA([luminance, alpha]) = pixel;
            let alpha_normalized = normalize_alpha_with_max(alpha, max_value);
            let luminance_f32 = luminance.into();

            // Apply premultiplication with proper clamping
            let merged_f32 = luminance_f32 * alpha_normalized;
            let merged = clamp_f32_to_primitive(merged_f32);

            LumaA([merged, alpha])
        }))
    }

    fn premultiply_alpha_keep_mut(&mut self) -> Result<&mut Self, ConvertColorError> {
        validate_image_dimensions(self)?;

        let max_value = S::DEFAULT_MAX_VALUE.into();
        let (width, height) = self.dimensions();

        for y in 0..height {
            for x in 0..width {
                let pixel = self.get_pixel_mut(x, y);
                let LumaA([luminance, alpha]) = *pixel;
                let alpha_normalized = normalize_alpha_with_max(alpha, max_value);
                let luminance_f32 = luminance.into();

                let merged_f32 = luminance_f32 * alpha_normalized;
                let merged = clamp_f32_to_primitive(merged_f32);

                *pixel = LumaA([merged, alpha]);
            }
        }

        Ok(self)
    }
}

/// Implementation for Rgba images to premultiply while keeping alpha
impl<S> PremultiplyAlphaInPlace for Image<Rgba<S>>
where
    Rgba<S>: Pixel<Subpixel = S>,
    S: Into<f32> + Clamp<f32> + Primitive,
{
    fn premultiply_alpha_keep(self) -> Result<Self, ConvertColorError> {
        validate_image_dimensions(&self)?;

        let max_value = S::DEFAULT_MAX_VALUE.into();

        Ok(map_colors(&self, |pixel| {
            let Rgba([red, green, blue, alpha]) = pixel;
            let alpha_normalized = normalize_alpha_with_max(alpha, max_value);

            // Premultiply each channel
            let premultiplied =
                compute_premultiplied_rgb_pixel([red, green, blue], alpha_normalized);
            let Rgb([r_pre, g_pre, b_pre]) = premultiplied;

            Rgba([r_pre, g_pre, b_pre, alpha])
        }))
    }

    fn premultiply_alpha_keep_mut(&mut self) -> Result<&mut Self, ConvertColorError> {
        validate_image_dimensions(self)?;

        let max_value = S::DEFAULT_MAX_VALUE.into();
        let (width, height) = self.dimensions();

        for y in 0..height {
            for x in 0..width {
                let pixel = self.get_pixel_mut(x, y);
                let Rgba([red, green, blue, alpha]) = *pixel;
                let alpha_normalized = normalize_alpha_with_max(alpha, max_value);

                let premultiplied =
                    compute_premultiplied_rgb_pixel([red, green, blue], alpha_normalized);
                let Rgb([r_pre, g_pre, b_pre]) = premultiplied;

                *pixel = Rgba([r_pre, g_pre, b_pre, alpha]);
            }
        }

        Ok(self)
    }
}

/// Computes premultiplied RGB pixel with proper clamping
#[inline]
fn compute_premultiplied_rgb_pixel<S>(channels: [S; 3], alpha_normalized: f32) -> Rgb<S>
where
    S: Into<f32> + Primitive + Clamp<f32>,
{
    let mut result = [S::DEFAULT_MIN_VALUE; 3];

    for (i, &channel) in channels.iter().enumerate() {
        let channel_f32 = channel.into() * alpha_normalized;
        result[i] = clamp_f32_to_primitive(channel_f32);
    }

    Rgb(result)
}

/// Validates image dimensions for processing
fn validate_image_dimensions<P>(image: &Image<P>) -> Result<(), ConvertColorError>
where
    P: Pixel,
{
    let (width, height) = image.dimensions();

    validate_non_empty_image(width, height, "AlphaPremultiply").map_err(|_| {
        ConvertColorError::DimensionMismatch {
            expected_width: 1,
            expected_height: 1,
            actual_width: width,
            actual_height: height,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_premultiplied_rgb_pixel() {
        let channels = [200u8, 150u8, 100u8];
        let alpha = 0.5;
        let result = compute_premultiplied_rgb_pixel(channels, alpha);

        assert_eq!(result[0], 100); // 200 * 0.5
        assert_eq!(result[1], 75); // 150 * 0.5
        assert_eq!(result[2], 50); // 100 * 0.5
    }

    #[test]
    fn test_validate_image_dimensions() {
        let valid_image: Image<Rgb<u8>> = Image::new(10, 10);
        assert!(validate_image_dimensions(&valid_image).is_ok());

        let empty_image: Image<Rgb<u8>> = Image::new(0, 0);
        assert!(validate_image_dimensions(&empty_image).is_err());

        let invalid_width: Image<Rgb<u8>> = Image::new(0, 10);
        assert!(validate_image_dimensions(&invalid_width).is_err());

        let invalid_height: Image<Rgb<u8>> = Image::new(10, 0);
        assert!(validate_image_dimensions(&invalid_height).is_err());
    }

    #[test]
    fn test_alpha_premultiply_luma() {
        let mut image: Image<LumaA<u8>> = Image::new(2, 2);
        image.put_pixel(0, 0, LumaA([200, 255])); // Full opacity
        image.put_pixel(1, 0, LumaA([200, 127])); // Half opacity
        image.put_pixel(0, 1, LumaA([200, 0])); // Transparent
        image.put_pixel(1, 1, LumaA([100, 255])); // Full opacity, different value

        let result = image.premultiply_alpha().unwrap();

        assert_eq!(result.get_pixel(0, 0)[0], 200); // 200 * 1.0
        assert_eq!(result.get_pixel(1, 0)[0], 99); // 200 * 0.498
        assert_eq!(result.get_pixel(0, 1)[0], 0); // 200 * 0.0
        assert_eq!(result.get_pixel(1, 1)[0], 100); // 100 * 1.0
    }

    #[test]
    fn test_alpha_premultiply_rgba() {
        let mut image: Image<Rgba<u8>> = Image::new(2, 2);
        image.put_pixel(0, 0, Rgba([200, 150, 100, 255])); // Full opacity
        image.put_pixel(1, 0, Rgba([200, 150, 100, 127])); // Half opacity
        image.put_pixel(0, 1, Rgba([200, 150, 100, 0])); // Transparent
        image.put_pixel(1, 1, Rgba([100, 50, 25, 255])); // Full opacity, different values

        let result = image.premultiply_alpha().unwrap();

        // Full opacity case
        let pixel_00 = result.get_pixel(0, 0);
        assert_eq!(pixel_00[0], 200);
        assert_eq!(pixel_00[1], 150);
        assert_eq!(pixel_00[2], 100);

        // Half opacity case
        let pixel_10 = result.get_pixel(1, 0);
        assert_eq!(pixel_10[0], 99); // 200 * 0.498
        assert_eq!(pixel_10[1], 74); // 150 * 0.498
        assert_eq!(pixel_10[2], 49); // 100 * 0.498

        // Transparent case
        let pixel_01 = result.get_pixel(0, 1);
        assert_eq!(pixel_01[0], 0);
        assert_eq!(pixel_01[1], 0);
        assert_eq!(pixel_01[2], 0);

        // Full opacity, different values
        let pixel_11 = result.get_pixel(1, 1);
        assert_eq!(pixel_11[0], 100);
        assert_eq!(pixel_11[1], 50);
        assert_eq!(pixel_11[2], 25);
    }

    #[test]
    fn test_premultiply_alpha_keep_luma() {
        use crate::PremultiplyAlphaInPlace;

        let mut image: Image<LumaA<u8>> = Image::new(2, 2);
        image.put_pixel(0, 0, LumaA([200, 255])); // Full opacity
        image.put_pixel(1, 0, LumaA([200, 127])); // Half opacity

        let result = image.clone().premultiply_alpha_keep().unwrap();

        // Check that luminance is premultiplied but alpha is preserved
        assert_eq!(result.get_pixel(0, 0).0, [200, 255]); // 200 * 1.0, alpha preserved
        assert_eq!(result.get_pixel(1, 0).0, [99, 127]); // 200 * 0.498, alpha preserved
    }

    #[test]
    fn test_premultiply_alpha_keep_mut_rgba() {
        use crate::PremultiplyAlphaInPlace;

        let mut image: Image<Rgba<u8>> = Image::new(2, 2);
        image.put_pixel(0, 0, Rgba([200, 150, 100, 255])); // Full opacity
        image.put_pixel(1, 0, Rgba([200, 150, 100, 127])); // Half opacity

        let mut image_copy = image.clone();
        image_copy.premultiply_alpha_keep_mut().unwrap();

        // Check that colors are premultiplied but alpha is preserved
        assert_eq!(image_copy.get_pixel(0, 0).0, [200, 150, 100, 255]); // Full opacity unchanged
        assert_eq!(image_copy.get_pixel(1, 0).0, [99, 74, 49, 127]); // Premultiplied, alpha preserved
    }
}
