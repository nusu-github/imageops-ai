use crate::error::ConvertColorError;
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
pub trait AlphaPremultiply {
    type Output;

    /// Premultiplies color channels by alpha and returns an image without alpha channel.
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
    /// use imageops_ai::alpha_premultiply::AlphaPremultiply;
    /// use imageproc::definitions::Image;
    /// use image::Rgba;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let rgba_image: Image<Rgba<u8>> = Image::new(100, 100);
    /// let rgb_image = rgba_image.premultiply_alpha()?;
    /// # Ok(())
    /// # }
    /// ```
    fn premultiply_alpha(&self) -> Result<Self::Output, ConvertColorError>;
}

/// Safe implementation for LumaA -> Luma conversion with alpha premultiplication
impl<S> AlphaPremultiply for Image<LumaA<S>>
where
    LumaA<S>: Pixel<Subpixel = S>,
    Luma<S>: Pixel<Subpixel = S>,
    S: Into<f32> + Clamp<f32> + Primitive,
{
    type Output = Image<Luma<S>>;

    fn premultiply_alpha(&self) -> Result<Self::Output, ConvertColorError> {
        validate_image_dimensions(self)?;

        let max_value = S::DEFAULT_MAX_VALUE.into();

        Ok(map_colors(self, |pixel| {
            let LumaA([luminance, alpha]) = pixel;
            let alpha_normalized = normalize_alpha(alpha, max_value);
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

    fn premultiply_alpha(&self) -> Result<Self::Output, ConvertColorError> {
        validate_image_dimensions(self)?;

        let max_value = S::DEFAULT_MAX_VALUE.into();

        Ok(map_colors(self, |pixel| {
            let Rgba([red, green, blue, alpha]) = pixel;
            let alpha_normalized = normalize_alpha(alpha, max_value);

            // Convert to f32 and premultiply with optimized computation
            compute_premultiplied_rgb_pixel([red, green, blue], alpha_normalized)
        }))
    }
}

/// Normalizes alpha value to [0.0, 1.0] range
#[inline]
fn normalize_alpha<S>(alpha: S, max_value: f32) -> f32
where
    S: Into<f32> + Primitive,
{
    alpha.into() / max_value
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

/// Clamps f32 to primitive type range using imageproc's Clamp trait
#[inline]
fn clamp_f32_to_primitive<S>(value: f32) -> S
where
    S: Primitive + Clamp<f32>,
{
    S::clamp(value)
}

/// Validates image dimensions for processing
fn validate_image_dimensions<P>(image: &Image<P>) -> Result<(), ConvertColorError>
where
    P: Pixel,
{
    let (width, height) = image.dimensions();

    if width == 0 || height == 0 {
        return Err(ConvertColorError::DimensionMismatch {
            expected_width: 1,
            expected_height: 1,
            actual_width: width,
            actual_height: height,
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_alpha() {
        assert_eq!(normalize_alpha(0u8, 255.0), 0.0);
        assert_eq!(normalize_alpha(127u8, 255.0), 127.0 / 255.0);
        assert_eq!(normalize_alpha(255u8, 255.0), 1.0);
    }

    #[test]
    fn test_clamp_f32_to_primitive() {
        assert_eq!(clamp_f32_to_primitive::<u8>(-10.0), 0);
        assert_eq!(clamp_f32_to_primitive::<u8>(0.0), 0);
        assert_eq!(clamp_f32_to_primitive::<u8>(127.5), 127);
        assert_eq!(clamp_f32_to_primitive::<u8>(255.0), 255);
        assert_eq!(clamp_f32_to_primitive::<u8>(300.0), 255);
    }

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
}
