use image::{GenericImageView, ImageBuffer, Luma, LumaA, Pixel, Primitive, Rgb, Rgba};
use imageproc::{
    definitions::{Clamp, Image},
    map::map_colors,
};
use itertools::Itertools;

use crate::{error::ColorConversionError, utils::normalize_alpha_with_max};

/// Trait for merging (premultiplying) alpha channel into color channels and discarding alpha.
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
/// premultiplication while keeping the alpha channel, use the `PremultiplyAlphaAndKeepExt` trait.
pub trait PremultiplyAlphaAndDropExt {
    type Output;

    /// Premultiplies color channels by alpha and returns an image without alpha channel.
    ///
    /// This consumes the original image.
    ///
    /// # Returns
    /// * `Ok(Self::Output)` - Successfully premultiplied image
    /// * `Err(ColorConversionError)` - If conversion fails
    ///
    /// # Panics
    /// This function does not panic. All error conditions are handled through `Result`.
    ///
    /// # Examples
    /// ```no_run
    /// use imageops_ai::PremultiplyAlphaAndDropExt;
    /// use imageproc::definitions::Image;
    /// use image::Rgba;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let rgba_image: Image<Rgba<u8>> = Image::new(100, 100);
    /// let rgb_image = rgba_image.premultiply_alpha_and_drop()?;
    /// # Ok(())
    /// # }
    /// ```
    fn premultiply_alpha_and_drop(self) -> Result<Self::Output, ColorConversionError>;
}

/// Trait for alpha premultiplication that keeps the alpha channel.
///
/// This trait provides functionality to premultiply color channels with alpha
/// while preserving the alpha channel in the output.
pub trait PremultiplyAlphaAndKeepExt {
    /// Premultiplies color channels by alpha, keeping the alpha channel.
    ///
    /// This consumes the original image and returns a premultiplied version
    /// with the same pixel type.
    ///
    /// # Returns
    /// * `Ok(Self)` - Successfully premultiplied image with alpha
    /// * `Err(ColorConversionError)` - If conversion fails
    ///
    /// # Examples
    /// ```no_run
    /// use imageops_ai::PremultiplyAlphaAndKeepExt;
    /// use imageproc::definitions::Image;
    /// use image::Rgba;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let rgba_image: Image<Rgba<u8>> = Image::new(100, 100);
    /// let premultiplied = rgba_image.premultiply_alpha_and_keep()?;
    /// # Ok(())
    /// # }
    /// ```
    fn premultiply_alpha_and_keep(self) -> Result<Self, ColorConversionError>
    where
        Self: Sized;

    /// Premultiplies color channels by alpha in-place, modifying the image.
    ///
    /// # Returns
    /// * `Ok(&mut Self)` - Successfully modified image
    /// * `Err(ColorConversionError)` - If conversion fails
    ///
    /// # Examples
    /// ```no_run
    /// use imageops_ai::PremultiplyAlphaAndKeepExt;
    /// use imageproc::definitions::Image;
    /// use image::Rgba;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut rgba_image: Image<Rgba<u8>> = Image::new(100, 100);
    /// rgba_image.premultiply_alpha_and_keep_mut()?;
    /// # Ok(())
    /// # }
    /// ```
    fn premultiply_alpha_and_keep_mut(&mut self) -> Result<&mut Self, ColorConversionError>
    where
        Self: Sized;
}

/// Generic fallback implementation for LumaA -> Luma conversion with alpha premultiplication
fn premultiply_lumaa_impl<S>(
    image: &Image<LumaA<S>>,
) -> Result<Image<Luma<S>>, ColorConversionError>
where
    S: Clamp<f32> + Primitive,
    f32: From<S>,
{
    validate_image_dimensions(image)?;

    let max_value = f32::from(S::DEFAULT_MAX_VALUE);

    Ok(map_colors(image, |pixel| {
        let LumaA([luminance, alpha]) = pixel;
        let alpha_normalized = normalize_alpha_with_max(alpha, max_value);
        let luminance = f32::from(luminance);

        // Apply premultiplication with proper clamping
        let premultiplied = luminance * alpha_normalized;
        let clamped = S::clamp(premultiplied);

        Luma([clamped])
    }))
}

/// Generic fallback implementation for Rgba -> Rgb conversion with alpha premultiplication
fn premultiply_rgba_impl<S>(image: &Image<Rgba<S>>) -> Result<Image<Rgb<S>>, ColorConversionError>
where
    Rgba<S>: Pixel<Subpixel = S>,
    Rgb<S>: Pixel<Subpixel = S>,
    S: Clamp<f32> + Primitive,
    f32: From<S>,
{
    validate_image_dimensions(image)?;

    let max_value = f32::from(S::DEFAULT_MAX_VALUE);

    Ok(map_colors(image, |pixel| {
        let Rgba([red, green, blue, alpha]) = pixel;
        let alpha_normalized = normalize_alpha_with_max(alpha, max_value);

        // Convert to f32 and premultiply with optimized computation
        compute_premultiplied_rgb_impl([red, green, blue], alpha_normalized)
    }))
}

/// Implementation for f32 LumaA -> Luma conversion
impl PremultiplyAlphaAndDropExt for Image<LumaA<f32>> {
    type Output = Image<Luma<f32>>;

    fn premultiply_alpha_and_drop(self) -> Result<Self::Output, ColorConversionError> {
        premultiply_lumaa_impl(&self)
    }
}

/// Implementation for u16 LumaA -> Luma conversion
impl PremultiplyAlphaAndDropExt for Image<LumaA<u16>> {
    type Output = Image<Luma<u16>>;

    fn premultiply_alpha_and_drop(self) -> Result<Self::Output, ColorConversionError> {
        map_pixels_to_new_image(&self, |&LumaA([luminance, alpha])| {
            Luma([premultiply_u16(luminance, alpha)])
        })
    }
}

/// Optimized implementation for u8 LumaA -> Luma conversion using LUT
impl PremultiplyAlphaAndDropExt for Image<LumaA<u8>> {
    type Output = Image<Luma<u8>>;

    fn premultiply_alpha_and_drop(self) -> Result<Self::Output, ColorConversionError> {
        map_pixels_to_new_image(&self, |&LumaA([luminance, alpha])| {
            Luma([premultiply_u8(luminance, alpha)])
        })
    }
}

/// Implementation for f32 Rgba -> Rgb conversion
impl PremultiplyAlphaAndDropExt for Image<Rgba<f32>> {
    type Output = Image<Rgb<f32>>;

    fn premultiply_alpha_and_drop(self) -> Result<Self::Output, ColorConversionError> {
        premultiply_rgba_impl(&self)
    }
}

/// Optimized implementation for u8 Rgba -> Rgb conversion using LUT
impl PremultiplyAlphaAndDropExt for Image<Rgba<u8>> {
    type Output = Image<Rgb<u8>>;

    fn premultiply_alpha_and_drop(self) -> Result<Self::Output, ColorConversionError> {
        map_pixels_to_new_image(&self, |&Rgba([red, green, blue, alpha])| {
            Rgb(premultiply_rgb_u8([red, green, blue], alpha))
        })
    }
}

/// Optimized implementation for u16 Rgba -> Rgb conversion using integer arithmetic
impl PremultiplyAlphaAndDropExt for Image<Rgba<u16>> {
    type Output = Image<Rgb<u16>>;

    fn premultiply_alpha_and_drop(self) -> Result<Self::Output, ColorConversionError> {
        map_pixels_to_new_image(&self, |&Rgba([red, green, blue, alpha])| {
            Rgb([
                premultiply_u16(red, alpha),
                premultiply_u16(green, alpha),
                premultiply_u16(blue, alpha),
            ])
        })
    }
}

/// Optimized implementation for u32 Rgba -> Rgb conversion using integer arithmetic
impl PremultiplyAlphaAndDropExt for Image<Rgba<u32>> {
    type Output = Image<Rgb<u32>>;

    fn premultiply_alpha_and_drop(self) -> Result<Self::Output, ColorConversionError> {
        map_pixels_to_new_image(&self, |&Rgba([red, green, blue, alpha])| {
            Rgb([
                premultiply_u32(red, alpha),
                premultiply_u32(green, alpha),
                premultiply_u32(blue, alpha),
            ])
        })
    }
}

/// Implementation for f32 LumaA images to premultiply while keeping alpha
impl PremultiplyAlphaAndKeepExt for Image<LumaA<f32>> {
    fn premultiply_alpha_and_keep(self) -> Result<Self, ColorConversionError> {
        validate_image_dimensions(&self)?;

        let max_value = f32::DEFAULT_MAX_VALUE;

        Ok(map_colors(&self, |pixel| {
            let LumaA([luminance, alpha]) = pixel;
            let alpha_normalized = normalize_alpha_with_max(alpha, max_value);
            let luminance: f32 = luminance;

            // Apply premultiplication with proper clamping
            let premultiplied: f32 = luminance * alpha_normalized;
            let clamped = premultiplied.clamp(0.0, f32::DEFAULT_MAX_VALUE);

            LumaA([clamped, alpha])
        }))
    }

    fn premultiply_alpha_and_keep_mut(&mut self) -> Result<&mut Self, ColorConversionError> {
        validate_image_dimensions(self)?;

        let max_value = f32::DEFAULT_MAX_VALUE;

        self.pixels_mut().for_each(|pixel| {
            let LumaA([luminance, alpha]) = *pixel;
            let alpha_normalized = normalize_alpha_with_max(alpha, max_value);
            let premultiplied = (luminance * alpha_normalized).clamp(0.0, max_value);
            *pixel = LumaA([premultiplied, alpha]);
        });

        Ok(self)
    }
}

/// Optimized implementation for u8 LumaA images using LUT
impl PremultiplyAlphaAndKeepExt for Image<LumaA<u8>> {
    fn premultiply_alpha_and_keep(self) -> Result<Self, ColorConversionError> {
        map_pixels_to_new_image(&self, |&LumaA([luminance, alpha])| {
            LumaA([premultiply_u8(luminance, alpha), alpha])
        })
    }

    fn premultiply_alpha_and_keep_mut(&mut self) -> Result<&mut Self, ColorConversionError> {
        validate_image_dimensions(self)?;

        self.pixels_mut().for_each(|pixel| {
            let LumaA([luminance, alpha]) = *pixel;
            *pixel = LumaA([premultiply_u8(luminance, alpha), alpha]);
        });

        Ok(self)
    }
}

/// Implementation for f32 Rgba images to premultiply while keeping alpha
impl PremultiplyAlphaAndKeepExt for Image<Rgba<f32>> {
    fn premultiply_alpha_and_keep(self) -> Result<Self, ColorConversionError> {
        validate_image_dimensions(&self)?;

        let max_value = f32::DEFAULT_MAX_VALUE;

        Ok(map_colors(&self, |pixel| {
            let Rgba([red, green, blue, alpha]) = pixel;
            let alpha_normalized = normalize_alpha_with_max(alpha, max_value);

            // Premultiply each channel
            let premultiplied =
                compute_premultiplied_rgb_impl([red, green, blue], alpha_normalized);
            let Rgb([r_pre, g_pre, b_pre]) = premultiplied;

            Rgba([r_pre, g_pre, b_pre, alpha])
        }))
    }

    fn premultiply_alpha_and_keep_mut(&mut self) -> Result<&mut Self, ColorConversionError> {
        validate_image_dimensions(self)?;

        let max_value = f32::DEFAULT_MAX_VALUE;

        self.pixels_mut().for_each(|pixel| {
            let Rgba([red, green, blue, alpha]) = *pixel;
            let alpha_normalized = normalize_alpha_with_max(alpha, max_value);

            let Rgb([r_pre, g_pre, b_pre]) =
                compute_premultiplied_rgb_impl([red, green, blue], alpha_normalized);
            *pixel = Rgba([r_pre, g_pre, b_pre, alpha]);
        });

        Ok(self)
    }
}

/// Optimized implementation for u8 Rgba images using LUT
impl PremultiplyAlphaAndKeepExt for Image<Rgba<u8>> {
    fn premultiply_alpha_and_keep(self) -> Result<Self, ColorConversionError> {
        map_pixels_to_new_image(&self, |&Rgba([red, green, blue, alpha])| {
            let [r, g, b] = premultiply_rgb_u8([red, green, blue], alpha);
            Rgba([r, g, b, alpha])
        })
    }

    fn premultiply_alpha_and_keep_mut(&mut self) -> Result<&mut Self, ColorConversionError> {
        validate_image_dimensions(self)?;

        self.pixels_mut().for_each(|pixel| {
            let Rgba([red, green, blue, alpha]) = *pixel;
            let [r, g, b] = premultiply_rgb_u8([red, green, blue], alpha);
            *pixel = Rgba([r, g, b, alpha]);
        });

        Ok(self)
    }
}

/// Compile-time Look-Up Table generator for u8 alpha premultiplication
const fn generate_alpha_lut() -> [[u8; 256]; 256] {
    let mut lut = [[0u8; 256]; 256];
    let mut alpha = 0;
    while alpha < 256 {
        let mut color = 0;
        while color < 256 {
            // (color * alpha) / 255 with proper rounding
            lut[alpha][color] = ((color * alpha + 127) / 255) as u8;
            color += 1;
        }
        alpha += 1;
    }
    lut
}

/// Compile-time generated Look-Up Table for u8 alpha premultiplication
static ALPHA_LUT: [[u8; 256]; 256] = generate_alpha_lut();

/// Fast u8 alpha premultiplication using compile-time LUT
#[inline]
const fn premultiply_u8(color: u8, alpha: u8) -> u8 {
    ALPHA_LUT[alpha as usize][color as usize]
}

/// Fast u8 RGB premultiplication using compile-time LUT
#[inline]
const fn premultiply_rgb_u8(channels: [u8; 3], alpha: u8) -> [u8; 3] {
    [
        premultiply_u8(channels[0], alpha),
        premultiply_u8(channels[1], alpha),
        premultiply_u8(channels[2], alpha),
    ]
}

/// Computes premultiplied RGB pixel with proper clamping
#[inline]
fn compute_premultiplied_rgb_impl<S>(channels: [S; 3], alpha_normalized: f32) -> Rgb<S>
where
    S: Clamp<f32> + Primitive,
    f32: From<S>,
{
    // Direct array construction to avoid intermediate allocations
    let [r, g, b] = channels;
    let r = f32::from(r) * alpha_normalized;
    let g = f32::from(g) * alpha_normalized;
    let b = f32::from(b) * alpha_normalized;

    Rgb([S::clamp(r), S::clamp(g), S::clamp(b)])
}

/// Optimized integer premultiplication for u16 type using fixed-point arithmetic
#[inline]
const fn premultiply_u16(color: u16, alpha: u16) -> u16 {
    // Use fixed-point arithmetic to avoid floating point operations
    // (color * alpha) / 65535 with proper rounding
    let result = (color as u32 * alpha as u32 + 32767) / 65535;
    result as u16
}

/// Optimized integer premultiplication for u32 type using fixed-point arithmetic
#[inline]
const fn premultiply_u32(color: u32, alpha: u32) -> u32 {
    // Use 64-bit arithmetic to avoid overflow
    let result = (color as u64 * alpha as u64 + (u32::MAX as u64 / 2)) / u32::MAX as u64;
    result as u32
}

/// Processes pixels from source to destination using a mapping function
fn process_pixels_with_mapping<SP, DP, F>(
    src_image: &Image<SP>,
    dst_image: &mut Image<DP>,
    mapper: F,
) where
    SP: Pixel,
    DP: Pixel,
    F: Fn(&SP) -> DP,
{
    src_image
        .pixels()
        .zip_eq(dst_image.pixels_mut())
        .for_each(|(src_pixel, dst_pixel)| {
            *dst_pixel = mapper(src_pixel);
        });
}

/// Creates a new image by processing each pixel with a mapping function
fn map_pixels_to_new_image<SP, DP, F>(
    src_image: &Image<SP>,
    mapper: F,
) -> Result<Image<DP>, ColorConversionError>
where
    SP: Pixel,
    DP: Pixel,
    F: Fn(&SP) -> DP,
{
    validate_image_dimensions(src_image)?;

    let (width, height) = src_image.dimensions();
    let mut out = ImageBuffer::new(width, height);

    process_pixels_with_mapping(src_image, &mut out, mapper);

    Ok(out)
}

/// Validates image dimensions for processing
fn validate_image_dimensions<I>(image: &I) -> Result<(), ColorConversionError>
where
    I: GenericImageView,
{
    let (width, height) = image.dimensions();

    if width == 0 || height == 0 {
        Err(ColorConversionError::EmptyImage)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_premultiplied_rgb_impl_with_valid_input_returns_correct_values() {
        let channels = [200u8, 150u8, 100u8];
        let alpha = 0.5;
        let result = compute_premultiplied_rgb_impl(channels, alpha);

        assert_eq!(result[0], 100); // 200 * 0.5
        assert_eq!(result[1], 75); // 150 * 0.5
        assert_eq!(result[2], 50); // 100 * 0.5
    }

    #[test]
    fn validate_image_dimensions_with_empty_images_rejects() {
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
    fn premultiply_alpha_and_drop_for_luma_drops_alpha_channel() {
        let mut image: Image<LumaA<u8>> = Image::new(2, 2);
        image.put_pixel(0, 0, LumaA([200, 255])); // Full opacity
        image.put_pixel(1, 0, LumaA([200, 127])); // Half opacity
        image.put_pixel(0, 1, LumaA([200, 0])); // Transparent
        image.put_pixel(1, 1, LumaA([100, 255])); // Full opacity, different value

        let result = image.premultiply_alpha_and_drop().unwrap();

        assert_eq!(result.get_pixel(0, 0)[0], 200); // 200 * 1.0
        assert_eq!(result.get_pixel(1, 0)[0], 100); // 200 * 127/255
        assert_eq!(result.get_pixel(0, 1)[0], 0); // 200 * 0.0
        assert_eq!(result.get_pixel(1, 1)[0], 100); // 100 * 1.0
    }

    #[test]
    fn premultiply_alpha_and_drop_for_rgba_drops_alpha_channel() {
        let mut image: Image<Rgba<u8>> = Image::new(2, 2);
        image.put_pixel(0, 0, Rgba([200, 150, 100, 255])); // Full opacity
        image.put_pixel(1, 0, Rgba([200, 150, 100, 127])); // Half opacity
        image.put_pixel(0, 1, Rgba([200, 150, 100, 0])); // Transparent
        image.put_pixel(1, 1, Rgba([100, 50, 25, 255])); // Full opacity, different values

        let result = image.premultiply_alpha_and_drop().unwrap();

        // Full opacity case
        let pixel_00 = result.get_pixel(0, 0);
        assert_eq!(pixel_00[0], 200);
        assert_eq!(pixel_00[1], 150);
        assert_eq!(pixel_00[2], 100);

        // Half opacity case
        let pixel_10 = result.get_pixel(1, 0);
        assert_eq!(pixel_10[0], 100); // 200 * 127/255
        assert_eq!(pixel_10[1], 75); // 150 * 127/255
        assert_eq!(pixel_10[2], 50); // 100 * 127/255

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
    fn premultiply_alpha_and_keep_for_luma_preserves_alpha() {
        let mut image: Image<LumaA<u8>> = Image::new(2, 2);
        image.put_pixel(0, 0, LumaA([200, 255])); // Full opacity
        image.put_pixel(1, 0, LumaA([200, 127])); // Half opacity

        let result = image.clone().premultiply_alpha_and_keep().unwrap();

        // Check that luminance is premultiplied but alpha is preserved
        assert_eq!(result.get_pixel(0, 0).0, [200, 255]); // 200 * 1.0, alpha preserved
        assert_eq!(result.get_pixel(1, 0).0, [100, 127]); // 200 * 127/255, alpha preserved
    }

    #[test]
    fn premultiply_alpha_and_keep_mut_for_rgba_preserves_alpha() {
        let mut image: Image<Rgba<u8>> = Image::new(2, 2);
        image.put_pixel(0, 0, Rgba([200, 150, 100, 255])); // Full opacity
        image.put_pixel(1, 0, Rgba([200, 150, 100, 127])); // Half opacity

        let mut image_copy = image.clone();
        image_copy.premultiply_alpha_and_keep_mut().unwrap();

        // Check that colors are premultiplied but alpha is preserved
        assert_eq!(image_copy.get_pixel(0, 0).0, [200, 150, 100, 255]); // Full opacity unchanged
        assert_eq!(image_copy.get_pixel(1, 0).0, [100, 75, 50, 127]); // Premultiplied, alpha preserved
    }
}
