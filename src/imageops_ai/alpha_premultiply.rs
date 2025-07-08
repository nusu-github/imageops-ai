use crate::error::ConvertColorError;
use crate::utils::normalize_alpha_with_max;
use image::{GenericImage, ImageBuffer, Luma, LumaA, Pixel, Primitive, Rgb, Rgba};
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

/// Generic fallback implementation for LumaA -> Luma conversion with alpha premultiplication
fn generic_premultiply_lumaa<S>(image: Image<LumaA<S>>) -> Result<Image<Luma<S>>, ConvertColorError>
where
    S: Into<f32> + Clamp<f32> + Primitive,
{
    validate_image_dimensions(&image)?;

    let max_value = S::DEFAULT_MAX_VALUE.into();

    Ok(map_colors(&image, |pixel| {
        let LumaA([luminance, alpha]) = pixel;
        let alpha_normalized = normalize_alpha_with_max(alpha, max_value);
        let luminance_f32 = luminance.into();

        // Apply premultiplication with proper clamping
        let merged_f32 = luminance_f32 * alpha_normalized;
        let merged = S::clamp(merged_f32);

        Luma([merged])
    }))
}

/// Generic fallback implementation for Rgba -> Rgb conversion with alpha premultiplication
fn generic_premultiply_rgba<S>(image: Image<Rgba<S>>) -> Result<Image<Rgb<S>>, ConvertColorError>
where
    Rgba<S>: Pixel<Subpixel = S>,
    Rgb<S>: Pixel<Subpixel = S>,
    S: Into<f32> + Clamp<f32> + Primitive,
{
    validate_image_dimensions(&image)?;

    let max_value = S::DEFAULT_MAX_VALUE.into();

    Ok(map_colors(&image, |pixel| {
        let Rgba([red, green, blue, alpha]) = pixel;
        let alpha_normalized = normalize_alpha_with_max(alpha, max_value);

        // Convert to f32 and premultiply with optimized computation
        compute_premultiplied_rgb_pixel([red, green, blue], alpha_normalized)
    }))
}

/// Implementation for f32 LumaA -> Luma conversion
impl AlphaPremultiply for Image<LumaA<f32>> {
    type Output = Image<Luma<f32>>;

    fn premultiply_alpha(self) -> Result<Self::Output, ConvertColorError> {
        generic_premultiply_lumaa(self)
    }
}

/// Implementation for u16 LumaA -> Luma conversion
impl AlphaPremultiply for Image<LumaA<u16>> {
    type Output = Image<Luma<u16>>;

    fn premultiply_alpha(self) -> Result<Self::Output, ConvertColorError> {
        validate_image_dimensions(&self)?;

        let (width, height) = self.dimensions();
        let mut out: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::new(width, height);

        // Use direct pixel iterator for better performance
        for (src_pixel, dst_pixel) in self.pixels().zip(out.pixels_mut()) {
            let LumaA([luminance, alpha]) = *src_pixel;
            let premultiplied = fast_premultiply_u16(luminance, alpha);
            *dst_pixel = Luma([premultiplied]);
        }

        Ok(out)
    }
}

/// Optimized implementation for u8 LumaA -> Luma conversion using LUT
impl AlphaPremultiply for Image<LumaA<u8>> {
    type Output = Image<Luma<u8>>;

    fn premultiply_alpha(self) -> Result<Self::Output, ConvertColorError> {
        validate_image_dimensions(&self)?;

        let (width, height) = self.dimensions();
        let mut out: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(width, height);

        // Use direct pixel iterator for better performance
        for (src_pixel, dst_pixel) in self.pixels().zip(out.pixels_mut()) {
            let LumaA([luminance, alpha]) = *src_pixel;
            let premultiplied = fast_premultiply_u8(luminance, alpha);
            *dst_pixel = Luma([premultiplied]);
        }

        Ok(out)
    }
}

/// Implementation for f32 Rgba -> Rgb conversion
impl AlphaPremultiply for Image<Rgba<f32>> {
    type Output = Image<Rgb<f32>>;

    fn premultiply_alpha(self) -> Result<Self::Output, ConvertColorError> {
        generic_premultiply_rgba(self)
    }
}

/// Optimized implementation for u8 Rgba -> Rgb conversion using LUT
impl AlphaPremultiply for Image<Rgba<u8>> {
    type Output = Image<Rgb<u8>>;

    fn premultiply_alpha(self) -> Result<Self::Output, ConvertColorError> {
        validate_image_dimensions(&self)?;

        let (width, height) = self.dimensions();
        let mut out: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(width, height);

        // Use direct pixel iterator for better performance
        for (src_pixel, dst_pixel) in self.pixels().zip(out.pixels_mut()) {
            let Rgba([red, green, blue, alpha]) = *src_pixel;
            let premultiplied = fast_premultiply_rgb_u8([red, green, blue], alpha);
            *dst_pixel = Rgb(premultiplied);
        }

        Ok(out)
    }
}

/// Optimized implementation for u16 Rgba -> Rgb conversion using integer arithmetic
impl AlphaPremultiply for Image<Rgba<u16>> {
    type Output = Image<Rgb<u16>>;

    fn premultiply_alpha(self) -> Result<Self::Output, ConvertColorError> {
        validate_image_dimensions(&self)?;

        let (width, height) = self.dimensions();
        let mut out: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::new(width, height);

        // Use direct pixel iterator for better performance
        for (src_pixel, dst_pixel) in self.pixels().zip(out.pixels_mut()) {
            let Rgba([red, green, blue, alpha]) = *src_pixel;
            let premultiplied = [
                fast_premultiply_u16(red, alpha),
                fast_premultiply_u16(green, alpha),
                fast_premultiply_u16(blue, alpha),
            ];
            *dst_pixel = Rgb(premultiplied);
        }

        Ok(out)
    }
}

/// Optimized implementation for u32 Rgba -> Rgb conversion using integer arithmetic
impl AlphaPremultiply for Image<Rgba<u32>> {
    type Output = Image<Rgb<u32>>;

    fn premultiply_alpha(self) -> Result<Self::Output, ConvertColorError> {
        validate_image_dimensions(&self)?;

        let (width, height) = self.dimensions();
        let mut out: ImageBuffer<Rgb<u32>, Vec<u32>> = ImageBuffer::new(width, height);

        // Use direct pixel iterator for better performance
        for (src_pixel, dst_pixel) in self.pixels().zip(out.pixels_mut()) {
            let Rgba([red, green, blue, alpha]) = *src_pixel;
            let premultiplied = [
                fast_premultiply_u32(red, alpha),
                fast_premultiply_u32(green, alpha),
                fast_premultiply_u32(blue, alpha),
            ];
            *dst_pixel = Rgb(premultiplied);
        }

        Ok(out)
    }
}

/// Implementation for f32 LumaA images to premultiply while keeping alpha
impl PremultiplyAlphaInPlace for Image<LumaA<f32>> {
    fn premultiply_alpha_keep(self) -> Result<Self, ConvertColorError> {
        validate_image_dimensions(&self)?;

        let max_value = f32::DEFAULT_MAX_VALUE;

        Ok(map_colors(&self, |pixel| {
            let LumaA([luminance, alpha]) = pixel;
            let alpha_normalized = normalize_alpha_with_max(alpha, max_value);
            let luminance_f32: f32 = luminance;

            // Apply premultiplication with proper clamping
            let merged_f32: f32 = luminance_f32 * alpha_normalized;
            let merged = merged_f32.clamp(0.0, f32::DEFAULT_MAX_VALUE);

            LumaA([merged, alpha])
        }))
    }

    fn premultiply_alpha_keep_mut(&mut self) -> Result<&mut Self, ConvertColorError> {
        validate_image_dimensions(self)?;

        let max_value = f32::DEFAULT_MAX_VALUE;

        // Use iterator for better performance and readability
        self.pixels_mut().for_each(|pixel| {
            let LumaA([luminance, alpha]) = *pixel;
            let alpha_normalized = normalize_alpha_with_max(alpha, max_value);
            let luminance_f32: f32 = luminance;

            let merged_f32: f32 = luminance_f32 * alpha_normalized;
            let merged = merged_f32.clamp(0.0, f32::DEFAULT_MAX_VALUE);

            *pixel = LumaA([merged, alpha]);
        });

        Ok(self)
    }
}

/// Optimized implementation for u8 LumaA images using LUT
impl PremultiplyAlphaInPlace for Image<LumaA<u8>> {
    fn premultiply_alpha_keep(self) -> Result<Self, ConvertColorError> {
        validate_image_dimensions(&self)?;

        let (width, height) = self.dimensions();
        let mut out: ImageBuffer<LumaA<u8>, Vec<u8>> = ImageBuffer::new(width, height);

        // Use direct pixel iterator for better performance
        for (src_pixel, dst_pixel) in self.pixels().zip(out.pixels_mut()) {
            let LumaA([luminance, alpha]) = *src_pixel;
            let premultiplied = fast_premultiply_u8(luminance, alpha);
            *dst_pixel = LumaA([premultiplied, alpha]);
        }

        Ok(out)
    }

    fn premultiply_alpha_keep_mut(&mut self) -> Result<&mut Self, ConvertColorError> {
        validate_image_dimensions(self)?;

        // Use direct pixel iterator for better performance
        self.pixels_mut().for_each(|pixel| {
            let LumaA([luminance, alpha]) = *pixel;
            let premultiplied = fast_premultiply_u8(luminance, alpha);
            *pixel = LumaA([premultiplied, alpha]);
        });

        Ok(self)
    }
}

/// Implementation for f32 Rgba images to premultiply while keeping alpha
impl PremultiplyAlphaInPlace for Image<Rgba<f32>> {
    fn premultiply_alpha_keep(self) -> Result<Self, ConvertColorError> {
        validate_image_dimensions(&self)?;

        let max_value = f32::DEFAULT_MAX_VALUE;

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

        let max_value = f32::DEFAULT_MAX_VALUE;

        // Use iterator for better performance and readability
        self.pixels_mut().for_each(|pixel| {
            let Rgba([red, green, blue, alpha]) = *pixel;
            let alpha_normalized = normalize_alpha_with_max(alpha, max_value);

            let premultiplied =
                compute_premultiplied_rgb_pixel([red, green, blue], alpha_normalized);
            let Rgb([r_pre, g_pre, b_pre]) = premultiplied;

            *pixel = Rgba([r_pre, g_pre, b_pre, alpha]);
        });

        Ok(self)
    }
}

/// Optimized implementation for u8 Rgba images using LUT
impl PremultiplyAlphaInPlace for Image<Rgba<u8>> {
    fn premultiply_alpha_keep(self) -> Result<Self, ConvertColorError> {
        validate_image_dimensions(&self)?;

        let (width, height) = self.dimensions();
        let mut out: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(width, height);

        // Use direct pixel iterator for better performance
        for (src_pixel, dst_pixel) in self.pixels().zip(out.pixels_mut()) {
            let Rgba([red, green, blue, alpha]) = *src_pixel;
            let premultiplied = fast_premultiply_rgb_u8([red, green, blue], alpha);
            *dst_pixel = Rgba([premultiplied[0], premultiplied[1], premultiplied[2], alpha]);
        }

        Ok(out)
    }

    fn premultiply_alpha_keep_mut(&mut self) -> Result<&mut Self, ConvertColorError> {
        validate_image_dimensions(self)?;

        // Use direct pixel iterator for better performance
        self.pixels_mut().for_each(|pixel| {
            let Rgba([red, green, blue, alpha]) = *pixel;
            let premultiplied = fast_premultiply_rgb_u8([red, green, blue], alpha);
            *pixel = Rgba([premultiplied[0], premultiplied[1], premultiplied[2], alpha]);
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
const fn fast_premultiply_u8(color: u8, alpha: u8) -> u8 {
    ALPHA_LUT[alpha as usize][color as usize]
}

/// Fast u8 RGB premultiplication using compile-time LUT
#[inline]
const fn fast_premultiply_rgb_u8(channels: [u8; 3], alpha: u8) -> [u8; 3] {
    [
        fast_premultiply_u8(channels[0], alpha),
        fast_premultiply_u8(channels[1], alpha),
        fast_premultiply_u8(channels[2], alpha),
    ]
}

/// Computes premultiplied RGB pixel with proper clamping
#[inline]
fn compute_premultiplied_rgb_pixel<S>(channels: [S; 3], alpha_normalized: f32) -> Rgb<S>
where
    S: Into<f32> + Clamp<f32> + Primitive,
{
    // Direct array construction to avoid intermediate allocations
    let [r, g, b] = channels;
    let r_f32 = r.into() * alpha_normalized;
    let g_f32 = g.into() * alpha_normalized;
    let b_f32 = b.into() * alpha_normalized;

    Rgb([S::clamp(r_f32), S::clamp(g_f32), S::clamp(b_f32)])
}

/// Optimized integer premultiplication for u16 type using fixed-point arithmetic
#[inline]
const fn fast_premultiply_u16(color: u16, alpha: u16) -> u16 {
    // Use fixed-point arithmetic to avoid floating point operations
    // (color * alpha) / 65535 with proper rounding
    let result = (color as u32 * alpha as u32 + 32767) / 65535;
    result as u16
}

/// Optimized integer premultiplication for u32 type using fixed-point arithmetic
#[inline]
const fn fast_premultiply_u32(color: u32, alpha: u32) -> u32 {
    // Use 64-bit arithmetic to avoid overflow
    let result = (color as u64 * alpha as u64 + (u32::MAX as u64 / 2)) / u32::MAX as u64;
    result as u32
}

/// Validates image dimensions for processing
fn validate_image_dimensions<I, P>(image: &I) -> Result<(), ConvertColorError>
where
    I: GenericImage<Pixel = P>,
    P: Pixel,
{
    let (width, height) = image.dimensions();

    if width == 0 || height == 0 {
        Err(ConvertColorError::EmptyImage)
    } else {
        Ok(())
    }
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
