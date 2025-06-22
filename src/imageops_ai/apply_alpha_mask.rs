use crate::error::AlphaMaskError;
use crate::utils::{clamp_f32_to_primitive, validate_matching_dimensions};
use image::{Luma, Pixel, Primitive, Rgb, Rgba};
use imageproc::definitions::{Clamp, Image};
use imageproc::map::map_colors2;

/// Trait providing functionality to apply alpha masks to images
///
/// This trait provides functionality to apply grayscale masks to RGB images
/// to generate RGBA images. This consumes the original image.
///
/// Note: This trait performs type conversion (e.g., Rgb -> Rgba). For modifying
/// existing RGBA images' alpha channel, use the `ModifyAlpha` trait.
pub trait ApplyAlphaMask<S>
where
    Rgba<S>: Pixel<Subpixel = S>,
    S: Primitive,
{
    /// Applies the specified mask to the image and generates an image with alpha channel
    ///
    /// This consumes the original image.
    ///
    /// # Arguments
    ///
    /// * `mask` - The alpha mask to apply (grayscale image)
    ///
    /// # Returns
    ///
    /// RGBA image with added alpha channel
    ///
    /// # Errors
    ///
    /// * `Error::DimensionMismatch` - When image and mask dimensions don't match
    /// * `Error::ImageBufferCreationFailed` - When result image creation fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use imageops_ai::{Image, ApplyAlphaMask};
    /// use image::{ImageBuffer, Rgb, Luma};
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // RGB image and mask must have the same dimensions
    /// let rgb_image: Image<Rgb<u8>> = ImageBuffer::new(10, 10);
    /// let mask: Image<Luma<u8>> = ImageBuffer::new(10, 10);
    ///
    /// let rgba_image = rgb_image.apply_alpha_mask(&mask)?;
    /// # Ok(())
    /// # }
    /// ```
    fn apply_alpha_mask(self, mask: &Image<Luma<S>>) -> Result<Image<Rgba<S>>, AlphaMaskError>;
}

/// Trait for modifying alpha channel of existing RGBA images
///
/// This trait provides functionality to replace or modify the alpha channel
/// of RGBA images while preserving the RGB color channels.
pub trait ModifyAlpha<S>
where
    Rgba<S>: Pixel<Subpixel = S>,
    S: Primitive,
{
    /// Replaces the alpha channel with the provided mask
    ///
    /// This consumes the original image.
    ///
    /// # Arguments
    ///
    /// * `mask` - The new alpha mask (grayscale image)
    ///
    /// # Returns
    ///
    /// RGBA image with replaced alpha channel
    ///
    /// # Errors
    ///
    /// * `Error::DimensionMismatch` - When image and mask dimensions don't match
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use imageops_ai::{Image, ModifyAlpha};
    /// use image::{ImageBuffer, Rgba, Luma};
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let rgba_image: Image<Rgba<u8>> = ImageBuffer::new(10, 10);
    /// let new_mask: Image<Luma<u8>> = ImageBuffer::new(10, 10);
    ///
    /// let updated = rgba_image.replace_alpha(&new_mask)?;
    /// # Ok(())
    /// # }
    /// ```
    fn replace_alpha(self, mask: &Image<Luma<S>>) -> Result<Self, AlphaMaskError>
    where
        Self: Sized;

    /// Replaces the alpha channel with the provided mask in-place
    ///
    /// # Arguments
    ///
    /// * `mask` - The new alpha mask (grayscale image)
    ///
    /// # Returns
    ///
    /// Mutable reference to self with replaced alpha channel
    ///
    /// # Errors
    ///
    /// * `Error::DimensionMismatch` - When image and mask dimensions don't match
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use imageops_ai::{Image, ModifyAlpha};
    /// use image::{ImageBuffer, Rgba, Luma};
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut rgba_image: Image<Rgba<u8>> = ImageBuffer::new(10, 10);
    /// let new_mask: Image<Luma<u8>> = ImageBuffer::new(10, 10);
    ///
    /// rgba_image.replace_alpha_mut(&new_mask)?;
    /// # Ok(())
    /// # }
    /// ```
    fn replace_alpha_mut(&mut self, mask: &Image<Luma<S>>) -> Result<&mut Self, AlphaMaskError>
    where
        Self: Sized;
}

/// Trait providing functionality to apply alpha masks to images (with type conversion support)
///
/// This trait provides functionality to apply grayscale masks to RGB images
/// to generate RGBA images.
/// Use this when the mask and image have different types.
pub trait ApplyAlphaMaskConvert<S>
where
    Rgba<S>: Pixel<Subpixel = S>,
    S: Primitive,
{
    /// Applies the specified mask to the image and generates an image with alpha channel
    ///
    /// This consumes the original image.
    ///
    /// # Arguments
    ///
    /// * `mask` - The alpha mask to apply (grayscale image)
    ///
    /// # Returns
    ///
    /// RGBA image with added alpha channel
    ///
    /// # Errors
    ///
    /// * `Error::DimensionMismatch` - When image and mask dimensions don't match
    /// * `Error::ImageBufferCreationFailed` - When result image creation fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use imageops_ai::{Image, ApplyAlphaMaskConvert};
    /// use image::{ImageBuffer, Rgb, Luma};
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // RGB image and mask must have the same dimensions
    /// let rgb_image: Image<Rgb<u8>> = ImageBuffer::new(10, 10);
    /// let mask: Image<Luma<u16>> = ImageBuffer::new(10, 10);
    ///
    /// let rgba_image = rgb_image.apply_alpha_mask(&mask)?;
    /// # Ok(())
    /// # }
    /// ```
    fn apply_alpha_mask<SM>(self, mask: &Image<Luma<SM>>) -> Result<Image<Rgba<S>>, AlphaMaskError>
    where
        SM: Into<f32> + Primitive;
}

impl<S> ApplyAlphaMask<S> for Image<Rgb<S>>
where
    Rgb<S>: Pixel<Subpixel = S>,
    Rgba<S>: Pixel<Subpixel = S>,
    S: Primitive,
{
    fn apply_alpha_mask(self, mask: &Image<Luma<S>>) -> Result<Image<Rgba<S>>, AlphaMaskError> {
        validate_dimensions(&self, mask)?;

        let result = map_colors2(&self, mask, |Rgb([red, green, blue]), Luma([alpha])| {
            Rgba([red, green, blue, alpha])
        });

        Ok(result)
    }
}

impl<S> ApplyAlphaMaskConvert<S> for Image<Rgb<S>>
where
    Rgb<S>: Pixel<Subpixel = S>,
    Rgba<S>: Pixel<Subpixel = S>,
    S: Into<f32> + Clamp<f32> + Primitive,
{
    fn apply_alpha_mask<SM>(self, mask: &Image<Luma<SM>>) -> Result<Image<Rgba<S>>, AlphaMaskError>
    where
        SM: Into<f32> + Primitive,
    {
        validate_dimensions(&self, mask)?;

        let source_max = S::DEFAULT_MAX_VALUE.into();
        let mask_max = SM::DEFAULT_MAX_VALUE.into();

        Ok(map_colors2(
            &self,
            mask,
            |Rgb([red, green, blue]), Luma([alpha_value])| {
                // Alpha value scaling and clamping
                let scaled_alpha = (alpha_value.into() / mask_max) * source_max;
                let alpha = clamp_f32_to_primitive(scaled_alpha);
                Rgba([red, green, blue, alpha])
            },
        ))
    }
}

impl<S> ModifyAlpha<S> for Image<Rgba<S>>
where
    Rgba<S>: Pixel<Subpixel = S>,
    S: Primitive,
{
    fn replace_alpha(self, mask: &Image<Luma<S>>) -> Result<Self, AlphaMaskError> {
        validate_dimensions(&self, mask)?;

        let result = map_colors2(&self, mask, |Rgba([red, green, blue, _]), Luma([alpha])| {
            Rgba([red, green, blue, alpha])
        });

        Ok(result)
    }

    fn replace_alpha_mut(&mut self, mask: &Image<Luma<S>>) -> Result<&mut Self, AlphaMaskError> {
        validate_dimensions(self, mask)?;

        let (width, height) = self.dimensions();

        for y in 0..height {
            for x in 0..width {
                let pixel = self.get_pixel_mut(x, y);
                let alpha = mask.get_pixel(x, y)[0];
                pixel[3] = alpha;
            }
        }

        Ok(self)
    }
}

/// Function to validate dimensions
#[inline]
fn validate_dimensions<P1, P2>(image: &Image<P1>, mask: &Image<P2>) -> Result<(), AlphaMaskError>
where
    P1: Pixel,
    P2: Pixel,
{
    let (img_w, img_h) = image.dimensions();
    let (mask_w, mask_h) = mask.dimensions();

    validate_matching_dimensions(img_w, img_h, mask_w, mask_h, "ApplyAlphaMask").map_err(|_| {
        AlphaMaskError::DimensionMismatch {
            expected: (img_w, img_h),
            actual: (mask_w, mask_h),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_dimensions() {
        let image: Image<Rgb<u8>> = Image::new(10, 10);
        let mask: Image<Luma<u8>> = Image::new(10, 10);

        assert!(validate_dimensions(&image, &mask).is_ok());

        let mask_wrong_size: Image<Luma<u8>> = Image::new(5, 5);
        assert!(validate_dimensions(&image, &mask_wrong_size).is_err());
    }

    #[test]
    fn test_apply_alpha_mask_same_type() {
        let mut image: Image<Rgb<u8>> = Image::new(2, 2);
        let mut mask: Image<Luma<u8>> = Image::new(2, 2);

        image.put_pixel(0, 0, Rgb([255, 0, 0]));
        image.put_pixel(1, 0, Rgb([0, 255, 0]));
        image.put_pixel(0, 1, Rgb([0, 0, 255]));
        image.put_pixel(1, 1, Rgb([255, 255, 255]));

        mask.put_pixel(0, 0, Luma([255]));
        mask.put_pixel(1, 0, Luma([128]));
        mask.put_pixel(0, 1, Luma([64]));
        mask.put_pixel(1, 1, Luma([0]));

        let result = ApplyAlphaMask::apply_alpha_mask(image, &mask).unwrap();

        assert_eq!(result.get_pixel(0, 0), &Rgba([255, 0, 0, 255]));
        assert_eq!(result.get_pixel(1, 0), &Rgba([0, 255, 0, 128]));
        assert_eq!(result.get_pixel(0, 1), &Rgba([0, 0, 255, 64]));
        assert_eq!(result.get_pixel(1, 1), &Rgba([255, 255, 255, 0]));
    }

    #[test]
    fn test_modify_alpha() {
        use crate::ModifyAlpha;

        let mut image: Image<Rgba<u8>> = Image::new(2, 2);
        let mut mask: Image<Luma<u8>> = Image::new(2, 2);

        image.put_pixel(0, 0, Rgba([255, 0, 0, 200]));
        image.put_pixel(1, 0, Rgba([0, 255, 0, 100]));
        image.put_pixel(0, 1, Rgba([0, 0, 255, 50]));
        image.put_pixel(1, 1, Rgba([255, 255, 255, 150]));

        mask.put_pixel(0, 0, Luma([255]));
        mask.put_pixel(1, 0, Luma([128]));
        mask.put_pixel(0, 1, Luma([64]));
        mask.put_pixel(1, 1, Luma([0]));

        let result = image.replace_alpha(&mask).unwrap();

        // Color channels should remain unchanged, only alpha is replaced
        assert_eq!(result.get_pixel(0, 0), &Rgba([255, 0, 0, 255]));
        assert_eq!(result.get_pixel(1, 0), &Rgba([0, 255, 0, 128]));
        assert_eq!(result.get_pixel(0, 1), &Rgba([0, 0, 255, 64]));
        assert_eq!(result.get_pixel(1, 1), &Rgba([255, 255, 255, 0]));
    }

    #[test]
    fn test_modify_alpha_mut() {
        use crate::ModifyAlpha;

        let mut image: Image<Rgba<u8>> = Image::new(2, 2);
        let mut mask: Image<Luma<u8>> = Image::new(2, 2);

        image.put_pixel(0, 0, Rgba([255, 0, 0, 200]));
        image.put_pixel(1, 0, Rgba([0, 255, 0, 100]));

        mask.put_pixel(0, 0, Luma([255]));
        mask.put_pixel(1, 0, Luma([128]));

        image.replace_alpha_mut(&mask).unwrap();

        // Color channels should remain unchanged, only alpha is replaced
        assert_eq!(image.get_pixel(0, 0), &Rgba([255, 0, 0, 255]));
        assert_eq!(image.get_pixel(1, 0), &Rgba([0, 255, 0, 128]));
    }
}
