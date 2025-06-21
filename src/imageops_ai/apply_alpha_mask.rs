use crate::error::AlphaMaskError;
use crate::utils::{clamp_f32_to_primitive, validate_matching_dimensions};
use image::{Luma, Pixel, Primitive, Rgb, Rgba};
use imageproc::definitions::{Clamp, Image};
use imageproc::map::map_colors2;

/// Trait providing functionality to apply alpha masks to images
///
/// This trait provides functionality to apply grayscale masks to RGB images
/// to generate RGBA images.
pub trait ApplyAlphaMask<S>
where
    Rgba<S>: Pixel<Subpixel = S>,
    S: Primitive,
{
    /// Applies the specified mask to the image and generates an image with alpha channel
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
    fn apply_alpha_mask(&self, mask: &Image<Luma<S>>) -> Result<Image<Rgba<S>>, AlphaMaskError>;
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
    fn apply_alpha_mask<SM>(
        &self,
        mask: &Image<Luma<SM>>,
    ) -> Result<Image<Rgba<S>>, AlphaMaskError>
    where
        SM: Into<f32> + Primitive;
}

impl<S> ApplyAlphaMask<S> for Image<Rgb<S>>
where
    Rgb<S>: Pixel<Subpixel = S>,
    Rgba<S>: Pixel<Subpixel = S>,
    S: Primitive,
{
    fn apply_alpha_mask(&self, mask: &Image<Luma<S>>) -> Result<Image<Rgba<S>>, AlphaMaskError> {
        validate_dimensions(self, mask)?;

        let result = map_colors2(self, mask, |Rgb([red, green, blue]), Luma([alpha])| {
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
    fn apply_alpha_mask<SM>(&self, mask: &Image<Luma<SM>>) -> Result<Image<Rgba<S>>, AlphaMaskError>
    where
        SM: Into<f32> + Primitive,
    {
        validate_dimensions(self, mask)?;

        let source_max = S::DEFAULT_MAX_VALUE.into();
        let mask_max = SM::DEFAULT_MAX_VALUE.into();

        let result = map_colors2(
            self,
            mask,
            |Rgb([red, green, blue]), Luma([alpha_value])| {
                // Alpha value scaling and clamping
                let scaled_alpha = (alpha_value.into() / mask_max) * source_max;
                let alpha = clamp_f32_to_primitive(scaled_alpha);
                Rgba([red, green, blue, alpha])
            },
        );

        Ok(result)
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

        let result = ApplyAlphaMask::apply_alpha_mask(&image, &mask).unwrap();

        assert_eq!(result.get_pixel(0, 0), &Rgba([255, 0, 0, 255]));
        assert_eq!(result.get_pixel(1, 0), &Rgba([0, 255, 0, 128]));
        assert_eq!(result.get_pixel(0, 1), &Rgba([0, 0, 255, 64]));
        assert_eq!(result.get_pixel(1, 1), &Rgba([255, 255, 255, 0]));
    }
}
