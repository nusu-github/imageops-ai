use image::{GenericImageView, ImageBuffer, Luma, LumaA, Pixel, Primitive, Rgb, Rgba};
use imageproc::definitions::{Clamp, Image};
use ndarray::prelude::*;

use crate::error::ConvertColorError;

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
    fn premultiply_alpha(self) -> Result<Self::Output, ConvertColorError>;
}

/// Trait for in-place alpha premultiplication (keeps alpha channel).
///
/// This trait provides functionality to premultiply color channels with alpha
/// while preserving the alpha channel in the output.
pub trait PremultiplyAlphaInPlace {
    /// Premultiplies color channels by alpha, keeping the alpha channel.
    ///
    /// This consumes the original image and returns a premultiplied version
    /// with the same pixel type.
    fn premultiply_alpha_keep(self) -> Result<Self, ConvertColorError>
    where
        Self: Sized;

    /// Premultiplies color channels by alpha in-place, modifying the image.
    fn premultiply_alpha_keep_mut(&mut self) -> Result<&mut Self, ConvertColorError>
    where
        Self: Sized;
}

/// Core implementation for alpha premultiplication that drops the alpha channel.
fn premultiply_and_drop_alpha_impl<P, S, O>(image: &Image<P>) -> Result<Image<O>, ConvertColorError>
where
    P: Pixel<Subpixel = S>,
    O: Pixel<Subpixel = S>,
    S: Clamp<f32> + Primitive,
    f32: From<S>,
{
    validate_image_dimensions(image)?;
    let (width, height) = image.dimensions();
    let max_value = f32::from(S::DEFAULT_MAX_VALUE);
    let num_pixels = (width * height) as usize;
    let in_channels = P::CHANNEL_COUNT as usize;
    let alpha_index = in_channels - 1;

    let array = if max_value == 1.0 {
        ArrayView1::from(image.as_raw()).mapv(f32::from)
    } else {
        ArrayView1::from(image.as_raw()).mapv(|x| f32::from(x) / max_value)
    }
    .into_shape_with_order((num_pixels, in_channels))
    .map_err(|_| ConvertColorError::BufferCreationFailed)?;

    let alphas = array.column(alpha_index);
    let colors = array.slice(s![.., 0..alpha_index]);

    let result = &colors * &alphas.insert_axis(Axis(1));

    let result_vec = if max_value == 1.0 {
        result.mapv(|x| S::clamp(x))
    } else {
        result.mapv(|x| S::clamp(x * max_value))
    }
    .into_raw_vec_and_offset()
    .0;

    ImageBuffer::from_raw(width, height, result_vec).ok_or(ConvertColorError::EmptyImage)
}

/// Core implementation for in-place alpha premultiplication that keeps the alpha channel.
/// This function reallocates the buffer.
fn premultiply_and_keep_alpha_impl<P, S>(image: &mut Image<P>) -> Result<(), ConvertColorError>
where
    P: Pixel<Subpixel = S>,
    S: Clamp<f32> + Primitive,
    f32: From<S>,
{
    validate_image_dimensions(image)?;
    let (width, height) = image.dimensions();
    let max_value = f32::from(S::DEFAULT_MAX_VALUE);
    let num_pixels = (width * height) as usize;
    let channels = P::CHANNEL_COUNT as usize;
    let alpha_index = channels - 1;

    let mut array = if max_value == 1.0 {
        ArrayView1::from(image.as_raw()).mapv(f32::from)
    } else {
        ArrayView1::from(image.as_raw()).mapv(|x| f32::from(x) / max_value)
    }
    .into_shape_with_order((num_pixels, channels))
    .map_err(|_| ConvertColorError::BufferCreationFailed)?;

    for mut pixel in array.axis_iter_mut(Axis(0)) {
        let alpha = pixel[alpha_index];
        let mut colors = pixel.slice_mut(s![0..alpha_index]);
        colors *= alpha;
    }

    let result_vec = if max_value == 1.0 {
        array.mapv(|x| S::clamp(x))
    } else {
        array.mapv(|x| S::clamp(x * max_value))
    }
    .into_raw_vec_and_offset()
    .0;

    *image =
        ImageBuffer::from_raw(width, height, result_vec).ok_or(ConvertColorError::EmptyImage)?;
    Ok(())
}

impl<S> AlphaPremultiply for Image<LumaA<S>>
where
    S: Clamp<f32> + Primitive,
    f32: From<S>,
{
    type Output = Image<Luma<S>>;

    fn premultiply_alpha(self) -> Result<Self::Output, ConvertColorError> {
        premultiply_and_drop_alpha_impl(&self)
    }
}

impl<S> AlphaPremultiply for Image<Rgba<S>>
where
    Rgba<S>: Pixel<Subpixel = S>,
    Rgb<S>: Pixel<Subpixel = S>,
    S: Clamp<f32> + Primitive,
    f32: From<S>,
{
    type Output = Image<Rgb<S>>;

    fn premultiply_alpha(self) -> Result<Self::Output, ConvertColorError> {
        premultiply_and_drop_alpha_impl(&self)
    }
}

impl<S> PremultiplyAlphaInPlace for Image<LumaA<S>>
where
    S: Clamp<f32> + Primitive,
    f32: From<S>,
{
    fn premultiply_alpha_keep(self) -> Result<Self, ConvertColorError> {
        let mut image = self;
        premultiply_and_keep_alpha_impl(&mut image)?;
        Ok(image)
    }

    fn premultiply_alpha_keep_mut(&mut self) -> Result<&mut Self, ConvertColorError> {
        premultiply_and_keep_alpha_impl(self)?;
        Ok(self)
    }
}

impl<S> PremultiplyAlphaInPlace for Image<Rgba<S>>
where
    Rgba<S>: Pixel<Subpixel = S>,
    S: Clamp<f32> + Primitive,
    f32: From<S>,
{
    fn premultiply_alpha_keep(self) -> Result<Self, ConvertColorError> {
        let mut image = self;
        premultiply_and_keep_alpha_impl(&mut image)?;
        Ok(image)
    }

    fn premultiply_alpha_keep_mut(&mut self) -> Result<&mut Self, ConvertColorError> {
        premultiply_and_keep_alpha_impl(self)?;
        Ok(self)
    }
}

/// Validates image dimensions for processing
fn validate_image_dimensions<I, P>(image: &I) -> Result<(), ConvertColorError>
where
    I: GenericImageView<Pixel = P>,
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
