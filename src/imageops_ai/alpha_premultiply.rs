use image::{GenericImageView, ImageBuffer, Luma, LumaA, Pixel, Primitive, Rgb, Rgba};
use imageproc::definitions::{Clamp, Image};
use ndarray::prelude::*;

use crate::error::ConvertColorError;

// --- Public Traits ---

/// A trait for premultiplying the alpha channel into color channels.
///
/// This trait provides methods for standard alpha premultiplication, where the
/// color channels are multiplied by the alpha value, and the alpha channel is preserved.
/// It offers both an owning (`premultiply_alpha`) and a mutable (`premultiply_alpha_mut`) version.
pub trait PremultiplyAlpha {
    /// Consumes the image and returns a new image with premultiplied alpha.
    ///
    /// The alpha channel is preserved.
    fn premultiply_alpha(self) -> Result<Self, ConvertColorError>
    where
        Self: Sized;

    /// Premultiplies the alpha of the image in-place.
    ///
    /// This method operates on a mutable reference and avoids extra allocations.
    /// The alpha channel is preserved.
    fn premultiply_alpha_mut(&mut self) -> Result<&mut Self, ConvertColorError>
    where
        Self: Sized;
}

/// A trait for premultiplying the alpha channel and dropping it from the final image.
///
/// This operation is useful when converting an image with an alpha channel (e.g., `Rgba`)
/// to one without (e.g., `Rgb`), such as for displaying on a solid background.
pub trait PremultiplyAlphaAndDrop {
    /// The output image type, which does not have an alpha channel.
    type Output;

    /// Consumes the image, premultiplies the alpha, and returns a new image
    /// without an alpha channel.
    fn premultiply_alpha_and_drop(self) -> Result<Self::Output, ConvertColorError>;
}

// --- Trait Implementations ---

impl<S> PremultiplyAlpha for Image<LumaA<S>>
where
    S: Clamp<f32> + Primitive,
    f32: From<S>,
{
    fn premultiply_alpha(self) -> Result<Self, ConvertColorError> {
        let mut image = self;
        premultiply_alpha_in_place_impl(&mut image)?;
        Ok(image)
    }

    fn premultiply_alpha_mut(&mut self) -> Result<&mut Self, ConvertColorError> {
        premultiply_alpha_in_place_impl(self)?;
        Ok(self)
    }
}

impl<S> PremultiplyAlpha for Image<Rgba<S>>
where
    Rgba<S>: Pixel<Subpixel = S>,
    S: Clamp<f32> + Primitive,
    f32: From<S>,
{
    fn premultiply_alpha(self) -> Result<Self, ConvertColorError> {
        let mut image = self;
        premultiply_alpha_in_place_impl(&mut image)?;
        Ok(image)
    }

    fn premultiply_alpha_mut(&mut self) -> Result<&mut Self, ConvertColorError> {
        premultiply_alpha_in_place_impl(self)?;
        Ok(self)
    }
}

impl<S> PremultiplyAlphaAndDrop for Image<LumaA<S>>
where
    S: Clamp<f32> + Primitive,
    f32: From<S>,
{
    type Output = Image<Luma<S>>;

    fn premultiply_alpha_and_drop(self) -> Result<Self::Output, ConvertColorError> {
        premultiply_and_drop_alpha_impl(&self)
    }
}

impl<S> PremultiplyAlphaAndDrop for Image<Rgba<S>>
where
    Rgba<S>: Pixel<Subpixel = S>,
    Rgb<S>: Pixel<Subpixel = S>,
    S: Clamp<f32> + Primitive,
    f32: From<S>,
{
    type Output = Image<Rgb<S>>;

    fn premultiply_alpha_and_drop(self) -> Result<Self::Output, ConvertColorError> {
        premultiply_and_drop_alpha_impl(&self)
    }
}

// --- Core Implementations ---

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
    let num_channels = P::CHANNEL_COUNT as usize;
    let alpha_index = num_channels - 1;

    let array = if max_value == 1.0 {
        ArrayView1::from(image.as_raw()).mapv(f32::from)
    } else {
        ArrayView1::from(image.as_raw()).mapv(|x| f32::from(x) / max_value)
    }
    .into_shape_with_order((num_pixels, num_channels))
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

/// Core implementation for in-place alpha premultiplication (preserves alpha channel).
/// This function modifies the image buffer directly without reallocation.
fn premultiply_alpha_in_place_impl<P, S>(image: &mut Image<P>) -> Result<(), ConvertColorError>
where
    P: Pixel<Subpixel = S>,
    S: Clamp<f32> + Primitive,
    f32: From<S>,
{
    validate_image_dimensions(image)?;
    let max_value = f32::from(S::DEFAULT_MAX_VALUE);
    let num_channels = P::CHANNEL_COUNT as usize;
    let alpha_index = num_channels - 1;

    let buffer = image.as_mut();
    let array: ArrayBase<ndarray::ViewRepr<&mut S>, Dim<[usize; 2]>> =
        ArrayViewMut2::from_shape((buffer.len() / num_channels, num_channels), buffer)
            .map_err(|_| ConvertColorError::BufferCreationFailed)?;

    // `split_at_mut` splits the array into two mutable views.
    let (mut colors, alphas) = array.split_at(Axis(1), alpha_index);

    // `alphas` is a column view (shape [n, 1]). Convert it to a 1D array of f32 values
    // for use in `azip!`.
    let alpha_values = alphas.column(0).mapv(|a| f32::from(a) / max_value);

    // Zip the rows of the color channels with the 1D array of alpha values.
    azip!((mut color_row in colors.rows_mut(), &alpha in &alpha_values) {
        // Each `color_row` is an `ArrayViewMut1`, which can be modified in place.
        color_row.mapv_inplace(|c| {
            let c_f32 = f32::from(c) / max_value;
            S::clamp(c_f32 * alpha * max_value)
        });
    });

    Ok(())
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
    fn validate_dimensions_accepts_valid_image() {
        let valid_image: Image<Rgb<u8>> = Image::new(10, 10);
        assert!(validate_image_dimensions(&valid_image).is_ok());
    }

    #[test]
    fn validate_dimensions_rejects_empty_image() {
        let empty_image: Image<Rgb<u8>> = Image::new(0, 0);
        assert!(validate_image_dimensions(&empty_image).is_err());
    }

    #[test]
    fn validate_dimensions_rejects_zero_width_image() {
        let invalid_width: Image<Rgb<u8>> = Image::new(0, 10);
        assert!(validate_image_dimensions(&invalid_width).is_err());
    }

    #[test]
    fn validate_dimensions_rejects_zero_height_image() {
        let invalid_height: Image<Rgb<u8>> = Image::new(10, 0);
        assert!(validate_image_dimensions(&invalid_height).is_err());
    }

    #[test]
    fn premultiply_and_drop_for_luma_a_works_correctly() {
        let mut image: Image<LumaA<u8>> = Image::new(2, 2);
        image.put_pixel(0, 0, LumaA([200, 255])); // Full opacity
        image.put_pixel(1, 0, LumaA([200, 127])); // Half opacity
        image.put_pixel(0, 1, LumaA([200, 0])); // Transparent
        image.put_pixel(1, 1, LumaA([100, 255])); // Full opacity, different value

        let result = image.premultiply_alpha_and_drop().unwrap();

        assert_eq!(result.get_pixel(0, 0)[0], 200); // 200 * 1.0
        assert_eq!(result.get_pixel(1, 0)[0], 99); // 200 * 0.498
        assert_eq!(result.get_pixel(0, 1)[0], 0); // 200 * 0.0
        assert_eq!(result.get_pixel(1, 1)[0], 100); // 100 * 1.0
    }

    #[test]
    fn premultiply_and_drop_for_rgba_works_correctly() {
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
    fn premultiply_alpha_owned_for_luma_a_preserves_alpha() {
        let mut image: Image<LumaA<u8>> = Image::new(2, 2);
        image.put_pixel(0, 0, LumaA([200, 255])); // Full opacity
        image.put_pixel(1, 0, LumaA([200, 127])); // Half opacity

        let result = image.clone().premultiply_alpha().unwrap();

        // Check that luminance is premultiplied but alpha is preserved
        assert_eq!(result.get_pixel(0, 0).0, [200, 255]); // 200 * 1.0, alpha preserved
        assert_eq!(result.get_pixel(1, 0).0, [99, 127]); // 200 * 0.498, alpha preserved
    }

    #[test]
    fn premultiply_alpha_mut_for_rgba_preserves_alpha() {
        let mut image: Image<Rgba<u8>> = Image::new(2, 2);
        image.put_pixel(0, 0, Rgba([200, 150, 100, 255])); // Full opacity
        image.put_pixel(1, 0, Rgba([200, 150, 100, 127])); // Half opacity

        image.premultiply_alpha_mut().unwrap();

        // Check that colors are premultiplied but alpha is preserved
        assert_eq!(image.get_pixel(0, 0).0, [200, 150, 100, 255]); // Full opacity unchanged
        assert_eq!(image.get_pixel(1, 0).0, [99, 74, 49, 127]); // Premultiplied, alpha preserved
    }
}
