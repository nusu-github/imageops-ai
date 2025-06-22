//! Functions for unifying images with different numeric types.
//!
//! This module provides functionality to take two ImageBuffers with potentially different
//! subpixel types and return a unified ImageBuffer where both images are converted to
//! the larger numeric type.

use image::{Luma, Pixel, Primitive, Rgb};
use imageproc::definitions::Image;
use imageproc::map::{map_colors, WithChannel};

/// Trait for determining the larger of two numeric types.
/// This is used to unify two images with different subpixel types.
pub trait LargerType<T> {
    /// The larger of the two types.
    type Output;
}

/// Macro to implement LargerType for two types, where the second type is larger.
macro_rules! impl_larger_type {
    ($smaller:ty, $larger:ty) => {
        impl LargerType<$larger> for $smaller {
            type Output = $larger;
        }
        impl LargerType<$smaller> for $larger {
            type Output = $larger;
        }
    };
}

/// Macro to implement LargerType for a type with itself.
macro_rules! impl_larger_type_self {
    ($type:ty) => {
        impl LargerType<$type> for $type {
            type Output = $type;
        }
    };
}

// Implement LargerType for all combinations
impl_larger_type_self!(u8);
impl_larger_type_self!(u16);
impl_larger_type_self!(u32);
impl_larger_type_self!(u64);
impl_larger_type_self!(i8);
impl_larger_type_self!(i16);
impl_larger_type_self!(i32);
impl_larger_type_self!(i64);
impl_larger_type_self!(f32);
impl_larger_type_self!(f64);

// Unsigned integer hierarchy: u8 < u16 < u32 < u64
impl_larger_type!(u8, u16);
impl_larger_type!(u8, u32);
impl_larger_type!(u8, u64);
impl_larger_type!(u16, u32);
impl_larger_type!(u16, u64);
impl_larger_type!(u32, u64);

// Signed integer hierarchy: i8 < i16 < i32 < i64
impl_larger_type!(i8, i16);
impl_larger_type!(i8, i32);
impl_larger_type!(i8, i64);
impl_larger_type!(i16, i32);
impl_larger_type!(i16, i64);
impl_larger_type!(i32, i64);

// Float hierarchy: f32 < f64
impl_larger_type!(f32, f64);

// Mixed type promotions: integers promote to larger floats
impl_larger_type!(u8, f32);
impl_larger_type!(u16, f32);
impl_larger_type!(u8, f64);
impl_larger_type!(u16, f64);
impl_larger_type!(u32, f64);
impl_larger_type!(u64, f64);
impl_larger_type!(i8, f32);
impl_larger_type!(i16, f32);
impl_larger_type!(i8, f64);
impl_larger_type!(i16, f64);
impl_larger_type!(i32, f64);
impl_larger_type!(i64, f64);

// Cross-sign promotions: promote to larger signed type or float
impl_larger_type!(u8, i16);
impl_larger_type!(u8, i32);
impl_larger_type!(u8, i64);
impl_larger_type!(u16, i32);
impl_larger_type!(u16, i64);
impl_larger_type!(u32, i64);
impl_larger_type!(i8, u16);
impl_larger_type!(i8, u32);
impl_larger_type!(i8, u64);
impl_larger_type!(i16, u32);
impl_larger_type!(i16, u64);
impl_larger_type!(i32, u64);

/// Simplified function for the most common use case: unify two RGB images.
///
/// # Examples
/// ```no_run
/// use image::{ImageBuffer, Rgb};
/// use imageops_ai::unify_rgb_images;
///
/// let image1: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(2, 2, vec![
///     10, 20, 30, 40, 50, 60,
///     70, 80, 90, 100, 110, 120
/// ]).unwrap();
///
/// let image2: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::from_raw(2, 2, vec![
///     1000, 2000, 3000, 4000, 5000, 6000,
///     7000, 8000, 9000, 10000, 11000, 12000
/// ]).unwrap();
///
/// let (unified1, unified2) = unify_rgb_images(&image1, &image2);
/// // Both are now RGB<u16> images
/// ```
pub fn unify_rgb_images<T, U>(
    image1: &Image<Rgb<T>>,
    image2: &Image<Rgb<U>>,
) -> (
    Image<Rgb<<T as LargerType<U>>::Output>>,
    Image<Rgb<<T as LargerType<U>>::Output>>,
)
where
    T: LargerType<U> + Primitive + Into<<T as LargerType<U>>::Output>,
    U: Primitive + Into<<T as LargerType<U>>::Output>,
    <T as LargerType<U>>::Output: Primitive,
    Rgb<T>: WithChannel<<T as LargerType<U>>::Output>,
    Rgb<U>: WithChannel<<T as LargerType<U>>::Output>,
    Rgb<<T as LargerType<U>>::Output>: Pixel<Subpixel = <T as LargerType<U>>::Output>,
{
    let unified1 = map_colors(image1, |x| Rgb([x[0].into(), x[1].into(), x[2].into()]));
    let unified2 = map_colors(image2, |x| Rgb([x[0].into(), x[1].into(), x[2].into()]));
    (unified1, unified2)
}

/// Simplified function for the most common use case: unify two grayscale images.
///
/// # Examples
/// ```no_run
/// use image::{ImageBuffer, Luma};
/// use imageops_ai::unify_gray_images;
///
/// let image1: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_raw(2, 2, vec![
///     10, 20, 30, 40
/// ]).unwrap();
///
/// let image2: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::from_raw(2, 2, vec![
///     1000, 2000, 3000, 4000
/// ]).unwrap();
///
/// let (unified1, unified2) = unify_gray_images(&image1, &image2);
/// // Both are now Luma<u16> images
/// ```
pub fn unify_gray_images<T, U>(
    image1: &Image<Luma<T>>,
    image2: &Image<Luma<U>>,
) -> (
    Image<Luma<<T as LargerType<U>>::Output>>,
    Image<Luma<<T as LargerType<U>>::Output>>,
)
where
    T: LargerType<U> + Primitive + Into<<T as LargerType<U>>::Output>,
    U: Primitive + Into<<T as LargerType<U>>::Output>,
    <T as LargerType<U>>::Output: Primitive,
    Luma<T>: WithChannel<<T as LargerType<U>>::Output>,
    Luma<U>: WithChannel<<T as LargerType<U>>::Output>,
{
    let unified1 = map_colors(image1, |x| Luma([x[0].into()]));
    let unified2 = map_colors(image2, |x| Luma([x[0].into()]));
    (unified1, unified2)
}

#[cfg(test)]
mod tests {
    use super::*;

    use imageproc::{gray_image, rgb_image};

    #[test]
    fn test_unify_gray_u8_u16() {
        let image_u8 = gray_image!(
            10, 20;
            30, 40);

        let image_u16 = gray_image!(type: u16,
            1000, 2000;
            3000, 4000);

        let (unified1, unified2) = unify_gray_images(&image_u8, &image_u16);

        // Check that conversion worked correctly
        assert_eq!(unified1.get_pixel(0, 0).0[0], 10u16);
        assert_eq!(unified1.get_pixel(1, 0).0[0], 20u16);
        assert_eq!(unified2.get_pixel(0, 0).0[0], 1000u16);
        assert_eq!(unified2.get_pixel(1, 0).0[0], 2000u16);
    }

    #[test]
    fn test_unify_rgb_u8_u16() {
        let image_u8 = rgb_image!([10, 20, 30], [40, 50, 60]);

        let image_u16 = rgb_image!(type: u16,
            [1000, 2000, 3000], [4000, 5000, 6000]);

        let (unified1, unified2) = unify_rgb_images(&image_u8, &image_u16);

        // Check that conversion worked correctly
        assert_eq!(unified1.get_pixel(0, 0).0, [10u16, 20u16, 30u16]);
        assert_eq!(unified1.get_pixel(1, 0).0, [40u16, 50u16, 60u16]);
        assert_eq!(unified2.get_pixel(0, 0).0, [1000u16, 2000u16, 3000u16]);
        assert_eq!(unified2.get_pixel(1, 0).0, [4000u16, 5000u16, 6000u16]);
    }

    #[test]
    fn test_unify_same_types() {
        let image1 = gray_image!(10, 20);
        let image2 = gray_image!(30, 40);

        let (unified1, unified2) = unify_gray_images(&image1, &image2);

        // Should work even with same types
        assert_eq!(unified1.get_pixel(0, 0).0[0], 10u8);
        assert_eq!(unified2.get_pixel(0, 0).0[0], 30u8);
    }

    #[test]
    fn test_unify_gray_u8_f32() {
        let image_u8 = gray_image!(100, 200);

        let image_f32 = gray_image!(type: f32,
            0.5, 0.8);

        let (unified1, unified2) = unify_gray_images(&image_u8, &image_f32);

        // u8 should be converted to f32
        assert_eq!(unified1.get_pixel(0, 0).0[0], 100.0f32);
        assert_eq!(unified1.get_pixel(1, 0).0[0], 200.0f32);
        assert_eq!(unified2.get_pixel(0, 0).0[0], 0.5f32);
        assert_eq!(unified2.get_pixel(1, 0).0[0], 0.8f32);
    }
}
