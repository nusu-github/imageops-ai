use crate::imageops_ai::box_filter::BoxFilter;
use crate::Image;
use image::{ImageBuffer, Luma, LumaA, Pixel, Primitive, Rgb, Rgb32FImage, Rgba};

pub trait MargeAlpha {
    type Output;
    fn marge_alpha(&self) -> Self::Output;
}

impl<S> MargeAlpha for Image<LumaA<S>>
where
    LumaA<S>: Pixel<Subpixel = S>,
    Luma<S>: Pixel<Subpixel = S>,
    S: Primitive + 'static,
    f32: From<S>,
{
    type Output = Image<Luma<S>>;

    fn marge_alpha(&self) -> Self::Output {
        let max_value = f32::from(S::DEFAULT_MAX_VALUE);
        ImageBuffer::from_fn(self.width(), self.height(), |x, y| unsafe {
            let LumaA([luminance, alpha]) = *self.get_pixel(x, y);
            let alpha_normalized = f32::from(alpha) / max_value;
            let luminance_f32 = f32::from(luminance);
            Luma([S::from(luminance_f32 * alpha_normalized).unwrap_unchecked()])
        })
    }
}

impl<S> MargeAlpha for Image<Rgba<S>>
where
    Rgba<S>: Pixel<Subpixel = S>,
    Rgb<S>: Pixel<Subpixel = S>,
    S: Primitive + 'static,
    f32: From<S>,
{
    type Output = Image<Rgb<S>>;

    fn marge_alpha(&self) -> Self::Output {
        let max_value = f32::from(S::DEFAULT_MAX_VALUE);
        ImageBuffer::from_fn(self.width(), self.height(), |x, y| unsafe {
            let Rgba([red, green, blue, alpha]) = *self.get_pixel(x, y);
            let alpha_normalized = f32::from(alpha) / max_value;
            let red_f32 = f32::from(red);
            let green_f32 = f32::from(green);
            let blue_f32 = f32::from(blue);
            Rgb([
                S::from(red_f32 * alpha_normalized).unwrap_unchecked(),
                S::from(green_f32 * alpha_normalized).unwrap_unchecked(),
                S::from(blue_f32 * alpha_normalized).unwrap_unchecked(),
            ])
        })
    }
}

pub trait ForegroundEstimator<S>
where
    Rgb<S>: Pixel<Subpixel = S>,
    Luma<S>: Pixel<Subpixel = S>,
    S: Primitive + 'static,
    f32: From<S>,
{
    fn estimate_foreground<SM>(self, mask: &Image<Luma<SM>>, r: u32) -> Image<Rgb<S>>
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
    fn estimate_foreground<SM>(self, mask: &Image<Luma<SM>>, r: u32) -> Self
    where
        SM: Primitive + 'static,
        f32: From<SM>,
    {
        let max = f32::from(S::DEFAULT_MAX_VALUE);
        let image = unsafe {
            ImageBuffer::from_raw(
                self.width(),
                self.height(),
                self.iter()
                    .map(|x| f32::from(*x) / max)
                    .collect(),
            )
            .unwrap_unchecked()
        };
        let mask_max = f32::from(SM::DEFAULT_MAX_VALUE);
        let alpha = unsafe {
            ImageBuffer::from_raw(
                mask.width(),
                mask.height(),
                mask.iter()
                    .map(|x| f32::from(*x) / mask_max)
                    .collect(),
            )
            .unwrap_unchecked()
        };

        let image = estimate(&image, &alpha, r);

        unsafe {
            ImageBuffer::from_raw(
                image.width(),
                image.height(),
                image
                    .iter()
                    .map(|x| S::from(x * max).unwrap_unchecked())
                    .collect(),
            )
            .unwrap_unchecked()
        }
    }
}

fn estimate(image: &Rgb32FImage, alpha: &Image<Luma<f32>>, r: u32) -> Rgb32FImage {
    let (f, blur_b) = blur_fusion_estimator(image, image, image, alpha, r);
    let (f, _) = blur_fusion_estimator(image, &f, &blur_b, alpha, 6);
    f
}

fn blur_fusion_estimator(
    image: &Rgb32FImage,
    f: &Rgb32FImage,
    b: &Rgb32FImage,
    alpha: &Image<Luma<f32>>,
    r: u32,
) -> (Rgb32FImage, Rgb32FImage) {
    const MIN_DENOMINATOR: f32 = 1e-5;

    let blurred_alpha = alpha.box_filter(r, r).expect("blurred alpha");
    let blurred_fa: Rgb32FImage = unsafe {
        ImageBuffer::from_raw(
            f.width(),
            f.height(),
            f.pixels()
                .zip(image.pixels())
                .flat_map(|(f_pixel, image_pixel)| {
                    let Rgb([f_r, f_g, f_b]) = f_pixel;
                    let Rgb([r, g, b]) = image_pixel;
                    [f_r * r, f_g * g, f_b * b]
                })
                .collect(),
        )
        .unwrap_unchecked()
    };
    let blurred_fa = blurred_fa.box_filter(r, r).expect("blurred fa");
    let blurred_f: Rgb32FImage = unsafe {
        ImageBuffer::from_raw(
            blurred_fa.width(),
            blurred_fa.height(),
            blurred_fa
                .pixels()
                .zip(blurred_alpha.pixels())
                .flat_map(|(fa_pixel, alpha_pixel)| {
                    let Rgb([fa_r, fa_g, fa_b]) = fa_pixel;
                    let Luma([a]) = alpha_pixel;
                    let a = a + MIN_DENOMINATOR;
                    [fa_r / a, fa_g / a, fa_b / a]
                })
                .collect(),
        )
        .unwrap_unchecked()
    };

    let blurred_b1a: Rgb32FImage = unsafe {
        ImageBuffer::from_raw(
            b.width(),
            b.height(),
            b.pixels()
                .zip(alpha.pixels())
                .flat_map(|(b_pixel, alpha_pixel)| {
                    let Rgb([b_r, b_g, b_b]) = b_pixel;
                    let Luma([a]) = alpha_pixel;
                    let a = 1.0 - a;
                    [b_r * a, b_g * a, b_b * a]
                })
                .collect(),
        )
        .unwrap_unchecked()
    };
    let blurred_b1a = blurred_b1a.box_filter(r, r).expect("blurred b1a");
    let blurred_b: Rgb32FImage = unsafe {
        ImageBuffer::from_raw(
            blurred_b1a.width(),
            blurred_b1a.height(),
            blurred_b1a
                .pixels()
                .zip(blurred_alpha.pixels())
                .flat_map(|(b1a_pixel, alpha_pixel)| {
                    let Rgb([b_r, b_g, b_b]) = b1a_pixel;
                    let Luma([a]) = alpha_pixel;
                    let a = (1.0 - a) + MIN_DENOMINATOR;
                    [b_r / a, b_g / a, b_b / a]
                })
                .collect(),
        )
        .unwrap_unchecked()
    };

    let updated_f: Rgb32FImage = unsafe {
        ImageBuffer::from_raw(
            blurred_f.width(),
            blurred_f.height(),
            blurred_f
                .pixels()
                .zip(image.pixels())
                .zip(alpha.pixels())
                .zip(blurred_f.pixels())
                .zip(blurred_b.pixels())
                .flat_map(
                    |(
                        (((f_pixel, image_pixel), alpha_pixel), f_blurred_pixel),
                        b_blurred_pixel,
                    )| {
                        let Rgb([f_r, f_g, f_b]) = f_pixel;
                        let Rgb([r, g, b]) = image_pixel;
                        let Luma([a]) = alpha_pixel;
                        let Rgb([f_blurred_r, f_blurred_g, f_blurred_b]) = f_blurred_pixel;
                        let Rgb([b_blurred_r, b_blurred_g, b_blurred_b]) = b_blurred_pixel;
                        let a = a + MIN_DENOMINATOR;
                        let updated_r = f_r + a * (r - a * f_blurred_r - (1.0 - a) * b_blurred_r);
                        let updated_g = f_g + a * (g - a * f_blurred_g - (1.0 - a) * b_blurred_g);
                        let updated_b = f_b + a * (b - a * f_blurred_b - (1.0 - a) * b_blurred_b);
                        let clip = |channel: f32| channel.max(0.0).min(1.0);
                        [clip(updated_r), clip(updated_g), clip(updated_b)]
                    },
                )
                .collect(),
        )
        .unwrap_unchecked()
    };

    (updated_f, blurred_b)
}
