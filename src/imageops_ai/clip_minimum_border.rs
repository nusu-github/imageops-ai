use image::{GenericImageView, Luma, LumaA, Pixel, Primitive};
use imageproc::definitions::{Clamp, Image};
use imageproc::map::map_colors;

pub trait ClipMinimumBorder {
    fn clip_minimum_border(&self, iterations: usize, threshold: u8) -> Self;
}

impl<P, S> ClipMinimumBorder for Image<P>
where
    P: Pixel<Subpixel = S>,
    S: Primitive + Into<f32> + Clamp<f32>,
{
    fn clip_minimum_border(&self, iterations: usize, threshold: u8) -> Self {
        let mut image = self.clone();
        for i in 0..iterations {
            let corners = image.extract_corners();
            let background = &corners[i % 4];
            let [x, y, w, h] = image.find_content_bounds(background, threshold);

            if w == 0 || h == 0 {
                break;
            }

            image = image.view(x, y, w, h).inner().to_owned();
        }
        image
    }
}

trait ImageProcessing<P: Pixel> {
    fn extract_corners(&self) -> [Luma<P::Subpixel>; 4];
    fn find_content_bounds(&self, background: &Luma<P::Subpixel>, threshold: u8) -> [u32; 4];
    fn create_difference_map(&self, background: &Luma<P::Subpixel>) -> Image<Luma<u8>>;
}

impl<P: Pixel> ImageProcessing<P> for Image<P>
where
    P::Subpixel: Into<f32> + Primitive + Clamp<f32>,
{
    fn extract_corners(&self) -> [Luma<P::Subpixel>; 4] {
        let (width, height) = self.dimensions();
        [
            merge_alpha(self.get_pixel(0, 0).to_luma_alpha()),
            merge_alpha(self.get_pixel(width.saturating_sub(1), 0).to_luma_alpha()),
            merge_alpha(self.get_pixel(0, height.saturating_sub(1)).to_luma_alpha()),
            merge_alpha(
                self.get_pixel(width.saturating_sub(1), height.saturating_sub(1))
                    .to_luma_alpha(),
            ),
        ]
    }

    fn find_content_bounds(&self, background: &Luma<P::Subpixel>, threshold: u8) -> [u32; 4] {
        // Use map_colors to create a difference image for more efficient processing
        let diff_image = self.create_difference_map(background);

        let (width, height) = self.dimensions();
        let mut bounds = [width, height, 0, 0]; // [x1, y1, x2, y2]

        for (x, y, pixel) in diff_image.enumerate_pixels() {
            if pixel[0] > threshold {
                update_bounds(&mut bounds, x, y);
            }
        }

        [
            bounds[0],
            bounds[1],
            bounds[2].saturating_sub(bounds[0]),
            bounds[3].saturating_sub(bounds[1]),
        ]
    }

    fn create_difference_map(&self, background: &Luma<P::Subpixel>) -> Image<Luma<u8>> {
        let background_value = background[0].into();
        let max = P::Subpixel::DEFAULT_MAX_VALUE.into();
        let background_normalized = background_value / max * 255.0;
        let background_u8 = P::Subpixel::clamp(background_normalized);

        // Use map_colors to efficiently transform all pixels to difference values
        map_colors(self, |pixel| {
            let pixel_luma = merge_alpha(pixel.to_luma_alpha());
            let pixel_value = pixel_luma[0].into();
            let pixel_normalized = pixel_value / max * 255.0;
            let pixel_u8 = P::Subpixel::clamp(pixel_normalized);

            // Calculate absolute difference as u8
            let pixel_val: f32 = pixel_u8.into();
            let bg_val: f32 = background_u8.into();
            let diff = P::Subpixel::clamp((pixel_val - bg_val).abs());

            Luma([diff.into() as u8])
        })
    }
}

fn merge_alpha<S>(pixel: LumaA<S>) -> Luma<S>
where
    S: Primitive + Into<f32> + Clamp<f32>,
{
    let max = S::DEFAULT_MAX_VALUE.into();
    let LumaA([l, a]) = pixel;
    let l_f32 = l.into();
    let a_f32 = a.into() / max;
    let result = S::clamp(l_f32 * a_f32);
    Luma([result])
}

fn update_bounds(bounds: &mut [u32; 4], x: u32, y: u32) {
    bounds[0] = bounds[0].min(x);
    bounds[1] = bounds[1].min(y);
    bounds[2] = bounds[2].max(x);
    bounds[3] = bounds[3].max(y);
}
