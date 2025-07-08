use crate::error::GuidedFilterError;
use crate::imageops_ai::box_filter::{BoxFilter, BoxFilterIntegral};
use image::{GenericImage, GenericImageView, ImageBuffer, Luma, Pixel, Primitive, Rgb};
use imageproc::definitions::Image;
use imageproc::map::{map_colors, map_colors2, map_pixels, map_subpixels};

pub trait GuidedFilterExtension<P>
where
    P: Pixel,
{
    fn guided_filter(
        self,
        guidance: &Image<P>,
        radius: u32,
        epsilon: f32,
    ) -> Result<Image<P>, GuidedFilterError>;
    fn guided_filter_mut(
        &mut self,
        guidance: &Image<P>,
        radius: u32,
        epsilon: f32,
    ) -> Result<&mut Self, GuidedFilterError>;
    fn fast_guided_filter(
        self,
        guidance: &Image<P>,
        radius: u32,
        epsilon: f32,
        scale: u32,
    ) -> Result<Image<P>, GuidedFilterError>;
}

pub trait GuidedFilterWithColorGuidance {
    fn guided_filter_with_color_guidance(
        self,
        guidance: &Image<Rgb<u8>>,
        radius: u32,
        epsilon: f32,
    ) -> Result<Image<Luma<u8>>, GuidedFilterError>;
    fn guided_filter_with_color_guidance_mut(
        &mut self,
        guidance: &Image<Rgb<u8>>,
        radius: u32,
        epsilon: f32,
    ) -> Result<&mut Self, GuidedFilterError>;
    fn fast_guided_filter_with_color_guidance(
        self,
        guidance: &Image<Rgb<u8>>,
        radius: u32,
        epsilon: f32,
        scale: u32,
    ) -> Result<Image<Luma<u8>>, GuidedFilterError>;
}

impl GuidedFilterWithColorGuidance for Image<Luma<u8>> {
    fn guided_filter_with_color_guidance(
        self,
        guidance: &Image<Rgb<u8>>,
        radius: u32,
        epsilon: f32,
    ) -> Result<Image<Luma<u8>>, GuidedFilterError> {
        validate_filter_input(&self, guidance, radius, epsilon)?;

        let filter = GuidedFilterColor::new(guidance, radius, epsilon);
        let result_f32 = filter.filter(&self);
        Ok(convert_f32_to_u8_luma(&result_f32))
    }

    fn guided_filter_with_color_guidance_mut(
        &mut self,
        guidance: &Image<Rgb<u8>>,
        radius: u32,
        epsilon: f32,
    ) -> Result<&mut Self, GuidedFilterError> {
        validate_filter_input(self, guidance, radius, epsilon)?;

        let filter = GuidedFilterColor::new(guidance, radius, epsilon);
        let result_f32 = filter.filter(self);
        let result_u8 = convert_f32_to_u8_luma(&result_f32);
        *self = result_u8;
        Ok(self)
    }

    fn fast_guided_filter_with_color_guidance(
        self,
        guidance: &Image<Rgb<u8>>,
        radius: u32,
        epsilon: f32,
        scale: u32,
    ) -> Result<Image<Luma<u8>>, GuidedFilterError> {
        validate_fast_filter_input(&self, guidance, radius, epsilon, scale)?;

        let filter = FastGuidedFilterImpl::new_color(guidance, radius, epsilon, scale);
        let result_f32 = filter.filter(&self);
        Ok(convert_f32_to_u8_luma(&result_f32))
    }
}

impl GuidedFilterExtension<Luma<u8>> for Image<Luma<u8>> {
    fn guided_filter(
        self,
        guidance: &Self,
        radius: u32,
        epsilon: f32,
    ) -> Result<Self, GuidedFilterError> {
        validate_filter_input(&self, guidance, radius, epsilon)?;

        let filter = GuidedFilterGray::new(guidance, radius, epsilon);
        let result_f32 = filter.filter(&self);
        Ok(convert_f32_to_u8_luma(&result_f32))
    }

    fn guided_filter_mut(
        &mut self,
        guidance: &Self,
        radius: u32,
        epsilon: f32,
    ) -> Result<&mut Self, GuidedFilterError> {
        validate_filter_input(self, guidance, radius, epsilon)?;

        let filter = GuidedFilterGray::new(guidance, radius, epsilon);
        let result_f32 = filter.filter(self);
        let result_u8 = convert_f32_to_u8_luma(&result_f32);
        *self = result_u8;
        Ok(self)
    }

    fn fast_guided_filter(
        self,
        guidance: &Self,
        radius: u32,
        epsilon: f32,
        scale: u32,
    ) -> Result<Self, GuidedFilterError> {
        validate_fast_filter_input(&self, guidance, radius, epsilon, scale)?;

        let filter = FastGuidedFilterImpl::new_gray(guidance, radius, epsilon, scale);
        let result_f32 = filter.filter(&self);
        Ok(convert_f32_to_u8_luma(&result_f32))
    }
}

impl GuidedFilterExtension<Luma<f32>> for Image<Luma<f32>> {
    fn guided_filter(
        self,
        guidance: &Self,
        radius: u32,
        epsilon: f32,
    ) -> Result<Self, GuidedFilterError> {
        validate_filter_input(&self, guidance, radius, epsilon)?;

        let filter = GuidedFilterGray::new(guidance, radius, epsilon);
        Ok(filter.filter(&self))
    }

    fn guided_filter_mut(
        &mut self,
        guidance: &Self,
        radius: u32,
        epsilon: f32,
    ) -> Result<&mut Self, GuidedFilterError> {
        validate_filter_input(self, guidance, radius, epsilon)?;

        let filter = GuidedFilterGray::new(guidance, radius, epsilon);
        let result = filter.filter(self);
        *self = result;
        Ok(self)
    }

    fn fast_guided_filter(
        self,
        guidance: &Self,
        radius: u32,
        epsilon: f32,
        scale: u32,
    ) -> Result<Self, GuidedFilterError> {
        validate_fast_filter_input(&self, guidance, radius, epsilon, scale)?;

        let filter = FastGuidedFilterImpl::new_gray(guidance, radius, epsilon, scale);
        Ok(filter.filter(&self))
    }
}

fn validate_guided_filter_params(radius: u32, epsilon: f32) -> Result<(), GuidedFilterError> {
    if radius == 0 {
        return Err(GuidedFilterError::InvalidRadius { radius });
    }
    if epsilon <= 0.0 {
        return Err(GuidedFilterError::InvalidEpsilon { epsilon });
    }
    Ok(())
}

const fn validate_scale(scale: u32) -> Result<(), GuidedFilterError> {
    if scale <= 1 {
        return Err(GuidedFilterError::InvalidScale { scale });
    }
    Ok(())
}

fn validate_dimensions<P1, P2>(
    input: &Image<P1>,
    guidance: &Image<P2>,
) -> Result<(), GuidedFilterError>
where
    P1: Pixel,
    P2: Pixel,
{
    let input_dims = input.dimensions();
    let guidance_dims = guidance.dimensions();

    if input_dims != guidance_dims {
        return Err(GuidedFilterError::DimensionMismatch {
            guidance_dims,
            input_dims,
        });
    }
    Ok(())
}

// Removed validate_dimensions_mixed - was just calling validate_dimensions

fn validate_filter_input<P1, P2>(
    input: &Image<P1>,
    guidance: &Image<P2>,
    radius: u32,
    epsilon: f32,
) -> Result<(), GuidedFilterError>
where
    P1: Pixel,
    P2: Pixel,
{
    validate_guided_filter_params(radius, epsilon)?;
    validate_dimensions(input, guidance)?;
    Ok(())
}

fn validate_fast_filter_input<P1, P2>(
    input: &Image<P1>,
    guidance: &Image<P2>,
    radius: u32,
    epsilon: f32,
    scale: u32,
) -> Result<(), GuidedFilterError>
where
    P1: Pixel,
    P2: Pixel,
{
    validate_guided_filter_params(radius, epsilon)?;
    validate_scale(scale)?;
    validate_dimensions(input, guidance)?;
    Ok(())
}

fn convert_f32_to_u8_luma(image: &Image<Luma<f32>>) -> Image<Luma<u8>> {
    map_subpixels(image, |p| p.clamp(0.0, 255.0) as u8)
}

pub fn box_filter(image: &Image<Luma<f32>>, radius: u32) -> Image<Luma<f32>> {
    let filter = BoxFilterIntegral::new(radius).expect("Invalid radius");
    filter.filter(image).expect("Box filter failed")
}

pub fn box_filter_rgb(image: &Image<Rgb<f32>>, radius: u32) -> Image<Rgb<f32>> {
    let filter = BoxFilterIntegral::new(radius).expect("Invalid radius");
    filter.filter(image).expect("Box filter failed")
}

pub fn resize_image<P>(image: &Image<P>, new_width: u32, new_height: u32) -> Image<P>
where
    P: Pixel + 'static,
    P::Subpixel: 'static,
{
    use image::imageops;

    imageops::resize(image, new_width, new_height, imageops::FilterType::Nearest)
}

pub fn resize_image_with_filter<P>(
    image: &Image<P>,
    new_width: u32,
    new_height: u32,
    filter: image::imageops::FilterType,
) -> Image<P>
where
    P: Pixel + 'static,
    P::Subpixel: 'static,
{
    use image::imageops;

    imageops::resize(image, new_width, new_height, filter)
}

pub fn to_f32_luma<T>(image: &Image<Luma<T>>) -> Image<Luma<f32>>
where
    T: Primitive + Into<f32>,
    Luma<T>: Pixel<Subpixel = T>,
{
    map_colors(image, |p| Luma([p[0].into()]))
}

pub fn to_f32_rgb<T>(image: &Image<Rgb<T>>) -> Image<Rgb<f32>>
where
    T: Primitive + Into<f32>,
    Rgb<T>: Pixel<Subpixel = T>,
{
    map_colors(image, |p| Rgb([p[0].into(), p[1].into(), p[2].into()]))
}

pub struct GuidedFilterGray {
    guidance: Image<Luma<f32>>,
    radius: u32,
    epsilon: f32,
    guidance_mean: Image<Luma<f32>>,
    guidance_var: Image<Luma<f32>>,
}

impl GuidedFilterGray {
    pub fn new<T>(guidance: &Image<Luma<T>>, radius: u32, epsilon: f32) -> Self
    where
        T: Primitive + Into<f32>,
        Luma<T>: Pixel<Subpixel = T>,
    {
        let guidance_f32 = to_f32_luma(guidance);
        let guidance_mean = box_filter(&guidance_f32, radius);

        // Compute squared guidance values
        let guidance_sq = map_colors(&guidance_f32, |p| Luma([p[0] * p[0]]));

        let guidance_sq_mean = box_filter(&guidance_sq, radius);

        let guidance_var = map_colors2(&guidance_mean, &guidance_sq_mean, |mean_p, sq_mean_p| {
            let mean_val = mean_p[0];
            let sq_mean_val = sq_mean_p[0];
            let var_val = mean_val.mul_add(-mean_val, sq_mean_val);
            Luma([var_val])
        });

        Self {
            guidance: guidance_f32,
            radius,
            epsilon,
            guidance_mean,
            guidance_var,
        }
    }

    pub fn filter<T>(&self, input: &Image<Luma<T>>) -> Image<Luma<f32>>
    where
        T: Primitive + Into<f32>,
        Luma<T>: Pixel<Subpixel = T>,
    {
        let input_f32 = to_f32_luma(input);
        let input_mean = box_filter(&input_f32, self.radius);

        let (width, height) = input_f32.dimensions();

        // Compute I * p
        let input_guidance_prod = map_colors2(&input_f32, &self.guidance, |input_p, guidance_p| {
            Luma([input_p[0] * guidance_p[0]])
        });

        // E[I * p]
        let input_guidance_mean = box_filter(&input_guidance_prod, self.radius);

        let mut a = ImageBuffer::new(width, height);
        let mut b = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let ip_mean = unsafe { input_guidance_mean.unsafe_get_pixel(x, y)[0] };
                let input_mean_val = unsafe { input_mean.unsafe_get_pixel(x, y)[0] };
                let guidance_mean_val = unsafe { self.guidance_mean.unsafe_get_pixel(x, y)[0] };
                let guidance_var_val = unsafe { self.guidance_var.unsafe_get_pixel(x, y)[0] };

                // cov(I, p) = E[I * p] - E[I] * E[p]
                let cov = guidance_mean_val.mul_add(-input_mean_val, ip_mean);
                let a_val = cov / (guidance_var_val + self.epsilon);
                let b_val = a_val.mul_add(-guidance_mean_val, input_mean_val);

                unsafe { a.unsafe_put_pixel(x, y, Luma([a_val])) };
                unsafe { b.unsafe_put_pixel(x, y, Luma([b_val])) };
            }
        }

        let a_mean = box_filter(&a, self.radius);
        let b_mean = box_filter(&b, self.radius);

        map_pixels(&a_mean, |x, y, _| {
            let a_val = unsafe { a_mean.unsafe_get_pixel(x, y)[0] };
            let b_val = unsafe { b_mean.unsafe_get_pixel(x, y)[0] };
            let guidance_val = unsafe { self.guidance.unsafe_get_pixel(x, y)[0] };

            let output_val = a_val.mul_add(guidance_val, b_val);
            Luma([output_val])
        })
    }
}

pub struct GuidedFilterColor {
    guidance: Image<Rgb<f32>>,
    radius: u32,
    epsilon: f32,
    guidance_mean: Image<Rgb<f32>>,
    inv_cov: [Image<Luma<f32>>; 6], // [Irr, Irg, Irb, Igg, Igb, Ibb]
}

pub struct FastGuidedFilterImpl {
    guidance: Image<Luma<f32>>,
    guidance_color: Option<Image<Rgb<f32>>>,
    radius: u32,
    epsilon: f32,
    scale: u32,
    gray_filter: Option<GuidedFilterGray>,
    color_filter: Option<GuidedFilterColor>,
}

impl FastGuidedFilterImpl {
    // Common helper for initialization parameters
    fn compute_scale_params(width: u32, height: u32, scale: u32, radius: u32) -> (u32, u32, u32) {
        let new_width = width / scale;
        let new_height = height / scale;
        let scaled_radius = (radius as f32 / scale as f32).max(1.0) as u32;
        (new_width, new_height, scaled_radius)
    }
    pub fn new_gray<T>(guidance: &Image<Luma<T>>, radius: u32, epsilon: f32, scale: u32) -> Self
    where
        T: Primitive + Into<f32>,
        Luma<T>: Pixel<Subpixel = T>,
    {
        let guidance_f32 = to_f32_luma(guidance);
        let (width, height) = guidance_f32.dimensions();
        let (new_width, new_height, scaled_radius) =
            Self::compute_scale_params(width, height, scale, radius);

        // ダウンサンプリング用のリサイズ関数を作成
        let guidance_sub = resize_image(&guidance_f32, new_width, new_height);

        // Compute mean
        let guidance_mean = box_filter(&guidance_sub, scaled_radius);

        // Compute squared values
        let (sub_width, sub_height) = guidance_sub.dimensions();
        let mut guidance_sq = ImageBuffer::new(sub_width, sub_height);
        for y in 0..sub_height {
            for x in 0..sub_width {
                let val = unsafe { guidance_sub.unsafe_get_pixel(x, y)[0] };
                unsafe { guidance_sq.unsafe_put_pixel(x, y, Luma([val * val])) };
            }
        }

        // Compute mean of squared values
        let guidance_sq_mean = box_filter(&guidance_sq, scaled_radius);

        let mut guidance_var = ImageBuffer::new(sub_width, sub_height);

        for y in 0..sub_height {
            for x in 0..sub_width {
                let mean_val = unsafe { guidance_mean.unsafe_get_pixel(x, y)[0] };
                let sq_mean_val = unsafe { guidance_sq_mean.unsafe_get_pixel(x, y)[0] };
                let var_val = mean_val.mul_add(-mean_val, sq_mean_val);
                unsafe { guidance_var.unsafe_put_pixel(x, y, Luma([var_val])) };
            }
        }

        let gray_filter = GuidedFilterGray {
            guidance: guidance_sub,
            radius: scaled_radius,
            epsilon,
            guidance_mean,
            guidance_var,
        };

        Self {
            guidance: guidance_f32,
            guidance_color: None,
            radius,
            epsilon,
            scale,
            gray_filter: Some(gray_filter),
            color_filter: None,
        }
    }

    pub fn new_color<T>(guidance: &Image<Rgb<T>>, radius: u32, epsilon: f32, scale: u32) -> Self
    where
        T: Primitive + Into<f32>,
        Rgb<T>: Pixel<Subpixel = T>,
    {
        let guidance_f32_full = to_f32_rgb(guidance);
        let (width, height) = guidance_f32_full.dimensions();
        let (new_width, new_height, scaled_radius) =
            Self::compute_scale_params(width, height, scale, radius);

        let guidance_sub = resize_image(&guidance_f32_full, new_width, new_height);
        // Create color filter directly with f32 data
        // Color filter initialization is done inside the struct initialization
        let (new_width, new_height) = guidance_sub.dimensions();

        // Create the color filter directly with f32 data
        let color_filter = GuidedFilterColor {
            guidance: guidance_sub.clone(),
            radius: scaled_radius,
            epsilon,
            guidance_mean: box_filter_rgb(&guidance_sub, scaled_radius),
            inv_cov: [
                ImageBuffer::new(new_width, new_height),
                ImageBuffer::new(new_width, new_height),
                ImageBuffer::new(new_width, new_height),
                ImageBuffer::new(new_width, new_height),
                ImageBuffer::new(new_width, new_height),
                ImageBuffer::new(new_width, new_height),
            ],
        };

        // Initialize the inverse covariance matrices
        let mut color_filter = color_filter;
        let guidance_f32 = &color_filter.guidance;
        let guidance_mean = &color_filter.guidance_mean;
        // Using scaled_radius directly in computations

        // Compute covariance matrix elements
        let (width, height) = guidance_f32.dimensions();
        let mut rr_cov = ImageBuffer::new(width, height);
        let mut rg_cov = ImageBuffer::new(width, height);
        let mut rb_cov = ImageBuffer::new(width, height);
        let mut gg_cov = ImageBuffer::new(width, height);
        let mut gb_cov = ImageBuffer::new(width, height);
        let mut bb_cov = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                unsafe {
                    let Rgb([r, g, b]) = guidance_f32.unsafe_get_pixel(x, y);
                    rr_cov.unsafe_put_pixel(x, y, Luma([r * r]));
                    rg_cov.unsafe_put_pixel(x, y, Luma([r * g]));
                    rb_cov.unsafe_put_pixel(x, y, Luma([r * b]));
                    gg_cov.unsafe_put_pixel(x, y, Luma([g * g]));
                    gb_cov.unsafe_put_pixel(x, y, Luma([g * b]));
                    bb_cov.unsafe_put_pixel(x, y, Luma([b * b]));
                }
            }
        }

        let rr_mean = box_filter(&rr_cov, scaled_radius);
        let rg_mean = box_filter(&rg_cov, scaled_radius);
        let rb_mean = box_filter(&rb_cov, scaled_radius);
        let gg_mean = box_filter(&gg_cov, scaled_radius);
        let gb_mean = box_filter(&gb_cov, scaled_radius);
        let bb_mean = box_filter(&bb_cov, scaled_radius);

        // Compute inverse covariance matrix
        for y in 0..height {
            for x in 0..width {
                let Rgb([mean_r, mean_g, mean_b]) = unsafe { guidance_mean.unsafe_get_pixel(x, y) };

                // Covariance matrix elements (epsilon added to diagonal)
                let cov_rr =
                    mean_r.mul_add(-mean_r, unsafe { rr_mean.unsafe_get_pixel(x, y)[0] }) + epsilon;
                let cov_rg = mean_r.mul_add(-mean_g, unsafe { rg_mean.unsafe_get_pixel(x, y)[0] });
                let cov_rb = mean_r.mul_add(-mean_b, unsafe { rb_mean.unsafe_get_pixel(x, y)[0] });
                let cov_gg =
                    mean_g.mul_add(-mean_g, unsafe { gg_mean.unsafe_get_pixel(x, y)[0] }) + epsilon;
                let cov_gb = mean_g.mul_add(-mean_b, unsafe { gb_mean.unsafe_get_pixel(x, y)[0] });
                let cov_bb =
                    mean_b.mul_add(-mean_b, unsafe { bb_mean.unsafe_get_pixel(x, y)[0] }) + epsilon;

                // Inverse using cofactor method
                let det = cov_rb.mul_add(
                    cov_rg.mul_add(cov_gb, -(cov_gg * cov_rb)),
                    cov_rr.mul_add(
                        cov_gg.mul_add(cov_bb, -(cov_gb * cov_gb)),
                        -(cov_rg * cov_rg.mul_add(cov_bb, -(cov_rb * cov_gb))),
                    ),
                );

                if det.abs() > 1e-12 {
                    let inv_det = 1.0 / det;

                    unsafe {
                        color_filter.inv_cov[0].unsafe_put_pixel(
                            x,
                            y,
                            Luma([cov_gg.mul_add(cov_bb, -(cov_gb * cov_gb)) * inv_det]),
                        )
                    };
                    unsafe {
                        color_filter.inv_cov[1].unsafe_put_pixel(
                            x,
                            y,
                            Luma([cov_rb.mul_add(cov_gb, -(cov_rg * cov_bb)) * inv_det]),
                        )
                    };
                    unsafe {
                        color_filter.inv_cov[2].unsafe_put_pixel(
                            x,
                            y,
                            Luma([cov_rg.mul_add(cov_gb, -(cov_gg * cov_rb)) * inv_det]),
                        )
                    };
                    unsafe {
                        color_filter.inv_cov[3].unsafe_put_pixel(
                            x,
                            y,
                            Luma([cov_rr.mul_add(cov_bb, -(cov_rb * cov_rb)) * inv_det]),
                        )
                    };
                    unsafe {
                        color_filter.inv_cov[4].unsafe_put_pixel(
                            x,
                            y,
                            Luma([cov_rb.mul_add(cov_rg, -(cov_rr * cov_gb)) * inv_det]),
                        )
                    };
                    unsafe {
                        color_filter.inv_cov[5].unsafe_put_pixel(
                            x,
                            y,
                            Luma([cov_rr.mul_add(cov_gg, -(cov_rg * cov_rg)) * inv_det]),
                        )
                    };
                } else {
                    for i in 0..6 {
                        unsafe { color_filter.inv_cov[i].unsafe_put_pixel(x, y, Luma([0.0])) };
                    }
                }
            }
        }

        Self {
            guidance: ImageBuffer::new(width, height), // Dummy gray image
            guidance_color: Some(guidance_f32_full),
            radius,
            epsilon,
            scale,
            gray_filter: None,
            color_filter: Some(color_filter),
        }
    }

    // 係数a, bを計算するヘルパーメソッド
    fn compute_coefficients_gray(
        &self,
        input_sub: &Image<Luma<f32>>,
        gray_filter: &GuidedFilterGray,
    ) -> (Image<Luma<f32>>, Image<Luma<f32>>) {
        let input_mean = box_filter(input_sub, gray_filter.radius);
        let (width, height) = input_sub.dimensions();

        // Compute I * p
        let mut input_guidance_prod = ImageBuffer::new(width, height);
        for y in 0..height {
            for x in 0..width {
                unsafe {
                    let Luma([input_val]) = input_sub.unsafe_get_pixel(x, y);
                    let Luma([guidance_val]) = gray_filter.guidance.unsafe_get_pixel(x, y);
                    input_guidance_prod.unsafe_put_pixel(x, y, Luma([input_val * guidance_val]));
                }
            }
        }

        // E[I * p]
        let input_guidance_mean = box_filter(&input_guidance_prod, gray_filter.radius);

        let mut a = ImageBuffer::new(width, height);
        let mut b = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                unsafe {
                    let Luma([ip_mean]) = input_guidance_mean.unsafe_get_pixel(x, y);
                    let Luma([input_mean_val]) = input_mean.unsafe_get_pixel(x, y);
                    let Luma([guidance_mean_val]) =
                        gray_filter.guidance_mean.unsafe_get_pixel(x, y);
                    let Luma([guidance_var_val]) = gray_filter.guidance_var.unsafe_get_pixel(x, y);

                    // cov(I, p) = E[I * p] - E[I] * E[p]
                    let cov = guidance_mean_val.mul_add(-input_mean_val, ip_mean);
                    let a_val = cov / (guidance_var_val + gray_filter.epsilon);
                    let b_val = a_val.mul_add(-guidance_mean_val, input_mean_val);

                    a.unsafe_put_pixel(x, y, Luma([a_val]));
                    b.unsafe_put_pixel(x, y, Luma([b_val]));
                }
            }
        }

        let a_mean = box_filter(&a, gray_filter.radius);
        let b_mean = box_filter(&b, gray_filter.radius);

        (a_mean, b_mean)
    }

    // カラー係数a, bを計算するヘルパーメソッド
    fn compute_coefficients_color(
        &self,
        input_sub: &Image<Luma<f32>>,
        color_filter: &GuidedFilterColor,
    ) -> (
        Image<Luma<f32>>,
        Image<Luma<f32>>,
        Image<Luma<f32>>,
        Image<Luma<f32>>,
    ) {
        let input_mean = box_filter(input_sub, color_filter.radius);
        let (width, height) = input_sub.dimensions();

        // Compute input-guidance covariances
        let mut ip_r = ImageBuffer::new(width, height);
        let mut ip_g = ImageBuffer::new(width, height);
        let mut ip_b = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                unsafe {
                    let Luma([input_val]) = input_sub.unsafe_get_pixel(x, y);
                    let Rgb([r, g, b]) = color_filter.guidance.unsafe_get_pixel(x, y);
                    ip_r.unsafe_put_pixel(x, y, Luma([input_val * r]));
                    ip_g.unsafe_put_pixel(x, y, Luma([input_val * g]));
                    ip_b.unsafe_put_pixel(x, y, Luma([input_val * b]));
                }
            }
        }

        let ip_r_mean = box_filter(&ip_r, color_filter.radius);
        let ip_g_mean = box_filter(&ip_g, color_filter.radius);
        let ip_b_mean = box_filter(&ip_b, color_filter.radius);

        // Compute coefficients a and b
        let mut a_r = ImageBuffer::new(width, height);
        let mut a_g = ImageBuffer::new(width, height);
        let mut a_b = ImageBuffer::new(width, height);
        let mut b = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                unsafe {
                    let Luma([input_mean_val]) = input_mean.unsafe_get_pixel(x, y);
                    let Rgb([mean_r, mean_g, mean_b]) =
                        color_filter.guidance_mean.unsafe_get_pixel(x, y);

                    let cov_ip_r =
                        input_mean_val.mul_add(-mean_r, ip_r_mean.unsafe_get_pixel(x, y)[0]);
                    let cov_ip_g =
                        input_mean_val.mul_add(-mean_g, ip_g_mean.unsafe_get_pixel(x, y)[0]);
                    let cov_ip_b =
                        input_mean_val.mul_add(-mean_b, ip_b_mean.unsafe_get_pixel(x, y)[0]);

                    let Luma([inv_rr]) = color_filter.inv_cov[0].unsafe_get_pixel(x, y);
                    let Luma([inv_rg]) = color_filter.inv_cov[1].unsafe_get_pixel(x, y);
                    let Luma([inv_rb]) = color_filter.inv_cov[2].unsafe_get_pixel(x, y);
                    let Luma([inv_gg]) = color_filter.inv_cov[3].unsafe_get_pixel(x, y);
                    let Luma([inv_gb]) = color_filter.inv_cov[4].unsafe_get_pixel(x, y);
                    let Luma([inv_bb]) = color_filter.inv_cov[5].unsafe_get_pixel(x, y);

                    let a_r_val =
                        inv_rb.mul_add(cov_ip_b, inv_rr.mul_add(cov_ip_r, inv_rg * cov_ip_g));
                    let a_g_val =
                        inv_gb.mul_add(cov_ip_b, inv_rg.mul_add(cov_ip_r, inv_gg * cov_ip_g));
                    let a_b_val =
                        inv_bb.mul_add(cov_ip_b, inv_rb.mul_add(cov_ip_r, inv_gb * cov_ip_g));

                    let b_val = a_b_val.mul_add(
                        -mean_b,
                        a_g_val.mul_add(-mean_g, a_r_val.mul_add(-mean_r, input_mean_val)),
                    );

                    a_r.unsafe_put_pixel(x, y, Luma([a_r_val]));
                    a_g.unsafe_put_pixel(x, y, Luma([a_g_val]));
                    a_b.unsafe_put_pixel(x, y, Luma([a_b_val]));
                    b.unsafe_put_pixel(x, y, Luma([b_val]));
                }
            }
        }

        let a_r_mean = box_filter(&a_r, color_filter.radius);
        let a_g_mean = box_filter(&a_g, color_filter.radius);
        let a_b_mean = box_filter(&a_b, color_filter.radius);
        let b_mean = box_filter(&b, color_filter.radius);

        (a_r_mean, a_g_mean, a_b_mean, b_mean)
    }

    // 最終出力を計算するヘルパーメソッド
    fn compute_final_output_gray(
        a: &Image<Luma<f32>>,
        b: &Image<Luma<f32>>,
        guidance: &Image<Luma<f32>>,
    ) -> Image<Luma<f32>> {
        let (width, height) = guidance.dimensions();
        let mut result = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                unsafe {
                    let Luma([a_val]) = a.unsafe_get_pixel(x, y);
                    let Luma([b_val]) = b.unsafe_get_pixel(x, y);
                    let Luma([guidance_val]) = guidance.unsafe_get_pixel(x, y);

                    let output_val = a_val.mul_add(guidance_val, b_val);
                    result.unsafe_put_pixel(x, y, Luma([output_val]));
                }
            }
        }

        result
    }

    // カラー最終出力を計算するヘルパーメソッド
    fn compute_final_output_color(
        a_r: &Image<Luma<f32>>,
        a_g: &Image<Luma<f32>>,
        a_b: &Image<Luma<f32>>,
        b: &Image<Luma<f32>>,
        guidance_color: &Image<Rgb<f32>>,
    ) -> Image<Luma<f32>> {
        let (width, height) = guidance_color.dimensions();
        let mut result = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                unsafe {
                    let Luma([a_r_val]) = a_r.unsafe_get_pixel(x, y);
                    let Luma([a_g_val]) = a_g.unsafe_get_pixel(x, y);
                    let Luma([a_b_val]) = a_b.unsafe_get_pixel(x, y);
                    let Luma([b_val]) = b.unsafe_get_pixel(x, y);

                    let Rgb([guidance_r, guidance_g, guidance_b]) =
                        guidance_color.unsafe_get_pixel(x, y);

                    let output_val = a_b_val.mul_add(
                        guidance_b,
                        a_r_val.mul_add(guidance_r, a_g_val * guidance_g),
                    ) + b_val;

                    result.unsafe_put_pixel(x, y, Luma([output_val]));
                }
            }
        }

        result
    }

    pub fn filter<T>(&self, input: &Image<Luma<T>>) -> Image<Luma<f32>>
    where
        T: Primitive + Into<f32>,
        Luma<T>: Pixel<Subpixel = T>,
    {
        let input_f32 = to_f32_luma(input);
        let (width, height) = input_f32.dimensions();
        let new_width = width / self.scale;
        let new_height = height / self.scale;

        let input_sub = resize_image(&input_f32, new_width, new_height);

        // Implementation of fast guided filter
        self.gray_filter.as_ref().map_or_else(
            || {
                if let (Some(ref color_filter), Some(ref guidance_color)) =
                    (&self.color_filter, &self.guidance_color)
                {
                    // Case with color guidance
                    // Compute coefficients with subsampled image
                    let (a_r_sub, a_g_sub, a_b_sub, b_sub) =
                        self.compute_coefficients_color(&input_sub, color_filter);

                    // Upsample coefficients to full size
                    let a_r_full = resize_image_with_filter(
                        &a_r_sub,
                        width,
                        height,
                        image::imageops::FilterType::Triangle,
                    );
                    let a_g_full = resize_image_with_filter(
                        &a_g_sub,
                        width,
                        height,
                        image::imageops::FilterType::Triangle,
                    );
                    let a_b_full = resize_image_with_filter(
                        &a_b_sub,
                        width,
                        height,
                        image::imageops::FilterType::Triangle,
                    );
                    let b_full = resize_image_with_filter(
                        &b_sub,
                        width,
                        height,
                        image::imageops::FilterType::Triangle,
                    );

                    // Compute final output with full-size color guidance image
                    Self::compute_final_output_color(
                        &a_r_full,
                        &a_g_full,
                        &a_b_full,
                        &b_full,
                        guidance_color,
                    )
                } else {
                    // Return input as-is when no filter is available
                    input_f32
                }
            },
            |gray_filter| {
                // Case with grayscale guidance
                // Compute coefficients with subsampled image
                let (a_sub, b_sub) = self.compute_coefficients_gray(&input_sub, gray_filter);

                // Upsample coefficients to full size
                let a_full = resize_image_with_filter(
                    &a_sub,
                    width,
                    height,
                    image::imageops::FilterType::Triangle,
                );
                let b_full = resize_image_with_filter(
                    &b_sub,
                    width,
                    height,
                    image::imageops::FilterType::Triangle,
                );

                // Compute final output with full-size guidance image
                Self::compute_final_output_gray(&a_full, &b_full, &self.guidance)
            },
        )
    }
}

impl GuidedFilterColor {
    pub fn new<T>(guidance: &Image<Rgb<T>>, radius: u32, epsilon: f32) -> Self
    where
        T: Primitive + Into<f32>,
        Rgb<T>: Pixel<Subpixel = T>,
    {
        let guidance_f32 = to_f32_rgb(guidance);
        let guidance_mean = box_filter_rgb(&guidance_f32, radius);

        let (width, height) = guidance_f32.dimensions();

        // Compute covariance matrix elements
        let mut rr_cov = ImageBuffer::new(width, height);
        let mut rg_cov = ImageBuffer::new(width, height);
        let mut rb_cov = ImageBuffer::new(width, height);
        let mut gg_cov = ImageBuffer::new(width, height);
        let mut gb_cov = ImageBuffer::new(width, height);
        let mut bb_cov = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                unsafe {
                    let Rgb([r, g, b]) = guidance_f32.unsafe_get_pixel(x, y);
                    rr_cov.unsafe_put_pixel(x, y, Luma([r * r]));
                    rg_cov.unsafe_put_pixel(x, y, Luma([r * g]));
                    rb_cov.unsafe_put_pixel(x, y, Luma([r * b]));
                    gg_cov.unsafe_put_pixel(x, y, Luma([g * g]));
                    gb_cov.unsafe_put_pixel(x, y, Luma([g * b]));
                    bb_cov.unsafe_put_pixel(x, y, Luma([b * b]));
                }
            }
        }

        let rr_mean = box_filter(&rr_cov, radius);
        let rg_mean = box_filter(&rg_cov, radius);
        let rb_mean = box_filter(&rb_cov, radius);
        let gg_mean = box_filter(&gg_cov, radius);
        let gb_mean = box_filter(&gb_cov, radius);
        let bb_mean = box_filter(&bb_cov, radius);

        // Compute covariance matrix and its inverse
        let mut inv_rr = ImageBuffer::new(width, height);
        let mut inv_rg = ImageBuffer::new(width, height);
        let mut inv_rb = ImageBuffer::new(width, height);
        let mut inv_gg = ImageBuffer::new(width, height);
        let mut inv_gb = ImageBuffer::new(width, height);
        let mut inv_bb = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let Rgb([mean_r, mean_g, mean_b]) = unsafe { guidance_mean.unsafe_get_pixel(x, y) };

                // Covariance matrix elements (epsilon added to diagonal)
                let cov_rr =
                    mean_r.mul_add(-mean_r, unsafe { rr_mean.unsafe_get_pixel(x, y)[0] }) + epsilon;
                let cov_rg = mean_r.mul_add(-mean_g, unsafe { rg_mean.unsafe_get_pixel(x, y)[0] });
                let cov_rb = mean_r.mul_add(-mean_b, unsafe { rb_mean.unsafe_get_pixel(x, y)[0] });
                let cov_gg =
                    mean_g.mul_add(-mean_g, unsafe { gg_mean.unsafe_get_pixel(x, y)[0] }) + epsilon;
                let cov_gb = mean_g.mul_add(-mean_b, unsafe { gb_mean.unsafe_get_pixel(x, y)[0] });
                let cov_bb =
                    mean_b.mul_add(-mean_b, unsafe { bb_mean.unsafe_get_pixel(x, y)[0] }) + epsilon;

                // Inverse using cofactor method
                let det = cov_rb.mul_add(
                    cov_rg.mul_add(cov_gb, -(cov_gg * cov_rb)),
                    cov_rr.mul_add(
                        cov_gg.mul_add(cov_bb, -(cov_gb * cov_gb)),
                        -(cov_rg * cov_rg.mul_add(cov_bb, -(cov_rb * cov_gb))),
                    ),
                );

                if det.abs() > 1e-12 {
                    let inv_det = 1.0 / det;

                    unsafe {
                        inv_rr.unsafe_put_pixel(
                            x,
                            y,
                            Luma([cov_gg.mul_add(cov_bb, -(cov_gb * cov_gb)) * inv_det]),
                        )
                    };
                    unsafe {
                        inv_rg.unsafe_put_pixel(
                            x,
                            y,
                            Luma([cov_rb.mul_add(cov_gb, -(cov_rg * cov_bb)) * inv_det]),
                        )
                    };
                    unsafe {
                        inv_rb.unsafe_put_pixel(
                            x,
                            y,
                            Luma([cov_rg.mul_add(cov_gb, -(cov_gg * cov_rb)) * inv_det]),
                        )
                    };
                    unsafe {
                        inv_gg.unsafe_put_pixel(
                            x,
                            y,
                            Luma([cov_rr.mul_add(cov_bb, -(cov_rb * cov_rb)) * inv_det]),
                        )
                    };
                    unsafe {
                        inv_gb.unsafe_put_pixel(
                            x,
                            y,
                            Luma([cov_rb.mul_add(cov_rg, -(cov_rr * cov_gb)) * inv_det]),
                        )
                    };
                    unsafe {
                        inv_bb.unsafe_put_pixel(
                            x,
                            y,
                            Luma([cov_rr.mul_add(cov_gg, -(cov_rg * cov_rg)) * inv_det]),
                        )
                    };
                } else {
                    unsafe { inv_rr.unsafe_put_pixel(x, y, Luma([0.0])) };
                    unsafe { inv_rg.unsafe_put_pixel(x, y, Luma([0.0])) };
                    unsafe { inv_rb.unsafe_put_pixel(x, y, Luma([0.0])) };
                    unsafe { inv_gg.unsafe_put_pixel(x, y, Luma([0.0])) };
                    unsafe { inv_gb.unsafe_put_pixel(x, y, Luma([0.0])) };
                    unsafe { inv_bb.unsafe_put_pixel(x, y, Luma([0.0])) };
                }
            }
        }

        Self {
            guidance: guidance_f32,
            radius,
            epsilon,
            guidance_mean,
            inv_cov: [inv_rr, inv_rg, inv_rb, inv_gg, inv_gb, inv_bb],
        }
    }

    pub fn filter<T>(&self, input: &Image<Luma<T>>) -> Image<Luma<f32>>
    where
        T: Primitive + Into<f32>,
        Luma<T>: Pixel<Subpixel = T>,
    {
        let input_f32 = to_f32_luma(input);
        let input_mean = box_filter(&input_f32, self.radius);

        let (width, height) = input_f32.dimensions();

        // Compute input-guidance covariances
        let mut ip_r = ImageBuffer::new(width, height);
        let mut ip_g = ImageBuffer::new(width, height);
        let mut ip_b = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let Luma([input_val]) = unsafe { input_f32.unsafe_get_pixel(x, y) };
                let Rgb([guidance_r, guidance_g, guidance_b]) =
                    unsafe { self.guidance.unsafe_get_pixel(x, y) };

                unsafe { ip_r.unsafe_put_pixel(x, y, Luma([input_val * guidance_r])) };
                unsafe { ip_g.unsafe_put_pixel(x, y, Luma([input_val * guidance_g])) };
                unsafe { ip_b.unsafe_put_pixel(x, y, Luma([input_val * guidance_b])) };
            }
        }

        let ip_r_mean = box_filter(&ip_r, self.radius);
        let ip_g_mean = box_filter(&ip_g, self.radius);
        let ip_b_mean = box_filter(&ip_b, self.radius);

        // Compute coefficients a and b
        let mut a_r = ImageBuffer::new(width, height);
        let mut a_g = ImageBuffer::new(width, height);
        let mut a_b = ImageBuffer::new(width, height);
        let mut b = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let Luma([input_mean_val]) = unsafe { input_mean.unsafe_get_pixel(x, y) };
                let Rgb([guidance_mean_r, guidance_mean_g, guidance_mean_b]) =
                    unsafe { self.guidance_mean.unsafe_get_pixel(x, y) };

                let cov_ip_r = input_mean_val.mul_add(-guidance_mean_r, unsafe {
                    ip_r_mean.unsafe_get_pixel(x, y)[0]
                });
                let cov_ip_g = input_mean_val.mul_add(-guidance_mean_g, unsafe {
                    ip_g_mean.unsafe_get_pixel(x, y)[0]
                });
                let cov_ip_b = input_mean_val.mul_add(-guidance_mean_b, unsafe {
                    ip_b_mean.unsafe_get_pixel(x, y)[0]
                });

                let Luma([inv_rr]) = unsafe { self.inv_cov[0].unsafe_get_pixel(x, y) };
                let Luma([inv_rg]) = unsafe { self.inv_cov[1].unsafe_get_pixel(x, y) };
                let Luma([inv_rb]) = unsafe { self.inv_cov[2].unsafe_get_pixel(x, y) };
                let Luma([inv_gg]) = unsafe { self.inv_cov[3].unsafe_get_pixel(x, y) };
                let Luma([inv_gb]) = unsafe { self.inv_cov[4].unsafe_get_pixel(x, y) };
                let Luma([inv_bb]) = unsafe { self.inv_cov[5].unsafe_get_pixel(x, y) };

                let a_r_val = inv_rb.mul_add(cov_ip_b, inv_rr.mul_add(cov_ip_r, inv_rg * cov_ip_g));
                let a_g_val = inv_gb.mul_add(cov_ip_b, inv_rg.mul_add(cov_ip_r, inv_gg * cov_ip_g));
                let a_b_val = inv_bb.mul_add(cov_ip_b, inv_rb.mul_add(cov_ip_r, inv_gb * cov_ip_g));

                let b_val = a_b_val.mul_add(
                    -guidance_mean_b,
                    a_g_val.mul_add(
                        -guidance_mean_g,
                        a_r_val.mul_add(-guidance_mean_r, input_mean_val),
                    ),
                );

                unsafe { a_r.unsafe_put_pixel(x, y, Luma([a_r_val])) };
                unsafe { a_g.unsafe_put_pixel(x, y, Luma([a_g_val])) };
                unsafe { a_b.unsafe_put_pixel(x, y, Luma([a_b_val])) };
                unsafe { b.unsafe_put_pixel(x, y, Luma([b_val])) };
            }
        }

        let a_r_mean = box_filter(&a_r, self.radius);
        let a_g_mean = box_filter(&a_g, self.radius);
        let a_b_mean = box_filter(&a_b, self.radius);
        let b_mean = box_filter(&b, self.radius);

        // Compute final output
        let mut result = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let Luma([a_r_val]) = unsafe { a_r_mean.unsafe_get_pixel(x, y) };
                let Luma([a_g_val]) = unsafe { a_g_mean.unsafe_get_pixel(x, y) };
                let Luma([a_b_val]) = unsafe { a_b_mean.unsafe_get_pixel(x, y) };
                let Luma([b_val]) = unsafe { b_mean.unsafe_get_pixel(x, y) };

                let Rgb([guidance_r, guidance_g, guidance_b]) =
                    unsafe { self.guidance.unsafe_get_pixel(x, y) };

                let output_val = a_b_val.mul_add(
                    guidance_b,
                    a_r_val.mul_add(guidance_r, a_g_val * guidance_g),
                ) + b_val;

                unsafe { result.unsafe_put_pixel(x, y, Luma([output_val])) };
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma, Rgb};

    #[test]
    fn test_box_filter() {
        let mut img = ImageBuffer::new(5, 5);
        for y in 0..5 {
            for x in 0..5 {
                img.put_pixel(x, y, Luma([1.0f32]));
            }
        }

        let filtered = box_filter(&img, 1);
        assert_eq!(filtered.get_pixel(2, 2)[0], 1.0);
    }

    #[test]
    fn test_guided_filter_gray() {
        let guidance = ImageBuffer::from_fn(10, 10, |x, y| Luma([(x + y) as u8]));

        let input = ImageBuffer::from_fn(10, 10, |x, y| Luma([(x * y) as u8]));

        let filter = GuidedFilterGray::new(&guidance, 1, 0.1);
        let result = filter.filter(&input);

        assert_eq!(result.dimensions(), (10, 10));
    }

    #[test]
    fn test_guided_filter_color() {
        let guidance = ImageBuffer::from_fn(10, 10, |x, y| {
            Rgb([(x + y) as u8, (x + y) as u8, (x + y) as u8])
        });

        let input = ImageBuffer::from_fn(10, 10, |x, y| Luma([(x * y) as u8]));

        let filter = GuidedFilterColor::new(&guidance, 1, 0.1);
        let result = filter.filter(&input);

        assert_eq!(result.dimensions(), (10, 10));
    }

    #[test]
    fn test_guided_filter_extension() {
        let guidance = ImageBuffer::from_fn(10, 10, |x, y| Luma([(x + y) as u8]));

        let input = ImageBuffer::from_fn(10, 10, |x, y| Luma([(x * y) as u8]));

        let result = input.guided_filter(&guidance, 1, 0.1).unwrap();
        assert_eq!(result.dimensions(), (10, 10));
    }

    #[test]
    fn test_fast_guided_filter() {
        let guidance = ImageBuffer::from_fn(20, 20, |x, y| Luma([(x + y) as u8]));

        let input = ImageBuffer::from_fn(20, 20, |x, y| Luma([(x * y) as u8]));

        let result = input.fast_guided_filter(&guidance, 2, 0.1, 2).unwrap();
        assert_eq!(result.dimensions(), (20, 20));
    }

    #[test]
    fn test_guided_filter_color_guidance() {
        let guidance = ImageBuffer::from_fn(10, 10, |x, y| Rgb([(x + y) as u8, x as u8, y as u8]));

        let input = ImageBuffer::from_fn(10, 10, |x, y| Luma([(x * y) as u8]));

        let result = input
            .guided_filter_with_color_guidance(&guidance, 1, 0.1)
            .unwrap();
        assert_eq!(result.dimensions(), (10, 10));
    }

    #[test]
    fn test_fast_guided_filter_with_color_guidance() {
        let guidance = ImageBuffer::from_fn(20, 20, |x, y| Rgb([(x + y) as u8, x as u8, y as u8]));

        let input = ImageBuffer::from_fn(20, 20, |x, y| Luma([(x * y) as u8]));

        let result = input
            .fast_guided_filter_with_color_guidance(&guidance, 2, 0.1, 2)
            .unwrap();
        assert_eq!(result.dimensions(), (20, 20));
    }

    #[test]
    fn test_validate_params() {
        assert!(validate_guided_filter_params(0, 0.1).is_err());
        assert!(validate_guided_filter_params(1, 0.0).is_err());
        assert!(validate_guided_filter_params(1, -0.1).is_err());
        assert!(validate_guided_filter_params(1, 0.1).is_ok());
    }

    #[test]
    fn test_validate_scale() {
        assert!(validate_scale(0).is_err());
        assert!(validate_scale(1).is_err());
        assert!(validate_scale(2).is_ok());
    }

    #[test]
    fn test_dimension_mismatch() {
        let guidance: Image<Luma<u8>> = ImageBuffer::new(10, 10);
        let input: Image<Luma<u8>> = ImageBuffer::new(5, 5);

        let result = input.guided_filter(&guidance, 1, 0.1);
        assert!(result.is_err());

        if let Err(GuidedFilterError::DimensionMismatch {
            guidance_dims,
            input_dims,
        }) = result
        {
            assert_eq!(guidance_dims, (10, 10));
            assert_eq!(input_dims, (5, 5));
        } else {
            panic!("Expected DimensionMismatch error");
        }
    }

    #[test]
    fn test_convert_f32_to_u8() {
        let mut img = ImageBuffer::new(3, 3);
        img.put_pixel(0, 0, Luma([0.0]));
        img.put_pixel(1, 0, Luma([127.5]));
        img.put_pixel(2, 0, Luma([255.0]));
        img.put_pixel(0, 1, Luma([-10.0])); // Clamp to 0
        img.put_pixel(1, 1, Luma([300.0])); // Clamp to 255

        let result = convert_f32_to_u8_luma(&img);

        assert_eq!(result.get_pixel(0, 0)[0], 0);
        assert_eq!(result.get_pixel(1, 0)[0], 127);
        assert_eq!(result.get_pixel(2, 0)[0], 255);
        assert_eq!(result.get_pixel(0, 1)[0], 0);
        assert_eq!(result.get_pixel(1, 1)[0], 255);
    }

    #[test]
    fn test_resize_image() {
        let input = ImageBuffer::from_fn(4, 4, |x, y| Luma([((x + y) * 10) as u8]));

        let resized = resize_image(&input, 2, 2);
        assert_eq!(resized.dimensions(), (2, 2));

        // Check that pixel values are reasonable (should be from original image)
        for y in 0..2 {
            for x in 0..2 {
                let pixel = resized.get_pixel(x, y);
                assert!(pixel[0] <= 60); // Max value from 4x4 input should be 60
            }
        }
    }
}
