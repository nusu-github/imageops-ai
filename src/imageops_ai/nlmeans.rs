use crate::error::NLMeansError;
use crate::utils::clamp_f32_to_primitive;
use image::{ImageBuffer, Luma, Pixel, Primitive, Rgb, Rgba};
use imageproc::definitions::{Clamp, Image};
use std::f32;

/// Non-Local Means denoising trait
///
/// Non-Local Means is a denoising algorithm that replaces each pixel with a weighted average
/// of pixels that have similar neighborhoods. The weight is determined by the similarity
/// between patches (small windows) around the pixels.
///
/// This implementation supports:
/// - Grayscale images (`Image<Luma<T>>`)
/// - RGB color images (`Image<Rgb<T>>`)
/// - RGBA color images with alpha channel (`Image<Rgba<T>>`)
///
/// # Parameters
///
/// * `h` - Filtering parameter. Higher values remove more noise but may also remove fine details
/// * `small_window` - Size of the patch used for similarity comparison (must be odd)
/// * `big_window` - Size of the search window where similar patches are searched (must be odd and > small_window)
///
/// # Algorithm
///
/// For each pixel p:
/// 1. Extract a patch of size `small_window` around p (single-channel for grayscale, multi-channel for color)
/// 2. Search for similar patches within a `big_window` around p
/// 3. Calculate weights based on patch similarity: w = exp(-||patch_p - patch_q||² / (h² × patch_size² × channels))
/// 4. Replace pixel value with weighted average: new_value = Σ(w × pixel_q) / Σw
///
/// # Examples
///
/// ```rust
/// use imageops_ai::NLMeans;
/// use image::{ImageBuffer, Rgb};
///
/// // Create a sample RGB image
/// let mut rgb_image = ImageBuffer::new(10, 10);
/// for y in 0..10 {
///     for x in 0..10 {
///         rgb_image.put_pixel(x, y, Rgb([100u8, 150u8, 200u8]));
///     }
/// }
///
/// // Apply Non-Local Means denoising
/// let denoised = rgb_image.nl_means(10.0, 3, 7).unwrap();
/// assert_eq!(denoised.dimensions(), (10, 10));
/// ```
pub trait NLMeans<T> {
    /// Apply Non-Local Means denoising to the image
    ///
    /// # Arguments
    ///
    /// * `h` - Filtering parameter (must be positive)
    /// * `small_window` - Patch size for similarity comparison (must be odd positive integer)
    /// * `big_window` - Search window size (must be odd and larger than small_window)
    ///
    /// # Returns
    ///
    /// Returns a denoised image or an error if the parameters are invalid
    ///
    /// # Errors
    ///
    /// * `NLMeansError::InvalidWindowSize` - If window sizes are not odd positive integers
    /// * `NLMeansError::InvalidFilteringParameter` - If h is not positive
    /// * `NLMeansError::InvalidWindowSizes` - If big_window <= small_window
    /// * `NLMeansError::ImageTooSmall` - If image is too small for the specified window sizes
    fn nl_means(&self, h: f32, small_window: u32, big_window: u32) -> Result<Self, NLMeansError>
    where
        Self: Sized;
}

/// Validation function for Non-Local Means parameters
fn validate_parameters(
    h: f32,
    small_window: u32,
    big_window: u32,
    width: u32,
    height: u32,
) -> Result<(), NLMeansError> {
    use crate::utils::validate_non_empty_image;

    // Check if h is positive
    if h <= 0.0 {
        return Err(NLMeansError::InvalidFilteringParameter { h });
    }

    // Check if window sizes are odd and positive
    if small_window == 0 || small_window % 2 == 0 {
        return Err(NLMeansError::InvalidWindowSize { size: small_window });
    }

    if big_window == 0 || big_window % 2 == 0 {
        return Err(NLMeansError::InvalidWindowSize { size: big_window });
    }

    // Check if big_window > small_window
    if big_window <= small_window {
        return Err(NLMeansError::InvalidWindowSizes {
            small_window,
            big_window,
        });
    }

    // Use utils function to validate image is not empty
    validate_non_empty_image(width, height, "NL-Means").map_err(|_| {
        NLMeansError::ImageTooSmall {
            width,
            height,
            big_window,
        }
    })?;

    // Check if image is large enough for the window sizes
    if width <= big_window || height <= big_window {
        return Err(NLMeansError::ImageTooSmall {
            width,
            height,
            big_window,
        });
    }

    Ok(())
}

/// Calculate squared Euclidean distance between two patches
/// Works for both single-channel and multi-channel patches
#[inline]
fn patch_distance<T>(patch1: &[T], patch2: &[T]) -> f32
where
    T: Copy + Into<f32>,
{
    patch1
        .iter()
        .zip(patch2.iter())
        .map(|(&p1, &p2)| {
            let diff = p1.into() - p2.into();
            diff * diff
        })
        .sum()
}

/// Extract a patch from the padded image at the given coordinates
fn extract_patch<T>(
    padded_image: &[T],
    padded_width: u32,
    padded_height: u32,
    center_x: u32,
    center_y: u32,
    patch_size: u32,
) -> Vec<T>
where
    T: Primitive,
{
    let half_size = patch_size / 2;
    let mut patch = Vec::with_capacity((patch_size * patch_size) as usize);

    let half_size_i32 = half_size as i32;
    let center_x_i32 = center_x as i32;
    let center_y_i32 = center_y as i32;

    for dy in 0..patch_size {
        for dx in 0..patch_size {
            let x = center_x_i32 + dx as i32 - half_size_i32;
            let y = center_y_i32 + dy as i32 - half_size_i32;

            // Ensure coordinates are valid
            if x >= 0 && y >= 0 && (x as u32) < padded_width && (y as u32) < padded_height {
                let idx = y as usize * padded_width as usize + x as usize;
                if idx < padded_image.len() {
                    patch.push(padded_image[idx]);
                } else {
                    patch.push(padded_image[0]);
                }
            } else {
                // Out of bounds, use a fallback value
                patch.push(padded_image[0]);
            }
        }
    }

    patch
}

/// Extract a RGB patch from the padded image at the given coordinates
fn extract_patch_rgb<T>(
    padded_image: &[T],
    padded_width: u32,
    padded_height: u32,
    center_x: u32,
    center_y: u32,
    patch_size: u32,
) -> Vec<T>
where
    T: Primitive,
{
    let half_size = patch_size / 2;
    let mut patch = Vec::with_capacity((patch_size * patch_size * 3) as usize);

    let half_size_i32 = half_size as i32;
    let center_x_i32 = center_x as i32;
    let center_y_i32 = center_y as i32;

    for dy in 0..patch_size {
        for dx in 0..patch_size {
            let x = center_x_i32 + dx as i32 - half_size_i32;
            let y = center_y_i32 + dy as i32 - half_size_i32;

            // Ensure coordinates are valid
            if x >= 0 && y >= 0 && (x as u32) < padded_width && (y as u32) < padded_height {
                let base_idx = (y as usize * padded_width as usize + x as usize) * 3;
                if base_idx + 2 < padded_image.len() {
                    patch.push(padded_image[base_idx]); // R
                    patch.push(padded_image[base_idx + 1]); // G
                    patch.push(padded_image[base_idx + 2]); // B
                } else {
                    // Out of bounds, use fallback values
                    patch.push(padded_image[0]);
                    patch.push(padded_image[1]);
                    patch.push(padded_image[2]);
                }
            } else {
                // Out of bounds, use fallback values
                patch.push(padded_image[0]);
                patch.push(padded_image[1]);
                patch.push(padded_image[2]);
            }
        }
    }

    patch
}

/// Extract a RGBA patch from the padded image at the given coordinates
fn extract_patch_rgba<T>(
    padded_image: &[T],
    padded_width: u32,
    padded_height: u32,
    center_x: u32,
    center_y: u32,
    patch_size: u32,
) -> Vec<T>
where
    T: Primitive,
{
    let half_size = patch_size / 2;
    let mut patch = Vec::with_capacity((patch_size * patch_size * 4) as usize);

    let half_size_i32 = half_size as i32;
    let center_x_i32 = center_x as i32;
    let center_y_i32 = center_y as i32;

    for dy in 0..patch_size {
        for dx in 0..patch_size {
            let x = center_x_i32 + dx as i32 - half_size_i32;
            let y = center_y_i32 + dy as i32 - half_size_i32;

            // Ensure coordinates are valid
            if x >= 0 && y >= 0 && (x as u32) < padded_width && (y as u32) < padded_height {
                let base_idx = (y as usize * padded_width as usize + x as usize) * 4;
                if base_idx + 3 < padded_image.len() {
                    patch.push(padded_image[base_idx]); // R
                    patch.push(padded_image[base_idx + 1]); // G
                    patch.push(padded_image[base_idx + 2]); // B
                    patch.push(padded_image[base_idx + 3]); // A
                } else {
                    // Out of bounds, use fallback values
                    patch.push(padded_image[0]);
                    patch.push(padded_image[1]);
                    patch.push(padded_image[2]);
                    patch.push(padded_image[3]);
                }
            } else {
                // Out of bounds, use fallback values
                patch.push(padded_image[0]);
                patch.push(padded_image[1]);
                patch.push(padded_image[2]);
                patch.push(padded_image[3]);
            }
        }
    }

    patch
}

/// Apply reflection padding to the image
fn reflect_pad<T>(image: &Image<Luma<T>>, pad_size: u32) -> (Vec<T>, u32, u32)
where
    T: Primitive,
{
    let (width, height) = image.dimensions();
    let padded_width = width + 2 * pad_size;
    let padded_height = height + 2 * pad_size;
    let mut padded = vec![image.get_pixel(0, 0).0[0]; (padded_width * padded_height) as usize];

    // Copy original image to center
    for y in 0..height {
        for x in 0..width {
            let dst_idx = ((y + pad_size) * padded_width + (x + pad_size)) as usize;
            padded[dst_idx] = image.get_pixel(x, y).0[0];
        }
    }

    // Apply reflection padding
    // Top and bottom borders
    for y in 0..pad_size {
        for x in pad_size..(padded_width - pad_size) {
            // Top border - reflect from bottom
            let src_y = pad_size + (pad_size - 1 - y);
            let src_idx = (src_y * padded_width + x) as usize;
            let dst_top_idx = (y * padded_width + x) as usize;
            padded[dst_top_idx] = padded[src_idx];

            // Bottom border - reflect from top
            let dst_bottom_y = padded_height - 1 - y;
            let src_y = padded_height - pad_size - 1 - (pad_size - 1 - y);
            let src_idx = (src_y * padded_width + x) as usize;
            let dst_bottom_idx = (dst_bottom_y * padded_width + x) as usize;
            padded[dst_bottom_idx] = padded[src_idx];
        }
    }

    // Left and right borders
    for y in 0..padded_height {
        for x in 0..pad_size {
            // Left border - reflect from right
            let src_x = pad_size + (pad_size - 1 - x);
            let src_idx = (y * padded_width + src_x) as usize;
            let dst_left_idx = (y * padded_width + x) as usize;
            padded[dst_left_idx] = padded[src_idx];

            // Right border - reflect from left
            let dst_right_x = padded_width - 1 - x;
            let src_x = padded_width - pad_size - 1 - (pad_size - 1 - x);
            let src_idx = (y * padded_width + src_x) as usize;
            let dst_right_idx = (y * padded_width + dst_right_x) as usize;
            padded[dst_right_idx] = padded[src_idx];
        }
    }

    (padded, padded_width, padded_height)
}

/// Apply reflection padding to RGB image
fn reflect_pad_rgb<T>(image: &Image<Rgb<T>>, pad_size: u32) -> (Vec<T>, u32, u32)
where
    T: Primitive,
    Rgb<T>: Pixel<Subpixel = T>,
{
    let (width, height) = image.dimensions();
    let padded_width = width + 2 * pad_size;
    let padded_height = height + 2 * pad_size;
    let mut padded = vec![image.get_pixel(0, 0).0[0]; (padded_width * padded_height * 3) as usize];

    // Copy original image to center
    for y in 0..height {
        for x in 0..width {
            let src_pixel = image.get_pixel(x, y);
            let dst_base = ((y + pad_size) * padded_width + (x + pad_size)) as usize * 3;
            padded[dst_base] = src_pixel.0[0]; // R
            padded[dst_base + 1] = src_pixel.0[1]; // G
            padded[dst_base + 2] = src_pixel.0[2]; // B
        }
    }

    // Apply reflection padding
    // Top and bottom borders
    for y in 0..pad_size {
        for x in pad_size..(padded_width - pad_size) {
            // Top border - reflect from bottom
            let src_y = pad_size + (pad_size - 1 - y);
            let src_base = (src_y * padded_width + x) as usize * 3;
            let dst_top_base = (y * padded_width + x) as usize * 3;
            padded[dst_top_base] = padded[src_base];
            padded[dst_top_base + 1] = padded[src_base + 1];
            padded[dst_top_base + 2] = padded[src_base + 2];

            // Bottom border - reflect from top
            let dst_bottom_y = padded_height - 1 - y;
            let src_y = padded_height - pad_size - 1 - (pad_size - 1 - y);
            let src_base = (src_y * padded_width + x) as usize * 3;
            let dst_bottom_base = (dst_bottom_y * padded_width + x) as usize * 3;
            padded[dst_bottom_base] = padded[src_base];
            padded[dst_bottom_base + 1] = padded[src_base + 1];
            padded[dst_bottom_base + 2] = padded[src_base + 2];
        }
    }

    // Left and right borders
    for y in 0..padded_height {
        for x in 0..pad_size {
            // Left border - reflect from right
            let src_x = pad_size + (pad_size - 1 - x);
            let src_base = (y * padded_width + src_x) as usize * 3;
            let dst_left_base = (y * padded_width + x) as usize * 3;
            padded[dst_left_base] = padded[src_base];
            padded[dst_left_base + 1] = padded[src_base + 1];
            padded[dst_left_base + 2] = padded[src_base + 2];

            // Right border - reflect from left
            let dst_right_x = padded_width - 1 - x;
            let src_x = padded_width - pad_size - 1 - (pad_size - 1 - x);
            let src_base = (y * padded_width + src_x) as usize * 3;
            let dst_right_base = (y * padded_width + dst_right_x) as usize * 3;
            padded[dst_right_base] = padded[src_base];
            padded[dst_right_base + 1] = padded[src_base + 1];
            padded[dst_right_base + 2] = padded[src_base + 2];
        }
    }

    (padded, padded_width, padded_height)
}

/// Apply reflection padding to RGBA image
fn reflect_pad_rgba<T>(image: &Image<Rgba<T>>, pad_size: u32) -> (Vec<T>, u32, u32)
where
    T: Primitive,
    Rgba<T>: Pixel<Subpixel = T>,
{
    let (width, height) = image.dimensions();
    let padded_width = width + 2 * pad_size;
    let padded_height = height + 2 * pad_size;
    let mut padded = vec![image.get_pixel(0, 0).0[0]; (padded_width * padded_height * 4) as usize];

    // Copy original image to center
    for y in 0..height {
        for x in 0..width {
            let src_pixel = image.get_pixel(x, y);
            let dst_base = ((y + pad_size) * padded_width + (x + pad_size)) as usize * 4;
            padded[dst_base] = src_pixel.0[0]; // R
            padded[dst_base + 1] = src_pixel.0[1]; // G
            padded[dst_base + 2] = src_pixel.0[2]; // B
            padded[dst_base + 3] = src_pixel.0[3]; // A
        }
    }

    // Apply reflection padding
    // Top and bottom borders
    for y in 0..pad_size {
        for x in pad_size..(padded_width - pad_size) {
            // Top border - reflect from bottom
            let src_y = pad_size + (pad_size - 1 - y);
            let src_base = (src_y * padded_width + x) as usize * 4;
            let dst_top_base = (y * padded_width + x) as usize * 4;
            padded[dst_top_base] = padded[src_base];
            padded[dst_top_base + 1] = padded[src_base + 1];
            padded[dst_top_base + 2] = padded[src_base + 2];
            padded[dst_top_base + 3] = padded[src_base + 3];

            // Bottom border - reflect from top
            let dst_bottom_y = padded_height - 1 - y;
            let src_y = padded_height - pad_size - 1 - (pad_size - 1 - y);
            let src_base = (src_y * padded_width + x) as usize * 4;
            let dst_bottom_base = (dst_bottom_y * padded_width + x) as usize * 4;
            padded[dst_bottom_base] = padded[src_base];
            padded[dst_bottom_base + 1] = padded[src_base + 1];
            padded[dst_bottom_base + 2] = padded[src_base + 2];
            padded[dst_bottom_base + 3] = padded[src_base + 3];
        }
    }

    // Left and right borders
    for y in 0..padded_height {
        for x in 0..pad_size {
            // Left border - reflect from right
            let src_x = pad_size + (pad_size - 1 - x);
            let src_base = (y * padded_width + src_x) as usize * 4;
            let dst_left_base = (y * padded_width + x) as usize * 4;
            padded[dst_left_base] = padded[src_base];
            padded[dst_left_base + 1] = padded[src_base + 1];
            padded[dst_left_base + 2] = padded[src_base + 2];
            padded[dst_left_base + 3] = padded[src_base + 3];

            // Right border - reflect from left
            let dst_right_x = padded_width - 1 - x;
            let src_x = padded_width - pad_size - 1 - (pad_size - 1 - x);
            let src_base = (y * padded_width + src_x) as usize * 4;
            let dst_right_base = (y * padded_width + dst_right_x) as usize * 4;
            padded[dst_right_base] = padded[src_base];
            padded[dst_right_base + 1] = padded[src_base + 1];
            padded[dst_right_base + 2] = padded[src_base + 2];
            padded[dst_right_base + 3] = padded[src_base + 3];
        }
    }

    (padded, padded_width, padded_height)
}

impl<T> NLMeans<T> for Image<Luma<T>>
where
    T: Primitive + Into<f32> + Clamp<f32>,
    u8: Into<T>,
{
    fn nl_means(&self, h: f32, small_window: u32, big_window: u32) -> Result<Self, NLMeansError> {
        let (width, height) = self.dimensions();

        // Validate parameters
        validate_parameters(h, small_window, big_window, width, height)?;

        // Apply reflection padding
        let pad_size = big_window / 2;
        let (padded_image, padded_width, padded_height) = reflect_pad(self, pad_size);

        // Pre-compute normalization factor
        let nw = h * h * (small_window * small_window) as f32;

        // Initialize result image
        let mut result = ImageBuffer::new(width, height);

        // Process each pixel
        for y in 0..height {
            for x in 0..width {
                let padded_x = x + pad_size;
                let padded_y = y + pad_size;

                // Extract patch for current pixel
                let pixel_patch = extract_patch(
                    &padded_image,
                    padded_width,
                    padded_height,
                    padded_x,
                    padded_y,
                    small_window,
                );

                let mut weighted_sum = 0.0f32;
                let mut weight_sum = 0.0f32;

                // Search within big_window
                let half_big = big_window / 2;
                for ny in (padded_y - half_big)..=(padded_y + half_big) {
                    for nx in (padded_x - half_big)..=(padded_x + half_big) {
                        // Extract patch for neighbor pixel
                        let neighbor_patch = extract_patch(
                            &padded_image,
                            padded_width,
                            padded_height,
                            nx,
                            ny,
                            small_window,
                        );

                        // Calculate patch distance
                        let distance = patch_distance(&pixel_patch, &neighbor_patch);

                        // Calculate weight
                        let weight = f32::exp(-distance / nw);

                        // Get neighbor pixel value
                        let neighbor_value = padded_image[(ny * padded_width + nx) as usize].into();

                        weighted_sum += weight * neighbor_value;
                        weight_sum += weight;
                    }
                }

                // Calculate new pixel value
                let new_value = if weight_sum > 0.0 {
                    weighted_sum / weight_sum
                } else {
                    self.get_pixel(x, y).0[0].into()
                };

                // Clamp to valid range and convert back to T
                let clamped_value = clamp_f32_to_primitive::<T>(new_value);
                result.put_pixel(x, y, Luma([clamped_value]));
            }
        }

        Ok(result)
    }
}

impl<T> NLMeans<T> for Image<Rgb<T>>
where
    T: Primitive + Into<f32> + Clamp<f32>,
    u8: Into<T>,
    Rgb<T>: Pixel<Subpixel = T>,
{
    fn nl_means(&self, h: f32, small_window: u32, big_window: u32) -> Result<Self, NLMeansError> {
        let (width, height) = self.dimensions();

        // Validate parameters
        validate_parameters(h, small_window, big_window, width, height)?;

        // Apply reflection padding
        let pad_size = big_window / 2;
        let (padded_image, padded_width, padded_height) = reflect_pad_rgb(self, pad_size);

        // Pre-compute normalization factor
        let nw = h * h * (small_window * small_window * 3) as f32; // 3 channels

        // Initialize result image
        let mut result = ImageBuffer::new(width, height);

        // Process each pixel
        for y in 0..height {
            for x in 0..width {
                let padded_x = x + pad_size;
                let padded_y = y + pad_size;

                // Extract patch for current pixel
                let pixel_patch = extract_patch_rgb(
                    &padded_image,
                    padded_width,
                    padded_height,
                    padded_x,
                    padded_y,
                    small_window,
                );

                let mut weighted_sum = [0.0f32; 3];
                let mut weight_sum = 0.0f32;

                // Search within big_window
                let half_big = big_window / 2;
                for ny in (padded_y - half_big)..=(padded_y + half_big) {
                    for nx in (padded_x - half_big)..=(padded_x + half_big) {
                        // Extract patch for neighbor pixel
                        let neighbor_patch = extract_patch_rgb(
                            &padded_image,
                            padded_width,
                            padded_height,
                            nx,
                            ny,
                            small_window,
                        );

                        // Calculate patch distance
                        let distance = patch_distance(&pixel_patch, &neighbor_patch);

                        // Calculate weight
                        let weight = f32::exp(-distance / nw);

                        // Get neighbor pixel values (RGB)
                        let neighbor_base = (ny * padded_width + nx) as usize * 3;
                        let neighbor_r = padded_image[neighbor_base].into();
                        let neighbor_g = padded_image[neighbor_base + 1].into();
                        let neighbor_b = padded_image[neighbor_base + 2].into();

                        weighted_sum[0] += weight * neighbor_r;
                        weighted_sum[1] += weight * neighbor_g;
                        weighted_sum[2] += weight * neighbor_b;
                        weight_sum += weight;
                    }
                }

                // Calculate new pixel values
                let new_values = if weight_sum > 0.0 {
                    [
                        weighted_sum[0] / weight_sum,
                        weighted_sum[1] / weight_sum,
                        weighted_sum[2] / weight_sum,
                    ]
                } else {
                    let orig_pixel = self.get_pixel(x, y);
                    [
                        orig_pixel.0[0].into(),
                        orig_pixel.0[1].into(),
                        orig_pixel.0[2].into(),
                    ]
                };

                // Clamp to valid range and convert back to T
                let clamped_r = clamp_f32_to_primitive::<T>(new_values[0]);
                let clamped_g = clamp_f32_to_primitive::<T>(new_values[1]);
                let clamped_b = clamp_f32_to_primitive::<T>(new_values[2]);
                result.put_pixel(x, y, Rgb([clamped_r, clamped_g, clamped_b]));
            }
        }

        Ok(result)
    }
}

impl<T> NLMeans<T> for Image<Rgba<T>>
where
    T: Primitive + Into<f32> + Clamp<f32>,
    u8: Into<T>,
    Rgba<T>: Pixel<Subpixel = T>,
{
    fn nl_means(&self, h: f32, small_window: u32, big_window: u32) -> Result<Self, NLMeansError> {
        let (width, height) = self.dimensions();

        // Validate parameters
        validate_parameters(h, small_window, big_window, width, height)?;

        // Apply reflection padding
        let pad_size = big_window / 2;
        let (padded_image, padded_width, padded_height) = reflect_pad_rgba(self, pad_size);

        // Pre-compute normalization factor
        let nw = h * h * (small_window * small_window * 4) as f32; // 4 channels

        // Initialize result image
        let mut result = ImageBuffer::new(width, height);

        // Process each pixel
        for y in 0..height {
            for x in 0..width {
                let padded_x = x + pad_size;
                let padded_y = y + pad_size;

                // Extract patch for current pixel
                let pixel_patch = extract_patch_rgba(
                    &padded_image,
                    padded_width,
                    padded_height,
                    padded_x,
                    padded_y,
                    small_window,
                );

                let mut weighted_sum = [0.0f32; 4];
                let mut weight_sum = 0.0f32;

                // Search within big_window
                let half_big = big_window / 2;
                for ny in (padded_y - half_big)..=(padded_y + half_big) {
                    for nx in (padded_x - half_big)..=(padded_x + half_big) {
                        // Extract patch for neighbor pixel
                        let neighbor_patch = extract_patch_rgba(
                            &padded_image,
                            padded_width,
                            padded_height,
                            nx,
                            ny,
                            small_window,
                        );

                        // Calculate patch distance
                        let distance = patch_distance(&pixel_patch, &neighbor_patch);

                        // Calculate weight
                        let weight = f32::exp(-distance / nw);

                        // Get neighbor pixel values (RGBA)
                        let neighbor_base = (ny * padded_width + nx) as usize * 4;
                        let neighbor_r = padded_image[neighbor_base].into();
                        let neighbor_g = padded_image[neighbor_base + 1].into();
                        let neighbor_b = padded_image[neighbor_base + 2].into();
                        let neighbor_a = padded_image[neighbor_base + 3].into();

                        weighted_sum[0] += weight * neighbor_r;
                        weighted_sum[1] += weight * neighbor_g;
                        weighted_sum[2] += weight * neighbor_b;
                        weighted_sum[3] += weight * neighbor_a;
                        weight_sum += weight;
                    }
                }

                // Calculate new pixel values
                let new_values = if weight_sum > 0.0 {
                    [
                        weighted_sum[0] / weight_sum,
                        weighted_sum[1] / weight_sum,
                        weighted_sum[2] / weight_sum,
                        weighted_sum[3] / weight_sum,
                    ]
                } else {
                    let orig_pixel = self.get_pixel(x, y);
                    [
                        orig_pixel.0[0].into(),
                        orig_pixel.0[1].into(),
                        orig_pixel.0[2].into(),
                        orig_pixel.0[3].into(),
                    ]
                };

                // Clamp to valid range and convert back to T
                let clamped_r = clamp_f32_to_primitive::<T>(new_values[0]);
                let clamped_g = clamp_f32_to_primitive::<T>(new_values[1]);
                let clamped_b = clamp_f32_to_primitive::<T>(new_values[2]);
                let clamped_a = clamp_f32_to_primitive::<T>(new_values[3]);
                result.put_pixel(x, y, Rgba([clamped_r, clamped_g, clamped_b, clamped_a]));
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma, Rgb, Rgba};

    #[test]
    fn test_validate_parameters() {
        // Valid parameters
        assert!(validate_parameters(10.0, 3, 7, 50, 50).is_ok());

        // Invalid h
        assert!(matches!(
            validate_parameters(0.0, 3, 7, 50, 50),
            Err(NLMeansError::InvalidFilteringParameter { h: 0.0 })
        ));

        // Invalid small_window (even)
        assert!(matches!(
            validate_parameters(10.0, 4, 7, 50, 50),
            Err(NLMeansError::InvalidWindowSize { size: 4 })
        ));

        // Invalid big_window (even)
        assert!(matches!(
            validate_parameters(10.0, 3, 8, 50, 50),
            Err(NLMeansError::InvalidWindowSize { size: 8 })
        ));

        // big_window <= small_window
        assert!(matches!(
            validate_parameters(10.0, 7, 7, 50, 50),
            Err(NLMeansError::InvalidWindowSizes {
                small_window: 7,
                big_window: 7
            })
        ));

        // Image too small
        assert!(matches!(
            validate_parameters(10.0, 3, 7, 5, 5),
            Err(NLMeansError::ImageTooSmall {
                width: 5,
                height: 5,
                big_window: 7
            })
        ));
    }

    #[test]
    fn test_patch_distance() {
        let patch1 = vec![1.0f32, 2.0, 3.0, 4.0];
        let patch2 = vec![1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(patch_distance(&patch1, &patch2), 0.0);

        let patch3 = vec![2.0f32, 3.0, 4.0, 5.0];
        assert_eq!(patch_distance(&patch1, &patch3), 4.0); // (1-2)² + (2-3)² + (3-4)² + (4-5)² = 4
    }

    #[test]
    fn test_nl_means_basic() {
        // Create a simple 10x10 test image (large enough for big_window=7)
        let mut image = ImageBuffer::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                // Create a simple pattern
                let value = ((x + y) * 20) as u8;
                image.put_pixel(x, y, Luma([value]));
            }
        }

        // Apply NL-Means with small parameters
        let result = image.nl_means(10.0, 3, 7);
        match &result {
            Ok(_) => {}
            Err(e) => panic!("NL-Means failed: {:?}", e),
        }
        assert!(result.is_ok());

        let denoised = result.unwrap();
        assert_eq!(denoised.dimensions(), (10, 10));
    }

    #[test]
    fn test_extract_patch() {
        let padded = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let patch = extract_patch(&padded, 3, 3, 1, 1, 3);
        assert_eq!(patch, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_patch_distance_multi() {
        let patch1 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let patch2 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(patch_distance(&patch1, &patch2), 0.0);

        let patch3 = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert_eq!(patch_distance(&patch1, &patch3), 6.0); // 6 × (diff of 1)² = 6
    }

    #[test]
    fn test_nl_means_rgb() {
        // Create a simple 10x10 RGB test image
        let mut image = ImageBuffer::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                // Create a simple RGB pattern
                let r = ((x + y) * 10) as u8;
                let g = ((x * 2) * 10) as u8;
                let b = ((y * 2) * 10) as u8;
                image.put_pixel(x, y, Rgb([r, g, b]));
            }
        }

        // Apply NL-Means with small parameters
        let result = image.nl_means(10.0, 3, 7);
        assert!(result.is_ok());

        let denoised = result.unwrap();
        assert_eq!(denoised.dimensions(), (10, 10));
    }

    #[test]
    fn test_nl_means_rgba() {
        // Create a simple 10x10 RGBA test image
        let mut image = ImageBuffer::new(10, 10);
        for y in 0..10 {
            for x in 0..10 {
                // Create a simple RGBA pattern
                let r = ((x + y) * 10) as u8;
                let g = ((x * 2) * 10) as u8;
                let b = ((y * 2) * 10) as u8;
                let a = 255u8; // Full opacity
                image.put_pixel(x, y, Rgba([r, g, b, a]));
            }
        }

        // Apply NL-Means with small parameters
        let result = image.nl_means(10.0, 3, 7);
        assert!(result.is_ok());

        let denoised = result.unwrap();
        assert_eq!(denoised.dimensions(), (10, 10));
    }

    #[test]
    fn test_extract_patch_rgb() {
        let padded = vec![
            1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27,
        ];
        let patch = extract_patch_rgb(&padded, 3, 3, 1, 1, 3);
        // Expected: 3x3 patch × 3 channels = 27 elements
        assert_eq!(patch.len(), 27);
    }

    #[test]
    fn test_extract_patch_rgba() {
        let padded = vec![1u8; 36]; // 3x3 × 4 channels
        let patch = extract_patch_rgba(&padded, 3, 3, 1, 1, 3);
        // Expected: 3x3 patch × 4 channels = 36 elements
        assert_eq!(patch.len(), 36);
    }
}
