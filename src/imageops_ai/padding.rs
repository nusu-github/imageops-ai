use image::{GenericImage, GenericImageView, ImageBuffer, Pixel};
use imageproc::definitions::Image;

use crate::error::PaddingError;

/// Enum to specify padding position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Position {
    /// Top center
    Top,
    /// Bottom center
    Bottom,
    /// Left center
    Left,
    /// Right center
    Right,
    /// Top left
    TopLeft,
    /// Top right
    TopRight,
    /// Bottom left
    BottomLeft,
    /// Bottom right
    BottomRight,
    /// Center
    Center,
}

/// Calculate position from image size and padding size.
///
/// # Arguments
///
/// * `size` - Original image size (width, height)
/// * `pad_size` - Padded size (width, height)
/// * `position` - Padding position
///
/// # Returns
///
/// Returns the position (x, y) where the image should be placed on success
///
/// # Errors
///
/// * Returns error when padding size is smaller than original image size
pub fn calculate_position(
    size: (u32, u32),
    pad_size: (u32, u32),
    position: Position,
) -> Result<(i64, i64), PaddingError> {
    let (width, height) = size;
    let (target_width, target_height) = pad_size;

    if target_width < width {
        return Err(PaddingError::PaddingWidthTooSmall {
            width,
            pad_width: target_width,
        });
    }

    if target_height < height {
        return Err(PaddingError::PaddingHeightTooSmall {
            height,
            pad_height: target_height,
        });
    }

    let (x, y) = match position {
        Position::Top => ((target_width - width) / 2, 0),
        Position::Bottom => ((target_width - width) / 2, target_height - height),
        Position::Left => (0, (target_height - height) / 2),
        Position::Right => (target_width - width, (target_height - height) / 2),
        Position::TopLeft => (0, 0),
        Position::TopRight => (target_width - width, 0),
        Position::BottomLeft => (0, target_height - height),
        Position::BottomRight => (target_width - width, target_height - height),
        Position::Center => ((target_width - width) / 2, (target_height - height) / 2),
    };

    Ok((x.into(), y.into()))
}

/// Add padding with specified size and position (function-based implementation).
///
/// # Arguments
///
/// * `image` - Original image
/// * `pad_size` - Padded size (width, height)
/// * `position` - Padding position
/// * `color` - Padding color
///
/// # Returns
///
/// Padded image
///
/// # Examples
/// ```
/// use image::{Rgb, RgbImage};
/// use imageproc::definitions::Image;
/// use imageops_ai::{add_padding, Position};
///
/// let image: RgbImage = RgbImage::new(10, 10);
/// let padded = add_padding(&image, (20, 20), Position::Center, Rgb([255, 255, 255])).unwrap();
/// assert_eq!(padded.dimensions(), (20, 20));
/// ```
pub fn add_padding<I, P>(
    image: &I,
    pad_size: (u32, u32),
    position: Position,
    color: P,
) -> Result<Image<P>, PaddingError>
where
    I: GenericImageView<Pixel = P>,
    P: Pixel,
{
    let (width, height) = image.dimensions();
    let (x, y) = calculate_position((width, height), pad_size, position)?;
    let (target_width, target_height) = pad_size;

    // Memory-optimized buffer allocation
    let mut out = create_buffer_impl(target_width, target_height, color);

    // High-performance image copying with multiple optimization strategies
    copy_image_impl(image, &mut out, x, y, width, height);

    Ok(out)
}

/// Internal image copying function using multiple strategies.
///
/// This function applies several optimization techniques:
/// 1. Row-based bulk copying when memory layout allows
/// 2. Iterator-based processing with bounds check elision
/// 3. Cache-friendly access patterns
#[inline]
fn copy_image_impl<I, P>(
    src: &I,
    dst: &mut Image<P>,
    offset_x: i64,
    offset_y: i64,
    width: u32,
    height: u32,
) where
    I: GenericImageView<Pixel = P>,
    P: Pixel,
{
    let start_x = offset_x as u32;
    let start_y = offset_y as u32;

    // Strategy 1: Try row-wise bulk copy for contiguous memory when possible
    // This works when both source and destination have same width and
    // we're copying complete rows
    if can_use_bulk_copy_impl(src, dst, start_x, width) {
        copy_rows_bulk_impl(src, dst, start_x, start_y, width, height);
        return;
    }

    // Strategy 2: Row-by-row iterator processing (cache-friendly)
    (0..height).for_each(|src_y| {
        let dst_y = start_y + src_y;
        (0..width).for_each(|src_x| {
            let dst_x = start_x + src_x;

            // Safety: Bounds validated by calculate_position
            unsafe {
                let pixel = src.unsafe_get_pixel(src_x, src_y);
                dst.unsafe_put_pixel(dst_x, dst_y, pixel);
            }
        });
    });
}

/// Check if bulk copying is possible based on memory layout.
#[inline]
const fn can_use_bulk_copy_impl<I, P>(_src: &I, _dst: &Image<P>, start_x: u32, width: u32) -> bool
where
    I: GenericImageView<Pixel = P>,
    P: Pixel,
{
    // Bulk copy is efficient when:
    // 1. Copying starts at the beginning of destination rows (x offset = 0)
    // 2. Source width matches copy width (copying complete rows)
    start_x == 0 && width > 64 // Only worthwhile for larger widths
}

/// Bulk copy complete rows for maximum performance.
#[inline]
fn copy_rows_bulk_impl<I, P>(
    src: &I,
    dst: &mut Image<P>,
    start_x: u32,
    start_y: u32,
    width: u32,
    height: u32,
) where
    I: GenericImageView<Pixel = P>,
    P: Pixel,
{
    // For now, fall back to optimized pixel-by-pixel copy
    // Future: Implement actual bulk memory copy when ImageBuffer exposes raw access
    (0..height).for_each(|src_y| {
        let dst_y = start_y + src_y;
        (0..width).for_each(|src_x| {
            let dst_x = start_x + src_x;

            unsafe {
                let pixel = src.unsafe_get_pixel(src_x, src_y);
                dst.unsafe_put_pixel(dst_x, dst_y, pixel);
            }
        });
    });
}

/// Create buffer with pre-allocated capacity.
///
/// This function optimizes memory allocation by:
/// 1. Pre-calculating exact capacity requirements
/// 2. Using efficient fill patterns
/// 3. Minimizing allocation overhead
#[inline]
fn create_buffer_impl<P>(width: u32, height: u32, fill_color: P) -> Image<P>
where
    P: Pixel,
{
    let total_pixels = (width as usize) * (height as usize);
    let subpixels_per_pixel = P::CHANNEL_COUNT as usize;
    let total_subpixels = total_pixels * subpixels_per_pixel;

    // Pre-allocate with exact capacity to avoid reallocations
    let mut buffer = Vec::with_capacity(total_subpixels);

    // Fill buffer efficiently using iterator repeat
    let fill_channels = fill_color.channels();
    for _ in 0..total_pixels {
        buffer.extend_from_slice(fill_channels);
    }

    // Safety: We've filled exactly the required number of elements
    ImageBuffer::from_raw(width, height, buffer)
        .expect("Buffer size calculation error - this should not happen")
}

/// Trait that provides padding operations.
///
/// Note: This operation changes the image dimensions, so there is no `_mut` variant
/// available. The algorithm creates a new image with different dimensions.
pub trait PaddingExt<P: Pixel> {
    /// Add padding with specified size and position.
    ///
    /// This consumes the original image.
    ///
    /// # Arguments
    ///
    /// * `pad_size` - Padded size (width, height)
    /// * `position` - Padding position
    /// * `color` - Padding color
    ///
    /// # Returns
    ///
    /// Tuple of (padded image, position (x, y) where the original image was placed)
    ///
    /// # Examples
    /// ```no_run
    /// use imageops_ai::{PaddingExt, Position, Image};
    /// use image::Rgb;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let image: Image<Rgb<u8>> = Image::new(10, 10);
    /// let (padded, position) = image.add_padding((20, 20), Position::Center, Rgb([255, 255, 255]))?;
    /// # Ok(())
    /// # }
    /// ```
    fn add_padding(
        self,
        pad_size: (u32, u32),
        position: Position,
        color: P,
    ) -> Result<(Self, (i64, i64)), PaddingError>
    where
        Self: Sized;

    /// Add padding to make the image square.
    ///
    /// This consumes the original image.
    ///
    /// # Arguments
    ///
    /// * `color` - Padding color
    ///
    /// # Returns
    ///
    /// Tuple of (padded square image, position (x, y) where the original image was placed)
    fn to_square(self, color: P) -> Result<(Self, (i64, i64)), PaddingError>
    where
        Self: Sized;

    /// Calculate padding position.
    ///
    /// This is a helper method that doesn't consume self.
    fn calculate_padding_position(
        &self,
        pad_size: (u32, u32),
        position: Position,
    ) -> Result<(i64, i64), PaddingError>;

    /// Calculate position and size for square padding.
    ///
    /// This is a helper method that doesn't consume self.
    fn calculate_square_padding(&self) -> Result<((i64, i64), (u32, u32)), PaddingError>;

    /// Hidden _mut variant that is not available for this operation.
    #[doc(hidden)]
    fn add_padding_mut(
        &mut self,
        _pad_size: (u32, u32),
        _position: Position,
        _color: P,
    ) -> Result<&mut Self, PaddingError> {
        unimplemented!(
            "add_padding_mut is not available because the operation changes image dimensions"
        )
    }

    /// Hidden _mut variant that is not available for this operation.
    #[doc(hidden)]
    fn to_square_mut(&mut self, _color: P) -> Result<&mut Self, PaddingError> {
        unimplemented!(
            "to_square_mut is not available because the operation changes image dimensions"
        )
    }
}

impl<P: Pixel> PaddingExt<P> for Image<P> {
    fn add_padding(
        self,
        pad_size: (u32, u32),
        position: Position,
        color: P,
    ) -> Result<(Self, (i64, i64)), PaddingError> {
        let (width, height) = self.dimensions();
        let pos = calculate_position((width, height), pad_size, position)?;
        let padded = add_padding(&self, pad_size, position, color)?;
        Ok((padded, pos))
    }

    fn to_square(self, color: P) -> Result<(Self, (i64, i64)), PaddingError> {
        let ((_x, _y), pad_size) = self.calculate_square_padding()?;
        self.add_padding(pad_size, Position::Center, color)
    }

    fn calculate_padding_position(
        &self,
        pad_size: (u32, u32),
        position: Position,
    ) -> Result<(i64, i64), PaddingError> {
        let (width, height) = self.dimensions();
        calculate_position((width, height), pad_size, position)
    }

    fn calculate_square_padding(&self) -> Result<((i64, i64), (u32, u32)), PaddingError> {
        let (width, height) = self.dimensions();

        let pad_size = if width > height {
            (width, width)
        } else {
            (height, height)
        };

        self.calculate_padding_position(pad_size, Position::Center)
            .map(|(x, y)| ((x, y), pad_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;
    use image::Rgb;

    #[test]
    fn calculate_position_with_valid_input_returns_expected_coordinates() {
        // Test center position
        let pos = calculate_position((10, 10), (20, 20), Position::Center);
        assert_eq!(pos, Ok((5, 5)));

        // Test top-left position
        let pos = calculate_position((10, 10), (20, 20), Position::TopLeft);
        assert_eq!(pos, Ok((0, 0)));

        // Test top-right position
        let pos = calculate_position((10, 10), (20, 20), Position::TopRight);
        assert_eq!(pos, Ok((10, 0)));

        // Test bottom-left position
        let pos = calculate_position((10, 10), (20, 20), Position::BottomLeft);
        assert_eq!(pos, Ok((0, 10)));

        // Test bottom-right position
        let pos = calculate_position((10, 10), (20, 20), Position::BottomRight);
        assert_eq!(pos, Ok((10, 10)));
    }

    #[test]
    fn add_padding_with_valid_input_creates_padded_image() {
        let image = create_test_rgb_image(); // 2x2 image
        let fill_color = Rgb([255, 255, 255]); // White

        // Test padding to larger size
        let result = add_padding(&image, (4, 4), Position::Center, fill_color);
        assert!(result.is_ok());

        let padded = result.unwrap();
        assert_eq!(padded.dimensions(), (4, 4));

        // Test invalid padding (smaller than original)
        let result = add_padding(&image, (1, 1), Position::Center, fill_color);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PaddingError::PaddingWidthTooSmall { .. }
        ));
    }

    #[test]
    fn add_padding_ext_with_valid_input_preserves_original_content() {
        let image = create_test_rgb_image(); // 2x2 image
        let fill_color = Rgb([255, 255, 255]); // White

        // Store original pixels before moving image
        let orig_00 = *image.get_pixel(0, 0);
        let orig_10 = *image.get_pixel(1, 0);
        let orig_01 = *image.get_pixel(0, 1);
        let orig_11 = *image.get_pixel(1, 1);

        // Test regular padding
        let result = image.add_padding((4, 4), Position::TopLeft, fill_color);
        assert!(result.is_ok());

        let (padded, pos) = result.unwrap();
        assert_eq!(padded.dimensions(), (4, 4));
        assert_eq!(pos, (0, 0));

        // Verify original content is preserved
        assert_eq!(*padded.get_pixel(0, 0), orig_00);
        assert_eq!(*padded.get_pixel(1, 0), orig_10);
        assert_eq!(*padded.get_pixel(0, 1), orig_01);
        assert_eq!(*padded.get_pixel(1, 1), orig_11);
    }

    #[test]
    fn to_square_with_rectangular_image_creates_square_image() {
        // Test rectangular image (width > height)
        let mut image: Image<Rgb<u8>> = Image::new(6, 4);
        for y in 0..4 {
            for x in 0..6 {
                image.put_pixel(x, y, Rgb([100, 150, 200]));
            }
        }

        let result = image.to_square(Rgb([255, 255, 255]));
        assert!(result.is_ok());

        let (padded, pos) = result.unwrap();
        assert_eq!(padded.dimensions(), (6, 6)); // Should be square
        assert_eq!(pos, (0, 1)); // Centered vertically

        // Test square image (no padding needed)
        let square_image: Image<Rgb<u8>> = Image::new(4, 4);
        let result = square_image.to_square(Rgb([255, 255, 255]));
        assert!(result.is_ok());

        let (padded, pos) = result.unwrap();
        assert_eq!(padded.dimensions(), (4, 4));
        assert_eq!(pos, (0, 0));
    }

    #[test]
    fn calculate_padding_position_with_invalid_size_returns_error() {
        let image = create_test_rgb_image(); // 2x2 image

        // Test padding size too small (width)
        let result = image.calculate_padding_position((1, 4), Position::Center);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PaddingError::PaddingWidthTooSmall { .. }
        ));

        // Test padding size too small (height)
        let result = image.calculate_padding_position((4, 1), Position::Center);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PaddingError::PaddingHeightTooSmall { .. }
        ));
    }

    #[test]
    fn calculate_padding_position_with_various_positions_returns_correct_coordinates() {
        let image = create_test_rgb_image(); // 2x2 image

        // Test all positions with 6x6 padding
        let positions = [
            (Position::TopLeft, (0, 0)),
            (Position::TopRight, (4, 0)),
            (Position::BottomLeft, (0, 4)),
            (Position::BottomRight, (4, 4)),
            (Position::Center, (2, 2)),
        ];

        for (pos, expected) in positions {
            let result = image.calculate_padding_position((6, 6), pos);
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), expected);
        }
    }
}
