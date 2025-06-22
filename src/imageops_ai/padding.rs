use crate::error::PaddingError;
use image::{GenericImage, ImageBuffer, Pixel};
use imageproc::definitions::Image;

/// Enum to specify padding position
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

/// Calculate position from image size and padding size
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
    let (pad_width, pad_height) = pad_size;

    if pad_width < width {
        return Err(PaddingError::PaddingWidthTooSmall { width, pad_width });
    }

    if pad_height < height {
        return Err(PaddingError::PaddingHeightTooSmall { height, pad_height });
    }

    let (x, y) = match position {
        Position::Top => ((pad_width - width) / 2, 0),
        Position::Bottom => ((pad_width - width) / 2, pad_height - height),
        Position::Left => (0, (pad_height - height) / 2),
        Position::Right => (pad_width - width, (pad_height - height) / 2),
        Position::TopLeft => (0, 0),
        Position::TopRight => (pad_width - width, 0),
        Position::BottomLeft => (0, pad_height - height),
        Position::BottomRight => (pad_width - width, pad_height - height),
        Position::Center => ((pad_width - width) / 2, (pad_height - height) / 2),
    };

    Ok((x.into(), y.into()))
}

/// Add padding with specified size and position (function-based implementation)
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
    I: GenericImage<Pixel = P>,
    P: Pixel,
{
    let (width, height) = image.dimensions();
    let (x, y) = calculate_position((width, height), pad_size, position)?;
    let (pad_width, pad_height) = pad_size;

    // Create padded image using efficient pixel operations
    let mut out: ImageBuffer<P, Vec<P::Subpixel>> =
        ImageBuffer::from_pixel(pad_width, pad_height, color);

    // Copy original image to new position
    for src_y in 0..height {
        for src_x in 0..width {
            let dst_x = (x + src_x as i64) as u32;
            let dst_y = (y + src_y as i64) as u32;

            if dst_x < pad_width && dst_y < pad_height {
                unsafe {
                    let pixel = image.unsafe_get_pixel(src_x, src_y);
                    out.unsafe_put_pixel(dst_x, dst_y, pixel);
                }
            }
        }
    }

    Ok(out)
}

/// Trait that provides padding operations
///
/// Note: This operation changes the image dimensions, so there is no `_mut` variant
/// available. The algorithm creates a new image with different dimensions.
pub trait Padding<P: Pixel> {
    /// Add padding with specified size and position
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
    /// use imageops_ai::{Padding, Position, Image};
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

    /// Add padding to make the image square
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
    fn add_padding_square(self, color: P) -> Result<(Self, (i64, i64)), PaddingError>
    where
        Self: Sized;

    /// Calculate padding position
    ///
    /// This is a helper method that doesn't consume self.
    fn calculate_padding_position(
        &self,
        pad_size: (u32, u32),
        position: Position,
    ) -> Result<(i64, i64), PaddingError>;

    /// Calculate position and size for square padding
    ///
    /// This is a helper method that doesn't consume self.
    fn calculate_square_padding(&self) -> Result<((i64, i64), (u32, u32)), PaddingError>;

    /// Hidden _mut variant that is not available for this operation
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

    /// Hidden _mut variant that is not available for this operation
    #[doc(hidden)]
    fn add_padding_square_mut(&mut self, _color: P) -> Result<&mut Self, PaddingError> {
        unimplemented!(
            "add_padding_square_mut is not available because the operation changes image dimensions"
        )
    }
}

impl<P: Pixel> Padding<P> for Image<P> {
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

    fn add_padding_square(self, color: P) -> Result<(Self, (i64, i64)), PaddingError> {
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
    fn test_calculate_position() {
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
    fn test_add_padding_function() {
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
    fn test_padding_trait() {
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
    fn test_square_padding() {
        // Test rectangular image (width > height)
        let mut image: Image<Rgb<u8>> = Image::new(6, 4);
        for y in 0..4 {
            for x in 0..6 {
                image.put_pixel(x, y, Rgb([100, 150, 200]));
            }
        }

        let result = image.add_padding_square(Rgb([255, 255, 255]));
        assert!(result.is_ok());

        let (padded, pos) = result.unwrap();
        assert_eq!(padded.dimensions(), (6, 6)); // Should be square
        assert_eq!(pos, (0, 1)); // Centered vertically

        // Test square image (no padding needed)
        let square_image: Image<Rgb<u8>> = Image::new(4, 4);
        let result = square_image.add_padding_square(Rgb([255, 255, 255]));
        assert!(result.is_ok());

        let (padded, pos) = result.unwrap();
        assert_eq!(padded.dimensions(), (4, 4));
        assert_eq!(pos, (0, 0));
    }

    #[test]
    fn test_calculate_padding_position_errors() {
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
    fn test_padding_position_calculation() {
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
