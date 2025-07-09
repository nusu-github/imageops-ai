use image::{GenericImageView, ImageBuffer, Pixel};
use imageproc::definitions::Image;

use crate::error::PaddingError;

/// Enum to specify padding position
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Position {
    /// Top center
    TopCenter,
    /// Bottom center
    BottomCenter,
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

/// Calculate offset from image size and canvas size
///
/// # Arguments
///
/// * `img_size` - Original image size (width, height)
/// * `canvas_size` - Canvas size (width, height)
/// * `position` - Padding position
///
/// # Returns
///
/// Returns the position (x, y) where the image should be placed on success
///
/// # Errors
///
/// * Returns error when padding size is smaller than original image size
pub fn calc_offset(
    img_size: (u32, u32),
    canvas_size: (u32, u32),
    position: Position,
) -> Result<(i64, i64), PaddingError> {
    let (width, height) = img_size;
    let (canvas_width, canvas_height) = canvas_size;

    if canvas_width < width {
        return Err(PaddingError::PaddingWidthTooSmall {
            width,
            pad_width: canvas_width,
        });
    }

    if canvas_height < height {
        return Err(PaddingError::PaddingHeightTooSmall {
            height,
            pad_height: canvas_height,
        });
    }

    let (x, y) = match position {
        Position::TopCenter => ((canvas_width - width) / 2, 0),
        Position::BottomCenter => ((canvas_width - width) / 2, canvas_height - height),
        Position::Left => (0, (canvas_height - height) / 2),
        Position::Right => (canvas_width - width, (canvas_height - height) / 2),
        Position::TopLeft => (0, 0),
        Position::TopRight => (canvas_width - width, 0),
        Position::BottomLeft => (0, canvas_height - height),
        Position::BottomRight => (canvas_width - width, canvas_height - height),
        Position::Center => ((canvas_width - width) / 2, (canvas_height - height) / 2),
    };

    Ok((x.into(), y.into()))
}

/// Pad image with specified size and position (function-based implementation)
///
/// # Arguments
///
/// * `image` - Original image
/// * `canvas_size` - Canvas size (width, height)
/// * `position` - Padding position
/// * `color` - Padding color
///
/// # Returns
///
/// Padded image
///
pub fn pad_image<I, P>(
    image: &I,
    canvas_size: (u32, u32),
    position: Position,
    color: P,
) -> Result<Image<P>, PaddingError>
where
    I: GenericImageView<Pixel = P>,
    P: Pixel,
{
    let (width, height) = image.dimensions();
    let (x, y) = calc_offset((width, height), canvas_size, position)?;
    let (canvas_width, canvas_height) = canvas_size;

    // Create buffer filled with padding color
    let mut out = ImageBuffer::from_pixel(canvas_width, canvas_height, color);

    // Iterator-based image copying following guidelines
    blit(image, &mut out, x, y, width, height);

    Ok(out)
}

/// Iterator-based image copying
///
/// This function uses safe iterator-based processing:
/// 1. Utilizes iterator chains for memory-efficient processing
/// 2. Avoids manual loops as per guidelines
/// 3. Uses safe pixel access methods
#[inline]
fn blit<I, P>(src: &I, dst: &mut Image<P>, x_off: i64, y_off: i64, width: u32, height: u32)
where
    I: GenericImageView<Pixel = P>,
    P: Pixel,
{
    let dst_x_start = x_off as u32;
    let dst_y_start = y_off as u32;

    // Iterator-based pixel copying using safe methods
    (0..height)
        .flat_map(|src_y| {
            let dst_y = dst_y_start + src_y;
            (0..width).map(move |src_x| {
                let dst_x = dst_x_start + src_x;
                (src_x, src_y, dst_x, dst_y)
            })
        })
        .for_each(|(src_x, src_y, dst_x, dst_y)| {
            let pixel = src.get_pixel(src_x, src_y);
            dst.put_pixel(dst_x, dst_y, pixel);
        });
}

/// Extension trait that provides padding operations
///
/// Note: This operation changes the image dimensions, so there is no `_mut` variant
/// available. The algorithm creates a new image with different dimensions.
pub trait PadExt<P: Pixel> {
    /// Add padding with specified size and position
    ///
    /// This consumes the original image.
    ///
    /// # Arguments
    ///
    /// * `canvas_size` - Canvas size (width, height)
    /// * `position` - Padding position
    /// * `color` - Padding color
    ///
    /// # Returns
    ///
    /// Tuple of (padded image, position (x, y) where the original image was placed)
    ///
    /// # Examples
    /// ```no_run
    /// use imageops_ai::{PadExt, Position, Image};
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
        canvas_size: (u32, u32),
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
        canvas_size: (u32, u32),
        position: Position,
    ) -> Result<(i64, i64), PaddingError>;

    /// Calculate position and size for square padding
    ///
    /// This is a helper method that doesn't consume self.
    fn calculate_square_padding(&self) -> Result<((i64, i64), (u32, u32)), PaddingError>;

    /// This mut variant is not available - DO NOT USE
    #[doc(hidden)]
    fn pad_mut(
        &mut self,
        _canvas_size: (u32, u32),
        _position: Position,
        _color: P,
    ) -> Result<&mut Self, PaddingError> {
        unimplemented!("pad_mut is not available because the operation changes image dimensions")
    }

    /// This mut variant is not available - DO NOT USE
    #[doc(hidden)]
    fn pad_square_mut(&mut self, _color: P) -> Result<&mut Self, PaddingError> {
        unimplemented!(
            "pad_square_mut is not available because the operation changes image dimensions"
        )
    }
}

impl<P: Pixel> PadExt<P> for Image<P> {
    fn add_padding(
        self,
        canvas_size: (u32, u32),
        position: Position,
        color: P,
    ) -> Result<(Self, (i64, i64)), PaddingError> {
        let (width, height) = self.dimensions();
        let pos = calc_offset((width, height), canvas_size, position)?;
        let padded = pad_image(&self, canvas_size, position, color)?;
        Ok((padded, pos))
    }

    fn add_padding_square(self, color: P) -> Result<(Self, (i64, i64)), PaddingError> {
        let ((_x, _y), canvas_size) = self.calculate_square_padding()?;
        self.add_padding(canvas_size, Position::Center, color)
    }

    fn calculate_padding_position(
        &self,
        canvas_size: (u32, u32),
        position: Position,
    ) -> Result<(i64, i64), PaddingError> {
        let (width, height) = self.dimensions();
        calc_offset((width, height), canvas_size, position)
    }

    fn calculate_square_padding(&self) -> Result<((i64, i64), (u32, u32)), PaddingError> {
        let (width, height) = self.dimensions();

        let canvas_size = if width > height {
            (width, width)
        } else {
            (height, height)
        };

        self.calculate_padding_position(canvas_size, Position::Center)
            .map(|(x, y)| ((x, y), canvas_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;
    use image::Rgb;

    #[test]
    fn test_calc_offset() {
        // Test center position
        let pos = calc_offset((10, 10), (20, 20), Position::Center);
        assert_eq!(pos, Ok((5, 5)));

        // Test top-left position
        let pos = calc_offset((10, 10), (20, 20), Position::TopLeft);
        assert_eq!(pos, Ok((0, 0)));

        // Test top-right position
        let pos = calc_offset((10, 10), (20, 20), Position::TopRight);
        assert_eq!(pos, Ok((10, 0)));

        // Test bottom-left position
        let pos = calc_offset((10, 10), (20, 20), Position::BottomLeft);
        assert_eq!(pos, Ok((0, 10)));

        // Test bottom-right position
        let pos = calc_offset((10, 10), (20, 20), Position::BottomRight);
        assert_eq!(pos, Ok((10, 10)));
    }

    #[test]
    fn test_add_padding_function() {
        let src_img = create_test_rgb_image(); // 2x2 image
        let fill_color = Rgb([255, 255, 255]); // White

        // Test padding to larger size
        let result = pad_image(&src_img, (4, 4), Position::Center, fill_color);
        assert!(result.is_ok());

        let padded = result.unwrap();
        assert_eq!(padded.dimensions(), (4, 4));

        // Test invalid padding (smaller than original)
        let result = pad_image(&src_img, (1, 1), Position::Center, fill_color);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PaddingError::PaddingWidthTooSmall { .. }
        ));
    }

    #[test]
    fn test_padding_trait() {
        let src_img = create_test_rgb_image(); // 2x2 image
        let fill_color = Rgb([255, 255, 255]); // White

        // Store original pixels before moving image
        let orig_00 = *src_img.get_pixel(0, 0);
        let orig_10 = *src_img.get_pixel(1, 0);
        let orig_01 = *src_img.get_pixel(0, 1);
        let orig_11 = *src_img.get_pixel(1, 1);

        // Test regular padding
        let result = src_img.add_padding((4, 4), Position::TopLeft, fill_color);
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
        let mut rect_img: Image<Rgb<u8>> = Image::new(6, 4);
        for y in 0..4 {
            for x in 0..6 {
                rect_img.put_pixel(x, y, Rgb([100, 150, 200]));
            }
        }

        let result = rect_img.add_padding_square(Rgb([255, 255, 255]));
        assert!(result.is_ok());

        let (padded, pos) = result.unwrap();
        assert_eq!(padded.dimensions(), (6, 6)); // Should be square
        assert_eq!(pos, (0, 1)); // Centered vertically

        // Test square image (no padding needed)
        let square_img: Image<Rgb<u8>> = Image::new(4, 4);
        let result = square_img.add_padding_square(Rgb([255, 255, 255]));
        assert!(result.is_ok());

        let (padded, pos) = result.unwrap();
        assert_eq!(padded.dimensions(), (4, 4));
        assert_eq!(pos, (0, 0));
    }

    #[test]
    fn test_calculate_padding_position_errors() {
        let src_img = create_test_rgb_image(); // 2x2 image

        // Test padding size too small (width)
        let result = src_img.calculate_padding_position((1, 4), Position::Center);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PaddingError::PaddingWidthTooSmall { .. }
        ));

        // Test padding size too small (height)
        let result = src_img.calculate_padding_position((4, 1), Position::Center);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PaddingError::PaddingHeightTooSmall { .. }
        ));
    }

    #[test]
    fn test_padding_position_calculation() {
        let src_img = create_test_rgb_image(); // 2x2 image

        // Test all positions with 6x6 padding
        let positions = [
            (Position::TopLeft, (0, 0)),
            (Position::TopRight, (4, 0)),
            (Position::BottomLeft, (0, 4)),
            (Position::BottomRight, (4, 4)),
            (Position::Center, (2, 2)),
        ];

        for (pos, expected) in positions {
            let result = src_img.calculate_padding_position((6, 6), pos);
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), expected);
        }
    }
}
