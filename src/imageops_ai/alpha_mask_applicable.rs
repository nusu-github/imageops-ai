use crate::error::Error;
use crate::Image;
use image::{ImageBuffer, Luma, Pixel, Primitive, Rgb, Rgba};

/// 画像にアルファマスクを適用する機能を提供するトレイト
///
/// このトレイトは、RGB画像にグレースケールマスクを適用して
/// RGBA画像を生成する機能を提供します。
pub trait ApplyAlphaMask<S>
where
    Rgba<S>: Pixel<Subpixel = S>,
    S: Primitive + 'static,
{
    /// 指定されたマスクを画像に適用してアルファチャンネル付きの画像を生成します
    ///
    /// # Arguments
    ///
    /// * `mask` - 適用するアルファマスク（グレースケール画像）
    ///
    /// # Returns
    ///
    /// アルファチャンネルが追加された RGBA 画像
    ///
    /// # Errors
    ///
    /// * `Error::DimensionMismatch` - 画像とマスクの寸法が一致しない場合
    /// * `Error::ImageBufferCreationFailed` - 結果画像の作成に失敗した場合
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use imageops_ai::{Image, ApplyAlphaMask};
    /// use image::{ImageBuffer, Rgb, Luma};
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // RGB画像とマスクが同じ寸法である必要があります
    /// let rgb_image: Image<Rgb<u8>> = ImageBuffer::new(10, 10);
    /// let mask: Image<Luma<u8>> = ImageBuffer::new(10, 10);
    ///
    /// let rgba_image = rgb_image.apply_alpha_mask(&mask)?;
    /// # Ok(())
    /// # }
    /// ```
    fn apply_alpha_mask(&self, mask: &Image<Luma<S>>) -> Result<Image<Rgba<S>>, Error>;
}

/// 画像にアルファマスクを適用する機能を提供するトレイト
///
/// このトレイトは、RGB画像にグレースケールマスクを適用して
/// RGBA画像を生成する機能を提供します。
/// こちらはマスクと画像の型が異なる場合に使用します。
pub trait ApplyAlphaMaskConvert<S>
where
    Rgba<S>: Pixel<Subpixel = S>,
    S: Primitive + 'static,
{
    /// 指定されたマスクを画像に適用してアルファチャンネル付きの画像を生成します
    ///
    /// # Arguments
    ///
    /// * `mask` - 適用するアルファマスク（グレースケール画像）
    ///
    /// # Returns
    ///
    /// アルファチャンネルが追加された RGBA 画像
    ///
    /// # Errors
    ///
    /// * `Error::DimensionMismatch` - 画像とマスクの寸法が一致しない場合
    /// * `Error::ImageBufferCreationFailed` - 結果画像の作成に失敗した場合
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use imageops_ai::{Image, ApplyAlphaMask};
    /// use image::{ImageBuffer, Rgb, Luma};
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// // RGB画像とマスクが同じ寸法である必要があります
    /// let rgb_image: Image<Rgb<u8>> = ImageBuffer::new(10, 10);
    /// let mask: Image<Luma<u8>> = ImageBuffer::new(10, 10);
    ///
    /// let rgba_image = rgb_image.apply_alpha_mask(&mask)?;
    /// # Ok(())
    /// # }
    /// ```
    fn apply_alpha_mask<SM>(&self, mask: &Image<Luma<SM>>) -> Result<Image<Rgba<S>>, Error>
    where
        SM: Primitive + 'static;
}

impl<S> ApplyAlphaMask<S> for Image<Rgb<S>>
where
    Rgb<S>: Pixel<Subpixel = S>,
    Rgba<S>: Pixel<Subpixel = S>,
    S: Primitive + 'static,
{
    fn apply_alpha_mask(&self, mask: &Image<Luma<S>>) -> Result<Image<Rgba<S>>, Error> {
        // 寸法の一致を確認
        if self.dimensions() != mask.dimensions() {
            return Err(Error::DimensionMismatch);
        }

        let mut image_buffer = ImageBuffer::new(self.width(), self.height());

        for (x, y, pixel) in image_buffer.enumerate_pixels_mut() {
            let Rgba([red, green, blue, _alpha]) = *pixel;
            let Luma([alpha_value]) = mask.get_pixel(x, y);

            *pixel = Rgba([red, green, blue, *alpha_value]);
        }

        Ok(image_buffer)
    }
}

impl<S> ApplyAlphaMaskConvert<S> for Image<Rgb<S>>
where
    Rgb<S>: Pixel<Subpixel = S>,
    Rgba<S>: Pixel<Subpixel = S>,
    S: Primitive + 'static,
{
    fn apply_alpha_mask<SM>(&self, mask: &Image<Luma<SM>>) -> Result<Image<Rgba<S>>, Error>
    where
        SM: Primitive + 'static,
    {
        // 寸法の一致を確認
        if self.dimensions() != mask.dimensions() {
            return Err(Error::DimensionMismatch);
        }

        // 型の最大値を取得してスケーリング用に使用
        // SとSMは必ずu8, u16, f32なので、uncheckedは安全
        let source_max = unsafe { S::DEFAULT_MAX_VALUE.to_f32().unwrap_unchecked() };
        let mask_max = unsafe { SM::DEFAULT_MAX_VALUE.to_f32().unwrap_unchecked() };

        let mut image_buffer = ImageBuffer::new(self.width(), self.height());

        for (x, y, pixel) in image_buffer.enumerate_pixels_mut() {
            let Rgba([red, green, blue, _alpha]) = *pixel;
            let Luma([alpha_value]) = mask.get_pixel(x, y);

            let scaled_alpha = alpha_value.to_f32().unwrap() / mask_max * source_max;
            let alpha = S::from(scaled_alpha).unwrap();

            *pixel = Rgba([red, green, blue, alpha]);
        }

        Ok(image_buffer)
    }
}
