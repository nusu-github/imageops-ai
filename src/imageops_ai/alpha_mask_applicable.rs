use crate::error::Error;
use crate::Image;
use image::{ImageBuffer, Luma, Pixel, Primitive, Rgb, Rgba};

/// アルファマスク操作の結果型
pub type AlphaMaskResult<T> = Result<T, Error>;

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
    fn apply_alpha_mask<SM>(&self, mask: &Image<Luma<SM>>) -> AlphaMaskResult<Image<Rgba<S>>>
    where
        SM: Primitive + 'static;
}

impl<S> ApplyAlphaMask<S> for Image<Rgb<S>>
where
    Rgb<S>: Pixel<Subpixel = S>,
    Rgba<S>: Pixel<Subpixel = S>,
    S: Primitive + 'static,
{
    fn apply_alpha_mask<SM>(&self, mask: &Image<Luma<SM>>) -> AlphaMaskResult<Image<Rgba<S>>>
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

        // ピクセルデータを処理
        let processed_pixels = self
            .pixels()
            .zip(mask.pixels())
            .flat_map(|(&image_pixel, mask_pixel)| {
                let Rgb([red, green, blue]) = image_pixel;
                let Luma([alpha_value]) = mask_pixel;

                // マスク値を対象の型の範囲にスケーリング
                let scaled_alpha = alpha_value.to_f32().unwrap() / mask_max * source_max;
                let alpha = S::from(scaled_alpha).unwrap();

                [red, green, blue, alpha]
            })
            .collect();

        // 結果のImageBufferを作成
        ImageBuffer::from_raw(self.width(), self.height(), processed_pixels)
            .ok_or(Error::ImageBufferCreationFailed)
    }
}
