use crate::error::Error;
use crate::Image;
use image::{ImageBuffer, Luma, Pixel, Primitive};

/// 積分画像操作の結果型
pub type SummedAreaTableResult<T> = Result<T, Error>;

/// 積分画像（Summed-Area Table）の構造体
///
/// 積分画像は、原点(0,0)から指定座標(x,y)までの矩形領域内の
/// 全ピクセル値の累積和を効率的に計算するためのデータ構造です。
pub struct SummedAreaTable<T> {
    /// 積分画像のデータ
    data: Vec<T>,
    /// 画像の幅
    width: u32,
    /// 画像の高さ
    height: u32,
}

/// 画像から積分画像を作成する機能を提供するトレイト
///
/// このトレイトは、画像データから効率的な矩形領域の合計計算を
/// 可能にする積分画像を生成する機能を提供します。
pub trait CreateSummedAreaTable<T>
where
    T: Primitive + 'static,
{
    /// 画像から積分画像を作成します
    ///
    /// # Returns
    ///
    /// 作成された積分画像
    ///
    /// # Errors
    ///
    /// * `Error::ImageBufferCreationFailed` - 積分画像の作成に失敗した場合
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use imageops_ai::{Image, CreateSummedAreaTable, SummedAreaTable};
    /// use image::{ImageBuffer, Luma};
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let image: Image<Luma<u8>> = ImageBuffer::new(10, 10);
    /// let sat: SummedAreaTable<u32> = image.create_summed_area_table()?;
    /// # Ok(())
    /// # }
    /// ```
    fn create_summed_area_table(&self) -> SummedAreaTableResult<SummedAreaTable<T>>;
}

impl<T> CreateSummedAreaTable<T> for Image<Luma<T>>
where
    T: Primitive + 'static,
{
    fn create_summed_area_table(&self) -> SummedAreaTableResult<SummedAreaTable<T>> {
        let (width, height) = self.dimensions();
        let mut data = vec![T::zero(); (width * height) as usize];

        for y in 0..height {
            for x in 0..width {
                let pixel_value = self.get_pixel(x, y).channels()[0];
                let current_index = (y * width + x) as usize;

                // 積分画像の計算式:
                // sat(x, y) = src(x, y) + sat(x-1, y) + sat(x, y-1) - sat(x-1, y-1)
                let mut sum = pixel_value;

                // 左のピクセルの値を加算
                if x > 0 {
                    sum = sum + data[current_index - 1];
                }

                // 上のピクセルの値を加算
                if y > 0 {
                    sum = sum + data[((y - 1) * width + x) as usize];
                }

                // 左上のピクセルの値を減算（重複分を除去）
                if x > 0 && y > 0 {
                    sum = sum - data[((y - 1) * width + (x - 1)) as usize];
                }

                data[current_index] = sum;
            }
        }

        Ok(SummedAreaTable {
            data,
            width,
            height,
        })
    }
}

/// u8画像用の特別な実装：オーバーフローを防ぐためu32を使用
impl CreateSummedAreaTable<u32> for Image<Luma<u8>> {
    fn create_summed_area_table(&self) -> SummedAreaTableResult<SummedAreaTable<u32>> {
        let (width, height) = self.dimensions();
        let mut data = vec![0u32; (width * height) as usize];

        for y in 0..height {
            for x in 0..width {
                let pixel_value = u32::from(self.get_pixel(x, y).channels()[0]);
                let current_index = (y * width + x) as usize;

                // 積分画像の計算式:
                // sat(x, y) = src(x, y) + sat(x-1, y) + sat(x, y-1) - sat(x-1, y-1)
                let mut sum = pixel_value;

                // 左のピクセルの値を加算
                if x > 0 {
                    sum += data[current_index - 1];
                }

                // 上のピクセルの値を加算
                if y > 0 {
                    sum += data[((y - 1) * width + x) as usize];
                }

                // 左上のピクセルの値を減算（重複分を除去）
                if x > 0 && y > 0 {
                    sum -= data[((y - 1) * width + (x - 1)) as usize];
                }

                data[current_index] = sum;
            }
        }

        Ok(SummedAreaTable {
            data,
            width,
            height,
        })
    }
}

impl<T> SummedAreaTable<T>
where
    T: Primitive,
{
    /// 指定された画像から積分画像を作成します
    ///
    /// # 引数
    /// * `image` - 元となる画像バッファ
    ///
    /// # 戻り値
    /// 作成された積分画像
    #[must_use]
    pub fn from_image<P>(image: &ImageBuffer<P, Vec<P::Subpixel>>) -> Self
    where
        P: Pixel<Subpixel = T>,
    {
        let (width, height) = image.dimensions();
        let mut data = vec![T::zero(); (width * height) as usize];

        for y in 0..height {
            for x in 0..width {
                let pixel_value = image.get_pixel(x, y).channels()[0];
                let current_index = (y * width + x) as usize;

                // 積分画像の計算式:
                // sat(x, y) = src(x, y) + sat(x-1, y) + sat(x, y-1) - sat(x-1, y-1)
                let mut sum = pixel_value;

                // 左のピクセルの値を加算
                if x > 0 {
                    sum = sum + data[current_index - 1];
                }

                // 上のピクセルの値を加算
                if y > 0 {
                    sum = sum + data[((y - 1) * width + x) as usize];
                }

                // 左上のピクセルの値を減算（重複分を除去）
                if x > 0 && y > 0 {
                    sum = sum - data[((y - 1) * width + (x - 1)) as usize];
                }

                data[current_index] = sum;
            }
        }

        Self {
            data,
            width,
            height,
        }
    }

    /// 単一チャンネルのデータから積分画像を作成します
    ///
    /// # 引数
    /// * `data` - 元となる画像データ（行優先順序）
    /// * `width` - 画像の幅
    /// * `height` - 画像の高さ
    ///
    /// # 戻り値
    /// 作成された積分画像
    pub fn from_data(data: &[T], width: u32, height: u32) -> Self {
        assert_eq!(data.len(), (width * height) as usize);

        let mut sat_data = vec![T::zero(); (width * height) as usize];

        for y in 0..height {
            for x in 0..width {
                let current_index = (y * width + x) as usize;
                let pixel_value = data[current_index];

                let mut sum = pixel_value;

                // 左のピクセルの値を加算
                if x > 0 {
                    sum = sum + sat_data[current_index - 1];
                }

                // 上のピクセルの値を加算
                if y > 0 {
                    sum = sum + sat_data[((y - 1) * width + x) as usize];
                }

                // 左上のピクセルの値を減算（重複分を除去）
                if x > 0 && y > 0 {
                    sum = sum - sat_data[((y - 1) * width + (x - 1)) as usize];
                }

                sat_data[current_index] = sum;
            }
        }

        Self {
            data: sat_data,
            width,
            height,
        }
    }

    /// 指定された座標での積分画像の値を取得します
    ///
    /// # 引数
    /// * `x` - X座標
    /// * `y` - Y座標
    ///
    /// # 戻り値
    /// 積分画像の値、座標が範囲外の場合は0
    #[must_use]
    pub fn get(&self, x: i32, y: i32) -> T {
        if x < 0 || y < 0 || x >= self.width as i32 || y >= self.height as i32 {
            T::zero()
        } else {
            self.data[(y as u32 * self.width + x as u32) as usize]
        }
    }

    /// 指定された矩形領域内のピクセル値の合計を計算します
    ///
    /// # 引数
    /// * `x1` - 矩形の左上X座標（含む）
    /// * `y1` - 矩形の左上Y座標（含む）
    /// * `x2` - 矩形の右下X座標（含む）
    /// * `y2` - 矩形の右下Y座標（含む）
    ///
    /// # 戻り値
    /// 矩形領域内のピクセル値の合計
    ///
    /// # 計算式
    /// Sum = sat(x2, y2) - sat(x1-1, y2) - sat(x2, y1-1) + sat(x1-1, y1-1)
    #[must_use]
    pub fn rectangle_sum(&self, x1: i32, y1: i32, x2: i32, y2: i32) -> T {
        // 範囲チェック
        let x1 = x1.max(0);
        let y1 = y1.max(0);
        let x2 = x2.min(self.width as i32 - 1);
        let y2 = y2.min(self.height as i32 - 1);

        if x1 > x2 || y1 > y2 {
            return T::zero();
        }

        // 積分画像を使った矩形領域の合計計算
        // Sum = sat(x2, y2) - sat(x1-1, y2) - sat(x2, y1-1) + sat(x1-1, y1-1)
        let bottom_right = self.get(x2, y2);
        let top_right = self.get(x2, y1 - 1);
        let bottom_left = self.get(x1 - 1, y2);
        let top_left = self.get(x1 - 1, y1 - 1);

        bottom_right - top_right - bottom_left + top_left
    }

    /// 画像の幅を取得します
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// 画像の高さを取得します
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// 積分画像の生データへの参照を取得します
    #[must_use]
    pub fn data(&self) -> &[T] {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma};

    #[test]
    fn test_trait_based_create_summed_area_table() {
        // 3x3のテスト画像をImage<Luma<u8>>として作成
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let image: Image<Luma<u8>> = ImageBuffer::from_raw(3, 3, data).unwrap();

        // トレイトメソッドを使用して積分画像を作成
        let sat: SummedAreaTable<u32> = image.create_summed_area_table().unwrap();

        // 積分画像の期待値をチェック
        // 1  3  6
        // 5  12 21
        // 12 27 45

        assert_eq!(sat.get(0, 0), 1);
        assert_eq!(sat.get(1, 0), 3);
        assert_eq!(sat.get(2, 0), 6);
        assert_eq!(sat.get(0, 1), 5);
        assert_eq!(sat.get(1, 1), 12);
        assert_eq!(sat.get(2, 1), 21);
        assert_eq!(sat.get(0, 2), 12);
        assert_eq!(sat.get(1, 2), 27);
        assert_eq!(sat.get(2, 2), 45);

        // 矩形の合計も確認
        assert_eq!(sat.rectangle_sum(0, 0, 2, 2), 45);
        assert_eq!(sat.rectangle_sum(1, 1, 1, 1), 5);
    }

    #[test]
    fn test_basic_summed_area_table() {
        // 3x3のテスト画像
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];

        let sat = SummedAreaTable::from_data(&data, 3, 3);

        // 積分画像の期待値
        // 1  3  6
        // 5  12 21
        // 12 27 45

        assert_eq!(sat.get(0, 0), 1);
        assert_eq!(sat.get(1, 0), 3);
        assert_eq!(sat.get(2, 0), 6);
        assert_eq!(sat.get(0, 1), 5);
        assert_eq!(sat.get(1, 1), 12);
        assert_eq!(sat.get(2, 1), 21);
        assert_eq!(sat.get(0, 2), 12);
        assert_eq!(sat.get(1, 2), 27);
        assert_eq!(sat.get(2, 2), 45);
    }

    #[test]
    fn test_rectangle_sum() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];

        let sat = SummedAreaTable::from_data(&data, 3, 3);

        // 全体の合計 (0,0) から (2,2)
        assert_eq!(sat.rectangle_sum(0, 0, 2, 2), 45);

        // 中央の1ピクセル (1,1) から (1,1)
        assert_eq!(sat.rectangle_sum(1, 1, 1, 1), 5);

        // 2x2領域 (0,0) から (1,1)
        assert_eq!(sat.rectangle_sum(0, 0, 1, 1), 12);

        // 右下の2x2領域 (1,1) から (2,2)
        assert_eq!(sat.rectangle_sum(1, 1, 2, 2), 28);
    }

    #[test]
    fn test_boundary_conditions() {
        let data = vec![1, 2, 3, 4];
        let sat = SummedAreaTable::from_data(&data, 2, 2);

        // 範囲外のアクセス
        assert_eq!(sat.get(-1, 0), 0);
        assert_eq!(sat.get(0, -1), 0);
        assert_eq!(sat.get(2, 0), 0);
        assert_eq!(sat.get(0, 2), 0);

        // 無効な矩形
        assert_eq!(sat.rectangle_sum(1, 1, 0, 0), 0);
        assert_eq!(sat.rectangle_sum(-1, -1, 0, 0), 1);
    }
}
