use crate::error::BoxFilterError;
use crate::imageops_ai::summed_area_table::{CreateSummedAreaTable, SummedAreaTable};
use crate::Image;
use image::{ImageBuffer, Luma, Rgb};

/// 積分画像を使用したボックスフィルタ操作を提供するトレイト
///
/// このトレイトは積分画像（Summed-Area Table）を使用して
/// 効率的にボックスフィルタを適用します。
/// 通常のボックスフィルタと比較して、大きなカーネルサイズでも
/// 計算時間が一定になる利点があります。
pub trait BoxFilter {
    /// フィルタ処理の出力型
    type Output;

    /// フィルタ処理で発生する可能性のあるエラー型
    type Error;

    /// 積分画像を使用して指定した半径でボックスフィルタを適用する
    ///
    /// # 引数
    ///
    /// * `x_radius` - X方向の半径（ピクセル単位）
    /// * `y_radius` - Y方向の半径（ピクセル単位）
    ///
    /// # 戻り値
    ///
    /// フィルタ処理された画像、またはエラー
    ///
    /// # エラー
    ///
    /// * 半径が画像サイズに対して大きすぎる場合
    /// * 空の画像に対して処理を実行した場合
    /// * 積分画像の作成に失敗した場合
    ///
    /// # パフォーマンス
    ///
    /// 積分画像を使用するため、カーネルサイズに関係なく
    /// 一定時間で計算が完了します。大きなカーネルサイズの
    /// ボックスフィルタに特に有効です。
    fn box_filter(&self, x_radius: u32, y_radius: u32) -> Result<Self::Output, Self::Error>;

    /// 積分画像を使用して正方形のカーネルでボックスフィルタを適用する
    ///
    /// # 引数
    ///
    /// * `radius` - カーネルの半径（ピクセル単位）
    ///
    /// # 戻り値
    ///
    /// フィルタ処理された画像、またはエラー
    fn box_filter_square(&self, radius: u32) -> Result<Self::Output, Self::Error> {
        self.box_filter(radius, radius)
    }
}

impl BoxFilter for Image<Luma<f32>> {
    type Output = Self;
    type Error = BoxFilterError;

    fn box_filter(&self, x_radius: u32, y_radius: u32) -> Result<Self::Output, Self::Error> {
        let (width, height) = self.dimensions();

        // 画像が空の場合のチェック
        if width == 0 || height == 0 {
            return Err(BoxFilterError::EmptyImage);
        }

        // 半径が適切かチェック
        if x_radius >= width || y_radius >= height {
            return Err(BoxFilterError::RadiusTooLarge {
                x_radius,
                y_radius,
                width,
                height,
            });
        }

        // 積分画像を作成
        let sat = self
            .create_summed_area_table()
            .map_err(|_| BoxFilterError::EmptyImage)?;

        Ok(apply_sat_box_filter_luma(&sat, x_radius, y_radius))
    }
}

impl BoxFilter for Image<Luma<u8>> {
    type Output = Image<Luma<f32>>;
    type Error = BoxFilterError;

    fn box_filter(&self, x_radius: u32, y_radius: u32) -> Result<Self::Output, Self::Error> {
        let (width, height) = self.dimensions();

        // 画像が空の場合のチェック
        if width == 0 || height == 0 {
            return Err(BoxFilterError::EmptyImage);
        }

        // 半径が適切かチェック
        if x_radius >= width || y_radius >= height {
            return Err(BoxFilterError::RadiusTooLarge {
                x_radius,
                y_radius,
                width,
                height,
            });
        }

        // 積分画像を作成
        let sat: SummedAreaTable<u32> = self
            .create_summed_area_table()
            .map_err(|_| BoxFilterError::EmptyImage)?;

        Ok(apply_sat_box_filter_luma_u8(&sat, x_radius, y_radius))
    }
}

impl BoxFilter for Image<Rgb<f32>> {
    type Output = Self;
    type Error = BoxFilterError;

    fn box_filter(&self, x_radius: u32, y_radius: u32) -> Result<Self::Output, Self::Error> {
        let (width, height) = self.dimensions();

        // 画像が空の場合のチェック
        if width == 0 || height == 0 {
            return Err(BoxFilterError::EmptyImage);
        }

        // 半径が適切かチェック
        if x_radius >= width || y_radius >= height {
            return Err(BoxFilterError::RadiusTooLarge {
                x_radius,
                y_radius,
                width,
                height,
            });
        }

        Ok(apply_sat_box_filter_rgb(self, x_radius, y_radius))
    }
}

impl BoxFilter for Image<Rgb<u8>> {
    type Output = Image<Rgb<f32>>;
    type Error = BoxFilterError;

    fn box_filter(&self, x_radius: u32, y_radius: u32) -> Result<Self::Output, Self::Error> {
        let (width, height) = self.dimensions();

        // 画像が空の場合のチェック
        if width == 0 || height == 0 {
            return Err(BoxFilterError::EmptyImage);
        }

        // 半径が適切かチェック
        if x_radius >= width || y_radius >= height {
            return Err(BoxFilterError::RadiusTooLarge {
                x_radius,
                y_radius,
                width,
                height,
            });
        }

        Ok(apply_sat_box_filter_rgb_u8(self, x_radius, y_radius))
    }
}

/// 積分画像を使用してLuma<f32>画像にボックスフィルタを適用する内部実装
///
/// # 引数
///
/// * `sat` - 積分画像
/// * `x_radius` - X方向の半径
/// * `y_radius` - Y方向の半径
///
/// # 戻り値
///
/// フィルタ処理された画像
fn apply_sat_box_filter_luma(
    sat: &SummedAreaTable<f32>,
    x_radius: u32,
    y_radius: u32,
) -> Image<Luma<f32>> {
    let width = sat.width();
    let height = sat.height();
    let mut output = ImageBuffer::new(width, height);

    let _kernel_area = ((2 * x_radius + 1) * (2 * y_radius + 1)) as f32;

    for y in 0..height {
        for x in 0..width {
            // 矩形領域の境界を計算
            let x1 = (x as i32).saturating_sub(x_radius as i32);
            let y1 = (y as i32).saturating_sub(y_radius as i32);
            let x2 = ((x + x_radius).min(width - 1)) as i32;
            let y2 = ((y + y_radius).min(height - 1)) as i32;

            // 積分画像を使用して矩形領域の合計を計算
            let sum = sat.rectangle_sum(x1, y1, x2, y2);

            // 実際のカーネルサイズを計算（境界での調整）
            let actual_width = (x2 - x1 + 1) as f32;
            let actual_height = (y2 - y1 + 1) as f32;
            let actual_area = actual_width * actual_height;

            // 平均値を計算
            let average = sum / actual_area;

            output.put_pixel(x, y, Luma([average]));
        }
    }

    output
}

/// 積分画像を使用してLuma<u8>画像にボックスフィルタを適用する内部実装
fn apply_sat_box_filter_luma_u8(
    sat: &SummedAreaTable<u32>,
    x_radius: u32,
    y_radius: u32,
) -> Image<Luma<f32>> {
    let width = sat.width();
    let height = sat.height();
    let mut output = ImageBuffer::new(width, height);

    for y in 0..height {
        for x in 0..width {
            // 矩形領域の境界を計算
            let x1 = (x as i32).saturating_sub(x_radius as i32);
            let y1 = (y as i32).saturating_sub(y_radius as i32);
            let x2 = ((x + x_radius).min(width - 1)) as i32;
            let y2 = ((y + y_radius).min(height - 1)) as i32;

            // 積分画像を使用して矩形領域の合計を計算
            let sum = sat.rectangle_sum(x1, y1, x2, y2) as f32;

            // 実際のカーネルサイズを計算（境界での調整）
            let actual_width = (x2 - x1 + 1) as f32;
            let actual_height = (y2 - y1 + 1) as f32;
            let actual_area = actual_width * actual_height;

            // 平均値を計算
            let average = sum / actual_area;

            output.put_pixel(x, y, Luma([average]));
        }
    }

    output
}

/// 積分画像を使用してRGB<f32>画像にボックスフィルタを適用する内部実装
///
/// RGB画像は各チャンネルを個別に処理します
fn apply_sat_box_filter_rgb(
    image: &Image<Rgb<f32>>,
    x_radius: u32,
    y_radius: u32,
) -> Image<Rgb<f32>> {
    let (width, height) = image.dimensions();
    let mut output = ImageBuffer::new(width, height);

    // 各チャンネルの積分画像を作成
    let r_data: Vec<f32> = image.pixels().map(|p| p[0]).collect();
    let g_data: Vec<f32> = image.pixels().map(|p| p[1]).collect();
    let b_data: Vec<f32> = image.pixels().map(|p| p[2]).collect();

    let r_sat = SummedAreaTable::from_data(&r_data, width, height);
    let g_sat = SummedAreaTable::from_data(&g_data, width, height);
    let b_sat = SummedAreaTable::from_data(&b_data, width, height);

    for y in 0..height {
        for x in 0..width {
            // 矩形領域の境界を計算
            let x1 = (x as i32).saturating_sub(x_radius as i32);
            let y1 = (y as i32).saturating_sub(y_radius as i32);
            let x2 = ((x + x_radius).min(width - 1)) as i32;
            let y2 = ((y + y_radius).min(height - 1)) as i32;

            // 各チャンネルの合計を計算
            let r_sum = r_sat.rectangle_sum(x1, y1, x2, y2);
            let g_sum = g_sat.rectangle_sum(x1, y1, x2, y2);
            let b_sum = b_sat.rectangle_sum(x1, y1, x2, y2);

            // 実際のカーネルサイズを計算
            let actual_width = (x2 - x1 + 1) as f32;
            let actual_height = (y2 - y1 + 1) as f32;
            let actual_area = actual_width * actual_height;

            // 各チャンネルの平均値を計算
            let r_avg = r_sum / actual_area;
            let g_avg = g_sum / actual_area;
            let b_avg = b_sum / actual_area;

            output.put_pixel(x, y, Rgb([r_avg, g_avg, b_avg]));
        }
    }

    output
}

/// 積分画像を使用してRGB<u8>画像にボックスフィルタを適用する内部実装
fn apply_sat_box_filter_rgb_u8(
    image: &Image<Rgb<u8>>,
    x_radius: u32,
    y_radius: u32,
) -> Image<Rgb<f32>> {
    let (width, height) = image.dimensions();
    let mut output = ImageBuffer::new(width, height);

    // 各チャンネルのu32データを作成（オーバーフロー防止）
    let r_data: Vec<u32> = image.pixels().map(|p| u32::from(p[0])).collect();
    let g_data: Vec<u32> = image.pixels().map(|p| u32::from(p[1])).collect();
    let b_data: Vec<u32> = image.pixels().map(|p| u32::from(p[2])).collect();

    let r_sat = SummedAreaTable::from_data(&r_data, width, height);
    let g_sat = SummedAreaTable::from_data(&g_data, width, height);
    let b_sat = SummedAreaTable::from_data(&b_data, width, height);

    for y in 0..height {
        for x in 0..width {
            // 矩形領域の境界を計算
            let x1 = (x as i32).saturating_sub(x_radius as i32);
            let y1 = (y as i32).saturating_sub(y_radius as i32);
            let x2 = ((x + x_radius).min(width - 1)) as i32;
            let y2 = ((y + y_radius).min(height - 1)) as i32;

            // 各チャンネルの合計を計算
            let r_sum = r_sat.rectangle_sum(x1, y1, x2, y2) as f32;
            let g_sum = g_sat.rectangle_sum(x1, y1, x2, y2) as f32;
            let b_sum = b_sat.rectangle_sum(x1, y1, x2, y2) as f32;

            // 実際のカーネルサイズを計算
            let actual_width = (x2 - x1 + 1) as f32;
            let actual_height = (y2 - y1 + 1) as f32;
            let actual_area = actual_width * actual_height;

            // 各チャンネルの平均値を計算
            let r_avg = r_sum / actual_area;
            let g_avg = g_sum / actual_area;
            let b_avg = b_sum / actual_area;

            output.put_pixel(x, y, Rgb([r_avg, g_avg, b_avg]));
        }
    }

    output
}
