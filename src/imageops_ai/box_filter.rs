use crate::error::BoxFilterError;
use crate::Image;
use image::{GenericImage, GenericImageView, ImageBuffer, Luma, Rgb, Rgb32FImage};

/// ボックスフィルタ操作を提供するトレイト
///
/// ボックスフィルタは各ピクセルの近傍領域の平均値を計算することで
/// 画像をぼかす効果を提供します。
pub trait BoxFilter {
    /// フィルタ処理の出力型
    type Output;

    /// フィルタ処理で発生する可能性のあるエラー型
    type Error;

    /// 指定した半径でボックスフィルタを適用する
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
    fn box_filter(&self, x_radius: u32, y_radius: u32) -> Result<Self::Output, Self::Error>;

    /// 正方形のカーネルでボックスフィルタを適用する
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

impl BoxFilter for Image<Rgb<f32>> {
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

        Ok(apply_box_filter_rgb(self, x_radius, y_radius))
    }
}

impl BoxFilter for Image<Luma<f32>> {
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

        Ok(apply_box_filter_luma(self, x_radius, y_radius))
    }
}

/// RGB画像にボックスフィルタを適用する内部実装
///
/// この関数は画像の各行と列に対して running sum を計算し、
/// 効率的にボックスフィルタを適用します。
fn apply_box_filter_rgb(image: &Rgb32FImage, x_radius: u32, y_radius: u32) -> Image<Rgb<f32>> {
    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::new(width, height);

    let kernel_width = (2 * x_radius + 1) as f32;
    let kernel_height = (2 * y_radius + 1) as f32;

    // 水平方向のフィルタリング
    let mut row_buffer = vec![[0.0; 3]; (width + 2 * x_radius) as usize];
    for y in 0..height {
        compute_row_running_sum_rgb(image, y, &mut row_buffer, x_radius);

        // 最初のピクセル
        let val = row_buffer[(2 * x_radius) as usize].map(|v| v / kernel_width);
        unsafe {
            debug_assert!(out.in_bounds(0, y));
            out.unsafe_put_pixel(0, y, Rgb(val));
        }

        // 残りのピクセル
        for x in 1..width {
            let u = (x + 2 * x_radius) as usize;
            let l = (x - 1) as usize;
            let val = [0, 1, 2].map(|i| (row_buffer[u][i] - row_buffer[l][i]) / kernel_width);
            unsafe {
                debug_assert!(out.in_bounds(x, y));
                out.unsafe_put_pixel(x, y, Rgb(val));
            }
        }
    }

    // 垂直方向のフィルタリング
    let mut col_buffer = vec![[0.0; 3]; (height + 2 * y_radius) as usize];
    for x in 0..width {
        compute_column_running_sum_rgb(&out, x, &mut col_buffer, y_radius);

        // 最初のピクセル
        let val = col_buffer[(2 * y_radius) as usize].map(|v| v / kernel_height);
        unsafe {
            debug_assert!(out.in_bounds(x, 0));
            out.unsafe_put_pixel(x, 0, Rgb(val));
        }

        // 残りのピクセル
        for y in 1..height {
            let u = (y + 2 * y_radius) as usize;
            let l = (y - 1) as usize;
            let val = [0, 1, 2].map(|i| (col_buffer[u][i] - col_buffer[l][i]) / kernel_height);
            unsafe {
                debug_assert!(out.in_bounds(x, y));
                out.unsafe_put_pixel(x, y, Rgb(val));
            }
        }
    }

    out
}

/// Luma画像にボックスフィルタを適用する内部実装
fn apply_box_filter_luma(
    image: &Image<Luma<f32>>,
    x_radius: u32,
    y_radius: u32,
) -> Image<Luma<f32>> {
    let (width, height) = image.dimensions();
    let mut out = ImageBuffer::new(width, height);

    let kernel_width = (2 * x_radius + 1) as f32;
    let kernel_height = (2 * y_radius + 1) as f32;

    // 水平方向のフィルタリング
    let mut row_buffer = vec![0.0; (width + 2 * x_radius) as usize];
    for y in 0..height {
        compute_row_running_sum_luma(image, y, &mut row_buffer, x_radius);

        // 最初のピクセル
        let val = row_buffer[(2 * x_radius) as usize] / kernel_width;
        unsafe {
            debug_assert!(out.in_bounds(0, y));
            out.unsafe_put_pixel(0, y, Luma([val]));
        }

        // 残りのピクセル
        for x in 1..width {
            let u = (x + 2 * x_radius) as usize;
            let l = (x - 1) as usize;
            let val = (row_buffer[u] - row_buffer[l]) / kernel_width;
            unsafe {
                debug_assert!(out.in_bounds(x, y));
                out.unsafe_put_pixel(x, y, Luma([val]));
            }
        }
    }

    // 垂直方向のフィルタリング
    let mut col_buffer = vec![0.0; (height + 2 * y_radius) as usize];
    for x in 0..width {
        compute_column_running_sum_luma(&out, x, &mut col_buffer, y_radius);

        // 最初のピクセル
        let val = col_buffer[(2 * y_radius) as usize] / kernel_height;
        unsafe {
            debug_assert!(out.in_bounds(x, 0));
            out.unsafe_put_pixel(x, 0, Luma([val]));
        }

        // 残りのピクセル
        for y in 1..height {
            let u = (y + 2 * y_radius) as usize;
            let l = (y - 1) as usize;
            let val = (col_buffer[u] - col_buffer[l]) / kernel_height;
            unsafe {
                debug_assert!(out.in_bounds(x, y));
                out.unsafe_put_pixel(x, y, Luma([val]));
            }
        }
    }

    out
}

/// RGB画像の指定行のrunning sumを計算する
///
/// # 引数
///
/// * `image` - 対象画像
/// * `row` - 行番号
/// * `buffer` - 結果を格納するバッファ
/// * `padding` - パディング幅
fn compute_row_running_sum_rgb(
    image: &Rgb32FImage,
    row: u32,
    buffer: &mut [[f32; 3]],
    padding: u32,
) {
    let (width, _height) = image.dimensions();
    let (width, padding) = (width as usize, padding as usize);

    let row_data = &(**image)[width * row as usize * 3..][..width * 3];
    let first = [row_data[0], row_data[1], row_data[2]];
    let last = [
        row_data[width * 3 - 3],
        row_data[width * 3 - 2],
        row_data[width * 3 - 1],
    ];

    let mut sum = [0.0; 3];

    // 左側のパディング
    for b in &mut buffer[..padding] {
        for i in 0..3 {
            sum[i] += first[i];
            b[i] = sum[i];
        }
    }

    // メイン部分
    for (b, chunk) in buffer[padding..].iter_mut().zip(row_data.chunks(3)) {
        for i in 0..3 {
            sum[i] += chunk[i];
            b[i] = sum[i];
        }
    }

    // 右側のパディング
    for b in &mut buffer[padding + width..] {
        for i in 0..3 {
            sum[i] += last[i];
            b[i] = sum[i];
        }
    }
}

/// RGB画像の指定列のrunning sumを計算する
fn compute_column_running_sum_rgb(
    image: &Rgb32FImage,
    column: u32,
    buffer: &mut [[f32; 3]],
    padding: u32,
) {
    let (_width, height) = image.dimensions();

    let first = image.get_pixel(column, 0).0;
    let last = image.get_pixel(column, height - 1).0;

    let mut sum = [0.0; 3];

    // 上側のパディング
    for b in &mut buffer[..padding as usize] {
        for i in 0..3 {
            sum[i] += first[i];
            b[i] = sum[i];
        }
    }

    // メイン部分
    unsafe {
        for y in 0..height {
            let pixel = image.unsafe_get_pixel(column, y).0;
            for i in 0..3 {
                sum[i] += pixel[i];
                buffer.get_unchecked_mut(y as usize + padding as usize)[i] = sum[i];
            }
        }
    }

    // 下側のパディング
    for b in &mut buffer[padding as usize + height as usize..] {
        for i in 0..3 {
            sum[i] += last[i];
            b[i] = sum[i];
        }
    }
}

/// Luma画像の指定行のrunning sumを計算する
fn compute_row_running_sum_luma(
    image: &Image<Luma<f32>>,
    row: u32,
    buffer: &mut [f32],
    padding: u32,
) {
    let (width, _height) = image.dimensions();
    let (width, padding) = (width as usize, padding as usize);

    let row_data = &(**image)[width * row as usize..][..width];
    let first = row_data[0];
    let last = row_data[width - 1];

    let mut sum = 0.0;

    // 左側のパディング
    for b in &mut buffer[..padding] {
        sum += first;
        *b = sum;
    }

    // メイン部分
    for (b, &pixel) in buffer[padding..].iter_mut().zip(row_data) {
        sum += pixel;
        *b = sum;
    }

    // 右側のパディング
    for b in &mut buffer[padding + width..] {
        sum += last;
        *b = sum;
    }
}

/// Luma画像の指定列のrunning sumを計算する
fn compute_column_running_sum_luma(
    image: &Image<Luma<f32>>,
    column: u32,
    buffer: &mut [f32],
    padding: u32,
) {
    let (_width, height) = image.dimensions();

    let first = image.get_pixel(column, 0)[0];
    let last = image.get_pixel(column, height - 1)[0];

    let mut sum = 0.0;

    // 上側のパディング
    for b in &mut buffer[..padding as usize] {
        sum += first;
        *b = sum;
    }

    // メイン部分
    unsafe {
        for y in 0..height {
            sum += image.unsafe_get_pixel(column, y)[0];
            *buffer.get_unchecked_mut(y as usize + padding as usize) = sum;
        }
    }

    // 下側のパディング
    for b in &mut buffer[padding as usize + height as usize..] {
        sum += last;
        *b = sum;
    }
}
