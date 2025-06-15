use crate::error::PaddingError;
use image::{imageops, ImageBuffer, Pixel};
use num_traits::AsPrimitive;

/// パディング位置を指定する列挙型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Position {
    /// 上部中央
    Top,
    /// 下部中央
    Bottom,
    /// 左側中央
    Left,
    /// 右側中央
    Right,
    /// 左上
    TopLeft,
    /// 右上
    TopRight,
    /// 左下
    BottomLeft,
    /// 右下
    BottomRight,
    /// 中央
    Center,
}

/// 画像サイズとパディングサイズから位置を計算する
///
/// # 引数
///
/// * `size` - 元画像のサイズ (幅, 高さ)
/// * `pad_size` - パディング後のサイズ (幅, 高さ)
/// * `position` - パディング位置
///
/// # 戻り値
///
/// 成功時は画像を配置する位置 (x, y) を返す
///
/// # エラー
///
/// * パディングサイズが元画像サイズより小さい場合
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

    Ok((x.as_(), y.as_()))
}

/// パディング操作を提供するトレイト
pub trait Padding<P: Pixel> {
    /// 指定したサイズと位置でパディングを追加する
    ///
    /// # 引数
    ///
    /// * `pad_size` - パディング後のサイズ (幅, 高さ)
    /// * `position` - パディング位置
    /// * `color` - パディング色
    ///
    /// # 戻り値
    ///
    /// パディング済み画像と元画像の配置位置のタプル
    fn add_padding(
        self,
        pad_size: (u32, u32),
        position: Position,
        color: P,
    ) -> Result<(ImageBuffer<P, Vec<P::Subpixel>>, (u32, u32)), PaddingError>;

    /// 正方形になるようにパディングを追加する
    ///
    /// # 引数
    ///
    /// * `color` - パディング色
    ///
    /// # 戻り値
    ///
    /// パディング済み画像と元画像の配置位置のタプル
    fn add_padding_square(
        self,
        color: P,
    ) -> Result<(ImageBuffer<P, Vec<P::Subpixel>>, (u32, u32)), PaddingError>;

    /// パディング位置を計算する
    fn calculate_padding_position(
        &self,
        pad_size: (u32, u32),
        position: Position,
    ) -> Result<(i64, i64), PaddingError>;

    /// 正方形パディングの位置とサイズを計算する
    fn calculate_square_padding(&self) -> Result<((i64, i64), (u32, u32)), PaddingError>;
}

impl<P: Pixel> Padding<P> for ImageBuffer<P, Vec<P::Subpixel>> {
    fn add_padding(
        self,
        pad_size: (u32, u32),
        position: Position,
        color: P,
    ) -> Result<(Self, (u32, u32)), PaddingError> {
        let (x, y) = self.calculate_padding_position(pad_size, position)?;
        let (pad_width, pad_height) = pad_size;
        let mut canvas = Self::from_pixel(pad_width, pad_height, color);
        imageops::overlay(&mut canvas, &self, x, y);
        Ok((canvas, (x as u32, y as u32)))
    }

    fn add_padding_square(self, color: P) -> Result<(Self, (u32, u32)), PaddingError> {
        let (_, pad_size) = self.calculate_square_padding()?;
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
