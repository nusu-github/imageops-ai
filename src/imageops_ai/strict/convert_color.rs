use crate::Image;
use image::{ImageBuffer, Luma, LumaA, Pixel, Primitive, Rgb, Rgba};
use num_bigint::BigInt;
use num_rational::Ratio;

// ---- ヘルパートレイトと型ごとの実装 ----
// このモジュールは、型の種類（整数か浮動小数点数か）によって
// 計算ロジックを切り替えるための内部的な仕組みです。
mod internal {
    use super::{BigInt, Primitive, Ratio};
    use std::convert::TryFrom;

    /// 型ごとにアルファ合成の具体的な計算を定義するためのヘルパートレイト
    pub trait MargeAlphaImpl: Primitive {
        fn marge(value: Self, alpha: Self) -> Self;
    }

    /// 整数型のための計算ロジックを実装するマクロ
    macro_rules! impl_marge_integer {
        ($($t:ty),*) => {
            $(
                impl MargeAlphaImpl for $t {
                    fn marge(value: Self, alpha: Self) -> Self {
                        // 計算途中のオーバーフローを防ぐため、任意精度の整数 `BigInt` に変換する
                        let max_value = Ratio::from_integer(BigInt::from(Self::DEFAULT_MAX_VALUE));
                        let value_r = Ratio::from_integer(BigInt::from(value));
                        let alpha_r = Ratio::from_integer(BigInt::from(alpha));

                        // 厳密な有理数として計算を実行
                        let alpha_normalized = alpha_r / max_value;
                        let new_value_r = value_r * alpha_normalized;

                        // 計算結果の有理数を整数に戻す
                        let final_value_bigint = new_value_r.to_integer();
                        // BigIntから元の型 `T` に変換する。
                        // 計算結果は必ず元の型の範囲内に収まるはずなので unwrap する
                        Self::try_from(final_value_bigint).unwrap_or_else(|_| panic!("MargeAlpha conversion failed"))
                    }
                }
            )*
        };
    }

    /// 浮動小数点数のための計算ロジックを実装するマクロ
    macro_rules! impl_marge_float {
        ($($t:ty),*) => {
            $(
                impl MargeAlphaImpl for $t {
                    fn marge(value: Self, alpha: Self) -> Self {
                        // f32/f64の場合、DEFAULT_MAX_VALUE は 1.0 なので、
                        // value * (alpha / 1.0) は value * alpha と等価
                        value * alpha
                    }
                }
            )*
        };
    }

    // 提供された Primitive 型すべてに対して、適切な計算ロジックを実装する
    impl_marge_integer!(u8, u16, u32, u64, usize, i8, i16, i32, i64, isize);
    impl_marge_float!(f32, f64);
}

/// Alphaチャンネルを輝度値に合成するトレイト
pub trait MargeAlpha {
    type Output;
    fn marge_alpha(&self) -> Self::Output;
}

impl<S> MargeAlpha for Image<LumaA<S>>
where
    S: Primitive + 'static + internal::MargeAlphaImpl,
    LumaA<S>: Pixel<Subpixel = S>,
    Luma<S>: Pixel<Subpixel = S>,
{
    type Output = Image<Luma<S>>;

    /// 有理数演算(または浮動小数点数演算)を用いて、厳密にアルファブレンディングを行う
    fn marge_alpha(&self) -> Self::Output {
        ImageBuffer::from_fn(self.width(), self.height(), |x, y| {
            let LumaA([luminance, alpha]) = *self.get_pixel(x, y);
            // 型(S)に応じて適切な `marge` 関数が呼び出される
            let final_luminance = internal::MargeAlphaImpl::marge(luminance, alpha);
            Luma([final_luminance])
        })
    }
}

impl<S> MargeAlpha for Image<Rgba<S>>
where
    S: Primitive + 'static + internal::MargeAlphaImpl,
    Rgba<S>: Pixel<Subpixel = S>,
    Rgb<S>: Pixel<Subpixel = S>,
{
    type Output = Image<Rgb<S>>;

    /// 有理数演算(または浮動小数点数演算)を用いて、厳密にアルファブレンディングを行う
    fn marge_alpha(&self) -> Self::Output {
        ImageBuffer::from_fn(self.width(), self.height(), |x, y| {
            let Rgba([red, green, blue, alpha]) = *self.get_pixel(x, y);
            // 型(S)に応じて適切な `marge` 関数が呼び出される
            let final_red = internal::MargeAlphaImpl::marge(red, alpha);
            let final_green = internal::MargeAlphaImpl::marge(green, alpha);
            let final_blue = internal::MargeAlphaImpl::marge(blue, alpha);
            Rgb([final_red, final_green, final_blue])
        })
    }
}
