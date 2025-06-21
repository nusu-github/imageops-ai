# imageops-ai

AI 画像処理操作のための Rust ライブラリ

[![Crates.io](https://img.shields.io/crates/v/imageops-ai.svg)](https://crates.io/crates/imageops-ai)
[![Documentation](https://docs.rs/imageops-ai/badge.svg)](https://docs.rs/imageops-ai)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

## 概要

このクレートは、高度な画像処理タスクのための専門的な操作を提供します。

### 主な機能

- **アルファ前乗算**: カラーチャンネルにアルファ値を前乗算
- **アルファマスク適用**: RGB 画像にグレースケールマスクを適用して RGBA 画像を生成
- **前景色推定**: Blur-Fusion アルゴリズムを使用した前景色推定
- **境界クリッピング**: 最小境界の自動検出とクリッピング
- **パディング**: 様々な位置でのスマートパディング
- **NL‑Means ノイズ除去**: 近傍画素の類似度を利用したノイズ低減

## 使用例

```rust
use imageops_ai::{AlphaPremultiply, ApplyAlphaMask, Position, Padding, Image};
use image::{Rgb, Rgba, Luma};

// RGBA画像からRGB画像への前乗算変換
let rgba_image: Image<Rgba<u8>> = Image::new(100, 100);
let rgb_image = rgba_image.premultiply_alpha()?;

// RGB画像にアルファマスクを適用
let rgb_image: Image<Rgb<u8>> = Image::new(100, 100);
let mask: Image<Luma<u8>> = Image::new(100, 100);
let rgba_result = rgb_image.apply_alpha_mask(&mask)?;

// 画像パディング
let image: Image<Rgb<u8>> = Image::new(50, 50);
let (padded, _position) = image.add_padding(
    (100, 100),
    Position::Center,
    Rgb([255, 255, 255])
)?;
```

## インストール

`Cargo.toml`に以下を追加してください：

```toml
[dependencies]
imageops-ai = "0.1"
```

## ドキュメント

詳しい API 仕様は [docs.rs](https://docs.rs/imageops-ai) を参照してください。

## 参考文献

- **Blur-Fusion**: A. Germer, "Approximate Fast Foreground Colour Estimation," ICIP 2021.

## 貢献

バグ報告やプルリクエストを歓迎します。GitHub の [issue トラッカー](https://github.com/nusu-github/imageops-ai/issues) でご連絡ください。

## ライセンス

このプロジェクトは MIT または Apache-2.0 ライセンスの下で公開されています。
