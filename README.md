# imageops-ai

A Rust library for advanced image processing operations

[![Crates.io](https://img.shields.io/crates/v/imageops-ai.svg)](https://crates.io/crates/imageops-ai)
[![Documentation](https://docs.rs/imageops-ai/badge.svg)](https://docs.rs/imageops-ai)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE-APACHE)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)

## Overview

This crate provides specialized operations for advanced image processing tasks.

### Key Features

- **Alpha Pre-multiplication**: Pre-multiplies alpha values across color channels
- **Alpha Mask Application**: Applies a grayscale mask to an RGB image to generate an RGBA image
- **Foreground Estimation**: Estimates the foreground using the Blur-Fusion algorithm
- **Boundary Clipping**: Automatically detects and clips to the minimum boundary
- **Padding**: Smart padding at various positions
- **NL-Means Denoising**: Noise reduction utilizing similarity between neighboring pixels

## Usage Example

```rust
use imageops_ai::{AlphaPremultiply, ApplyAlphaMask, Image, Padding, Position};

// Converts an RGBA image to an RGB image with alpha pre-multiplication
let rgba_image: Image<Rgba<u8>> = Image::new(100, 100);
let rgb_image = rgba_image.premultiply_alpha()?;

// Applies an alpha mask to an RGB image
let rgb_image: Image<Rgb<u8>> = Image::new(100, 100);
let mask: Image<Luma<u8>> = Image::new(100, 100);
let rgba_result = rgb_image.apply_alpha_mask(&mask)?;

// Image padding
let image: Image<Rgb<u8>> = Image::new(50, 50);
let (padded, position) = image.add_padding(
    (100, 100),
    Position::Center,
    Rgb([255, 255, 255])
)?;
```

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
imageops-ai = "0.1"
```

## Documentation

For detailed API specifications, see [docs.rs](https://docs.rs/imageops-ai).

## References

- **Blur-Fusion**: A. Germer, "Approximate Fast Foreground Colour Estimation," ICIP 2021.

## Contribution

We welcome bug reports and pull requests. Please contact us via
the [issue tracker](https://github.com/nusu-github/imageops-ai/issues) on GitHub.

## License

This project is published under the MIT or Apache-2.0 license.
