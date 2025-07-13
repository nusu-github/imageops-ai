# imageops-kit

A Rust library for image processing operations and utilities.

[![Crates.io](https://img.shields.io/crates/v/imageops-kit.svg)](https://crates.io/crates/imageops-kit)
[![Documentation](https://docs.rs/imageops-kit/badge.svg)](https://docs.rs/imageops-kit)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE-APACHE)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE-MIT)

## Overview

This crate provides a collection of image processing operations and utilities.

### Key Features

- **Alpha Pre-multiplication**: Premultiplies alpha values across color channels
- **Alpha Mask Application**: Applies a grayscale mask to an RGB image to generate an RGBA image
- **Blur-Fusion Foreground Estimation**: Estimates the foreground using the Blur-Fusion algorithm
- **Boundary Clipping**: Automatically detects and clips to the minimum boundary
- **Padding**: Smart padding at various positions
- **NL-Means Denoising**: Noise reduction utilizing similarity between neighboring pixels
- **One-Sided Box Filter**: Edge-preserving smoothing filter for image denoising
- **INTER_AREA Resize**: High-quality image downscaling using OpenCV's INTER_AREA algorithm

## Usage Example

```rust
use imageops_kit::{PremultiplyAlphaAndDropExt, ApplyAlphaMaskExt, PaddingExt, Position, OneSidedBoxFilterExt, InterAreaResizeExt};
use imageproc::definitions::Image;
use image::{Rgb, Rgba, Luma};

# fn example() -> Result<(), Box<dyn std::error.Error>> {
// Premultiplied conversion from RGBA to RGB image
let rgba_image: Image<Rgba<u8>> = Image::new(100, 100);
let rgb_image = rgba_image.premultiply_alpha_and_drop()?;

// Apply alpha mask to RGB image
let rgb_image: Image<Rgb<u8>> = Image::new(100, 100);
let mask: Image<Luma<u8>> = Image::new(100, 100);
let rgba_result = rgb_image.apply_alpha_mask(&mask)?;

// One-Sided Box Filter for edge-preserving smoothing
let image: Image<Rgb<u8>> = Image::new(100, 100);
let smoothed = image.one_sided_box_filter(2, 5)?; // radius=2, iterations=5

// INTER_AREA resize for high-quality downscaling
let image: Image<Rgb<u8>> = Image::new(100, 100);
let resized = image.resize_area(50, 50)?;

// Image padding
let image: Image<Rgb<u8>> = Image::new(50, 50);
let (padded, position) = image.add_padding(
    (100, 100),
    Position::Center,
    Rgb([255, 255, 255])
)?;
# Ok(())
# }
```

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
imageops-kit = "0.1"
```

## Documentation

For detailed API specifications, see [docs.rs](https://docs.rs/imageops-kit).

## References

- **Blur-Fusion**: A. Germer, "Approximate Fast Foreground Colour Estimation," ICIP 2021.

## Contribution

We welcome bug reports and pull requests. Please contact us via
the [issue tracker](https://github.com/nusu-github/imageops-kit/issues) on GitHub.

## License

This project is published under the MIT or Apache-2.0 license.