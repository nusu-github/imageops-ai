//! # imageops-ai
//!
//! A Rust library for AI-powered image processing operations.
//!
//! This crate provides specialized operations for advanced image processing tasks:
//!
//! - **Alpha Premultiplication**: Premultiplies color channels with alpha values
//! - **Alpha Mask Application**: Applies grayscale masks to RGB images to generate RGBA images
//! - **Foreground Color Estimation**: Foreground color estimation using the Blur-Fusion algorithm
//! - **Boundary Clipping**: Automatic detection and clipping of minimum boundaries
//! - **Padding**: Smart padding at various positions
//!
//! ## Example Usage
//!
//! ```no_run
//! use imageops_ai::{AlphaPremultiply, ApplyAlphaMask, Position, Padding};
//! use imageproc::definitions::Image;
//! use image::{Rgb, Rgba, Luma};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Premultiplied conversion from RGBA to RGB image
//! let rgba_image: Image<Rgba<u8>> = Image::new(100, 100);
//! let rgb_image = rgba_image.premultiply_alpha()?;
//!
//! // Apply alpha mask to RGB image
//! let rgb_image: Image<Rgb<u8>> = Image::new(100, 100);
//! let mask: Image<Luma<u8>> = Image::new(100, 100);
//! let rgba_result = rgb_image.apply_alpha_mask(&mask)?;
//!
//! // Image padding
//! let image: Image<Rgb<u8>> = Image::new(50, 50);
//! let (padded, _position) = image.add_padding(
//!     (100, 100),
//!     Position::Center,
//!     Rgb([255, 255, 255])
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Features
//!
//! - `serde`: Enables serialization support (optional)

mod error;
pub mod imageops_ai;
mod utils;

#[cfg(test)]
mod test_utils;

pub use error::{AlphaMaskError, ClipBorderError, ConvertColorError, NLMeansError, PaddingError};
pub use imageops_ai::alpha_premultiply::AlphaPremultiply;
pub use imageops_ai::apply_alpha_mask::{ApplyAlphaMask, ApplyAlphaMaskConvert};
pub use imageops_ai::blur_fusion::{estimate_foreground, ForegroundEstimator};
pub use imageops_ai::clip_minimum_border::ClipMinimumBorder;
pub use imageops_ai::nlmeans::NLMeans;
pub use imageops_ai::padding::{add_padding, Padding, Position};

// Re-export imageproc::definitions::Image for convenience
pub use imageproc::definitions::Image;
