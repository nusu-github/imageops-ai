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
//! - **Box Filtering**: High-performance box filtering with integral image and OP-SAT algorithms
//! - **One-Sided Box Filter**: Edge-preserving smoothing filter for image denoising
//!
//! ## Example Usage
//!
//! ```no_run
//! use imageops_ai::{AlphaPremultiply, ApplyAlphaMask, Position, Padding, BoxFilterExt, OSBFilterExt};
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
//! // Box filtering with method chaining
//! let image: Image<Rgb<u8>> = Image::new(100, 100);
//! let filtered = image.box_filter_integral(3)?;
//!
//! // One-Sided Box Filter for edge-preserving smoothing
//! let image: Image<Rgb<u8>> = Image::new(100, 100);
//! let smoothed = image.osbf(2, 5)?; // radius=2, iterations=5
//!
//! // Image padding
//! let image: Image<Rgb<u8>> = Image::new(50, 50);
//! let (padded, position) = image.add_padding(
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
mod imageops_ai;
mod utils;

#[cfg(test)]
mod test_utils;

pub use error::{
    AlphaMaskError, BoxFilterError, ClipBorderError, ConvertColorError, GuidedFilterError,
    NLMeansError, OSBFilterError, PaddingError,
};
pub use imageops_ai::alpha_premultiply::{AlphaPremultiply, PremultiplyAlphaInPlace};
pub use imageops_ai::apply_alpha_mask::{ApplyAlphaMask, ModifyAlpha};
pub use imageops_ai::blur_fusion::{estimate_foreground, ForegroundEstimator};
pub use imageops_ai::box_filter::{BoxFilter, BoxFilterExt, BoxFilterIntegral, BoxFilterOPSAT};
pub use imageops_ai::clip_minimum_border::ClipMinimumBorder;
pub use imageops_ai::guided_filter::{
    FastGuidedFilterImpl, GuidedFilterColor, GuidedFilterExtension, GuidedFilterGray,
    GuidedFilterWithColorGuidance,
};
pub use imageops_ai::nlmeans::NLMeans;
pub use imageops_ai::osbf::{OSBFilter, OSBFilterExt, OneSidedBoxFilter};
pub use imageops_ai::padding::{add_padding, Padding, Position};
pub use utils::{unify_gray_images, unify_rgb_images, LargerType, NormalizedFrom};

// Re-export imageproc::definitions::Image for convenience
pub use imageproc::definitions::Image;
