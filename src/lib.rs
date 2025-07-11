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
//! - **One-Sided Box Filter**: Edge-preserving smoothing filter for image denoising
//! - **INTER_AREA Resize**: High-quality image downscaling using OpenCV's INTER_AREA algorithm
//!
//! ## Example Usage
//!
//! ```no_run
//! use imageops_ai::{PremultiplyAlphaAndDropExt, ApplyAlphaMaskExt, PaddingExt, Position, OneSidedBoxFilterExt, InterAreaResizeExt};
//! use imageproc::definitions::Image;
//! use image::{Rgb, Rgba, Luma};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Premultiplied conversion from RGBA to RGB image
//! let rgba_image: Image<Rgba<u8>> = Image::new(100, 100);
//! let rgb_image = rgba_image.premultiply_alpha_and_drop()?;
//!
//! // Apply alpha mask to RGB image
//! let rgb_image: Image<Rgb<u8>> = Image::new(100, 100);
//! let mask: Image<Luma<u8>> = Image::new(100, 100);
//! let rgba_result = rgb_image.apply_alpha_mask(&mask)?;
//!
//! // One-Sided Box Filter for edge-preserving smoothing
//! let image: Image<Rgb<u8>> = Image::new(100, 100);
//! let smoothed = image.one_sided_box_filter(2, 5)?; // radius=2, iterations=5
//!
//! // INTER_AREA resize for high-quality downscaling
//! let image: Image<Rgb<u8>> = Image::new(100, 100);
//! let resized = image.resize_area(50, 50)?;
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
    AlphaMaskError, BoxFilterError, ClipBorderError, ColorConversionError, GuidedFilterError,
    InterAreaError, NLMeansError, OSBFilterError, PaddingError,
};
pub use imageops_ai::alpha_premultiply::{PremultiplyAlphaAndDropExt, PremultiplyAlphaAndKeepExt};
pub use imageops_ai::apply_alpha_mask::{ApplyAlphaMaskExt, ModifyAlphaExt};
pub use imageops_ai::blur_fusion::{estimate_foreground_colors, ForegroundEstimationExt};
pub use imageops_ai::clip_minimum_border::ClipMinimumBorderExt;
pub use imageops_ai::inter_area::{InterAreaResize, InterAreaResizeExt, InterpolationWeight};
pub use imageops_ai::nlmeans::NLMeansExt;
pub use imageops_ai::osbf::{OneSidedBoxFilter, OneSidedBoxFilterApplicator, OneSidedBoxFilterExt};
pub use imageops_ai::padding::{add_padding, calculate_position, PaddingExt, Position};
pub use utils::{unify_gray_images, unify_rgb_images, LargerType, NormalizedFrom};

// Re-export imageproc::definitions::Image for convenience
pub use imageproc::definitions::Image;
