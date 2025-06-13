mod error;
mod imageops_ai;

use image::{ImageBuffer, Pixel};

pub use error::{BoxFilterError, Error, PaddingError};
pub use imageops_ai::alpha_mask_applicable::ApplyAlphaMask;
pub use imageops_ai::box_filter::BoxFilter;
pub use imageops_ai::clip_minimum_border::ClipMinimumBorder;
pub use imageops_ai::convert_color::{ForegroundEstimator, MargeAlpha};
pub use imageops_ai::padding::{Padding, Position};
pub use imageops_ai::summed_area_table::{CreateSummedAreaTable, SummedAreaTable, SummedAreaTableResult};

#[cfg(feature = "strict")]
pub use imageops_ai::strict::box_filter::SatBoxFilter;
#[cfg(feature = "strict")]
pub use imageops_ai::strict::convert_color::MargeAlpha as StrictMargeAlpha;

pub type Image<P> = ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>;
