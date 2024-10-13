use image::{ImageBuffer, Pixel};

pub use alpha_mask_applicable::AlphaMaskImage;
pub use box_filter::BoxFilter;
pub use clip_minimum_border::ClipMinimumBorder;
pub use convert_color::{ForegroundEstimator, MargeAlpha};
pub use padding::Padding;

mod alpha_mask_applicable;
mod box_filter;
mod clip_minimum_border;
mod convert_color;
mod padding;

pub type Image<P> = ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>;
