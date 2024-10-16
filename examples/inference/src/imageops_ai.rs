use image::{ImageBuffer, Pixel};

pub use alpha_mask_applicable::AlphaMaskImage;
pub use box_filter::BoxFilter;
pub use convert_color::ForegroundEstimator;

mod alpha_mask_applicable;
mod box_filter;
mod convert_color;

pub type Image<P> = ImageBuffer<P, Vec<<P as Pixel>::Subpixel>>;
