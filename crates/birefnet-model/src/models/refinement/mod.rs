//! Output refinement modules appended after the `BiRefNet` decoder.

mod ref_unet;
mod refiner;

pub use ref_unet::{RefUNet, RefUNetConfig};
pub use refiner::{Refiner, RefinerConfig, RefinerPVTInChannels4, RefinerPVTInChannels4Config};
