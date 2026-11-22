#![doc = include_str!("../README.md")]

pub mod distance;
mod error;
pub mod filters;
pub mod foreground;
#[cfg(feature = "image-io")]
pub mod image;
pub mod morphology;
pub mod normalize;

#[doc(inline)]
pub use error::UtilError;
