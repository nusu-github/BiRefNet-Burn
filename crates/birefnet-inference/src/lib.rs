#![doc = include_str!("../README.md")]

mod error;
pub mod postprocess;
mod segmenter;

#[doc(inline)]
pub use error::InferenceError;
#[doc(inline)]
pub use segmenter::{SegmentationOutput, Segmenter, SegmenterConfig};
