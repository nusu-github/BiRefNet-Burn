//! High-level inference runner and mask postprocessing for `BiRefNet` models.

mod error;
mod postprocessing;
mod segmenter;

#[doc(inline)]
pub use error::InferenceError;
#[doc(inline)]
pub use postprocessing::{
    PostprocessConfig, apply_threshold, fill_holes, gaussian_blur, mask_to_u8,
    morphological_closing, morphological_opening, postprocess_mask, remove_small_components,
    resize_tensor,
};
#[doc(inline)]
pub use segmenter::{SegmentationOutput, Segmenter, SegmenterConfig};
