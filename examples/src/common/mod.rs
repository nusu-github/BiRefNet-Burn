//! Common utilities for BiRefNet examples.
//!
//! This module provides shared functionality used across
//! different example applications.

pub mod postprocessing;

// Re-export commonly used items
pub use postprocessing::{
    apply_threshold, fill_holes, gaussian_blur, morphological_closing, morphological_opening,
    postprocess_mask, remove_small_components, resize_tensor, tensor_to_image_data,
};
