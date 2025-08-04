//! Common utilities for BiRefNet examples.
//!
//! This module provides shared functionality used across
//! different example applications.

mod backend;
pub mod image;
mod postprocessing;
pub mod weights;

// Re-export commonly used items
pub use backend::{create_device, get_backend_name, SelectedBackend, SelectedDevice};
pub use image::ImageUtils;
pub use postprocessing::{
    apply_threshold, fill_holes, gaussian_blur, morphological_closing, morphological_opening,
    postprocess_mask, remove_small_components, resize_tensor, tensor_to_image_data,
};
