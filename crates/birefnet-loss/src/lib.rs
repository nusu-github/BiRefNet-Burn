//! Comprehensive loss functions for deep learning segmentation tasks.
//!
//! This crate provides a collection of loss functions optimized for image segmentation,
//! object detection, and related computer vision tasks using the Burn deep learning framework.
//! All loss functions are designed to be composable, efficient, and backend-agnostic.
//!
//! ## Core Loss Functions
//!
//! ### Geometric Losses
//! - **[`IoULoss`]**: Intersection over Union loss for shape-aware training
//! - **[`ContourLoss`]**: Boundary-aware loss emphasizing object contours
//! - **[`StructureLoss`]**: Edge-aware segmentation with adaptive weighting
//!
//! ### Pixel-wise Losses
//! - **[`MaeLoss`]**: Mean Absolute Error loss for robust pixel-level training
//! - **[`SSIMLoss`]**: Structural Similarity Index loss for perceptual quality
//! - **[`ThresholdRegularizationLoss`]**: Threshold regularization for binary segmentation
//!
//! ### Specialized Losses
//! - **[`PatchIoULoss`]**: Patch-based IoU for multi-scale evaluation
//! - **[`ClassificationLoss`]**: Classification loss for auxiliary supervision
//! - **[`PixLoss`]**: Multi-loss aggregation system with configurable weights
//!
//! ## Key Features
//!
//! - **Backend Agnostic**: Works with any Burn backend (CPU, GPU, etc.)
//! - **Configurable**: Extensive configuration options via the `Config` trait
//! - **Composable**: Individual losses can be combined for complex training scenarios
//! - **Efficient**: Optimized tensor operations with minimal memory overhead
//! - **Type Safe**: Compile-time dimension checking and tensor type safety
//!
//! ## Usage Example
//!
//! ```rust
//! use birefnet_loss::{IoULoss, IoULossConfig, SSIMLoss, SSIMLossConfig};
//! use burn::prelude::*;
//!
//! fn create_combined_loss<B: Backend>() -> (IoULoss, SSIMLoss) {
//!     let iou_loss = IoULossConfig::new().with_eps(1e-5).init();
//!
//!     let ssim_loss = SSIMLossConfig::new()
//!         .with_window_size(11)
//!         .with_sigma(1.5)
//!         .init();
//!
//!     (iou_loss, ssim_loss)
//! }
//! ```
//!
//! ## Loss Function Categories
//!
//! ### Individual Loss Functions
//! Each loss function implements both `forward` (with reduction) and `forward_no_reduction`
//! methods following Burn's standard patterns:
//!
//! - Shape-preserving computation in `forward_no_reduction`
//! - Configurable reduction (Mean, Sum, Auto) in `forward`
//! - Comprehensive input validation and error handling
//!
//! ### Multi-Loss Systems
//! The [`PixLoss`] system provides a framework for combining multiple loss functions
//! with configurable weights, supporting complex training scenarios common in
//! segmentation tasks.
//!
//! All loss functions are thoroughly tested and provide detailed documentation
//! for their mathematical formulations and expected tensor shapes.

mod classification;
mod contour;
mod iou;
mod mae;
mod patch_iou;
mod pixel;
mod ssim;
mod structure;
mod threshold_regularization;
// Integrated loss system
mod birefnet_loss;

// Re-export core loss functions and configurations
pub use birefnet_loss::{BiRefNetLoss, BiRefNetLossConfig};
pub use classification::{ClassificationLoss, ClassificationLossConfig};
pub use contour::{ContourLoss, ContourLossConfig};
pub use iou::{IoULoss, IoULossConfig};
pub use mae::{MaeLoss, MaeLossConfig};
pub use patch_iou::{PatchIoULoss, PatchIoULossConfig};
pub use pixel::{LossWeightsConfig, PixLoss, PixLossConfig};
pub use ssim::{SSIMLoss, SSIMLossConfig};
pub use structure::{StructureLoss, StructureLossConfig};
pub use threshold_regularization::{
    ThresholdRegularizationLoss, ThresholdRegularizationLossConfig,
};

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    pub type TestBackend = NdArray;
}
