//! Loss functions for BiRefNet training.
//!
//! This module implements a hierarchical loss system with three levels:
//!
//! ## Level 1: Basic Components (combined.rs)
//! Basic segmentation loss combining BCE and IoU - used as building blocks.
//!
//! ## Level 2: Pixel Loss System (pixel.rs)
//! Multi-loss integration system that combines 10+ different loss functions
//! with task-specific configurations (DIS5K, Matting, General).
//!
//! ## Level 3: BiRefNet Training System (birefnet_loss.rs)
//! Complete training loss system integrating pixel loss, classification loss,
//! and guidance distillation loss for full BiRefNet training.
//!
//! The implementation follows the original PyTorch BiRefNet loss.py structure
//! and provides both individual loss functions and integrated loss systems.

pub mod bce;
pub mod birefnet_loss;
pub mod classification;
pub mod combined;
pub mod contour;
pub mod iou;
pub mod mae;
pub mod mse;
pub mod multiscale;
pub mod pixel;
pub mod ssim;
pub mod structure;
pub mod threshold_regularization;
pub mod triplet;

// Re-export individual loss functions and their configs
pub use bce::{BCELoss, BCELossConfig};
pub use birefnet_loss::{BiRefNetLoss, BiRefNetLossConfig};
pub use classification::{ClsLoss, ClsLossConfig};
pub use combined::{CombinedLoss, CombinedLossConfig};
pub use contour::{ContourLoss, ContourLossConfig};
pub use iou::{PatchIoULoss, PatchIoULossConfig};
pub use mae::{MaeLoss, MaeLossConfig};
pub use mse::{MseLoss, MseLossConfig};
pub use multiscale::{MultiScaleLoss, MultiScaleLossConfig};
pub use pixel::{LossWeightsConfig, PixLoss, PixLossConfig};
pub use ssim::{SSIMLoss, SSIMLossConfig};
pub use structure::{StructureLoss, StructureLossConfig};
pub use threshold_regularization::{ThrRegLoss, ThrRegLossConfig};
// Triplet loss is currently not used in the main export
// pub use triplet::{TripletLoss, TripletLossConfig};
