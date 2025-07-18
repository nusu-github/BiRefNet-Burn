//! Loss functions for BiRefNet training.
//!
//! This module implements the loss functions used in BiRefNet training,
//! including Binary Cross-Entropy (BCE) loss and Intersection over Union (IoU) loss.
//!
//! The implementation follows the original PyTorch BiRefNet loss.py structure.

pub mod combined;
pub mod contour;
pub mod iou;
pub mod mae;
pub mod mse;
pub mod multiscale;
pub mod ssim;
pub mod structure;

// Re-export loss functions and their configs
pub use combined::{CombinedLoss, CombinedLossConfig};
pub use contour::{ContourLoss, ContourLossConfig};
pub use iou::{PatchIoULoss, PatchIoULossConfig};
pub use mae::{MaeLoss, MaeLossConfig};
pub use mse::{MseLoss, MseLossConfig};
pub use multiscale::{MultiScaleLoss, MultiScaleLossConfig};
pub use ssim::{SSIMLoss, SSIMLossConfig};
pub use structure::{StructureLoss, StructureLossConfig};

// Additional loss functions from the original file
pub mod classification;
pub mod threshold_regularization;

pub use classification::{ClsLoss, ClsLossConfig};
pub use threshold_regularization::{ThrRegLoss, ThrRegLossConfig};
