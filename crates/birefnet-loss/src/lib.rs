//! Segmentation loss functions and BiRefNet multi-scale training objectives.

mod birefnet_loss;
mod classification;
mod contour;
mod error;
mod iou;
mod mae;
mod ops;
mod patch_iou;
mod pixel;
mod ssim;
mod structure;
mod threshold_regularization;

#[doc(inline)]
pub use birefnet_loss::{BiRefNetLoss, BiRefNetLossConfig};
#[doc(inline)]
pub use classification::{ClassificationLoss, ClassificationLossConfig};
#[doc(inline)]
pub use contour::{ContourLoss, ContourLossConfig};
#[doc(inline)]
pub use error::LossError;
#[doc(inline)]
pub use iou::{IoULoss, IoULossConfig};
#[doc(inline)]
pub use mae::{MaeLoss, MaeLossConfig};
#[doc(inline)]
pub use patch_iou::{PatchIoULoss, PatchIoULossConfig};
#[doc(inline)]
pub use pixel::{LossWeightsConfig, PixLoss, PixLossConfig};
#[doc(inline)]
pub use ssim::{SSIMLoss, SSIMLossConfig};
#[doc(inline)]
pub use structure::{StructureLoss, StructureLossConfig};
#[doc(inline)]
pub use threshold_regularization::{
    ThresholdRegularizationLoss, ThresholdRegularizationLossConfig,
};
