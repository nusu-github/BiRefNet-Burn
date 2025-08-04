//! Basic segmentation loss function (Level 1).
//!
//! This module provides the fundamental building block combining BCE loss and IoU loss
//! as used in the original BiRefNet implementation. This represents Level 1 in the
//! hierarchical loss architecture:
//!
//! - **Level 1 (This module)**: Basic BCE + IoU segmentation loss
//! - **Level 2 (pixel.rs)**: Multi-loss integration system using this as a component  
//! - **Level 3 (birefnet_loss.rs)**: Complete training system including classification and GDT losses
//!
//! ## Usage
//!
//! Use this directly for simple binary segmentation tasks:
//! ```rust
//! let loss = CombinedLoss::new();
//! let result = loss.forward(pred, target);
//! ```
//!
//! For more complex scenarios, use PixLoss (Level 2) or BiRefNetLoss (Level 3).

use burn::{
    prelude::*,
    tensor::{backend::Backend, ElementConversion, Tensor},
};

/// Configuration for Combined Loss function.
#[derive(Config, Debug)]
pub struct CombinedLossConfig {
    #[config(default = 1.0)]
    pub bce_weight: f32,
    #[config(default = 1.0)]
    pub iou_weight: f32,
    #[config(default = 1e-6)]
    pub epsilon: f32,
}

/// Basic segmentation loss combining BCE and IoU (Level 1).
///
/// This provides the fundamental building block loss function that combines
/// Binary Cross-Entropy and Intersection over Union losses. It serves as a
/// core component for higher-level loss systems.
///
/// Use this directly for simple segmentation tasks, or as a component
/// within PixLoss (Level 2) or BiRefNetLoss (Level 3) systems.
#[derive(Module, Debug)]
pub struct CombinedLoss<B: Backend> {
    pub bce_weight: f32,
    pub iou_weight: f32,
    pub epsilon: f32,
    _phantom: std::marker::PhantomData<B>,
}

impl CombinedLossConfig {
    /// Initialize a new combined loss function with the given configuration.
    pub const fn init<B: Backend>(&self) -> CombinedLoss<B> {
        CombinedLoss {
            bce_weight: self.bce_weight,
            iou_weight: self.iou_weight,
            epsilon: self.epsilon,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Default for CombinedLoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> CombinedLoss<B> {
    /// Create a new combined loss function with default configuration.
    pub fn new() -> Self {
        CombinedLossConfig::new().init()
    }

    /// Create a new combined loss function with custom weights.
    pub fn with_weights(bce_weight: f32, iou_weight: f32) -> Self {
        CombinedLossConfig::new()
            .with_bce_weight(bce_weight)
            .with_iou_weight(iou_weight)
            .init()
    }

    /// Calculate the combined loss for binary segmentation.
    ///
    /// # Arguments
    /// * `pred` - Predicted segmentation map with shape [N, C, H, W]
    /// * `target` - Ground truth segmentation map with shape [N, C, H, W]
    ///
    /// # Returns
    /// Combined loss tensor
    pub fn forward(&self, pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        let bce_loss_value = self.bce_loss(pred.clone(), target.clone());
        let iou_loss_value = self.iou_loss(pred, target);

        bce_loss_value * self.bce_weight + iou_loss_value * self.iou_weight
    }

    /// Calculate Binary Cross-Entropy loss with logits.
    ///
    /// # Arguments
    /// * `pred` - Predicted logits with shape [N, C, H, W]
    /// * `target` - Ground truth binary masks with shape [N, C, H, W]
    ///
    /// # Returns
    /// BCE loss tensor
    pub fn bce_loss(&self, pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        // Manual implementation of binary cross entropy with logits
        // BCE = -[y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x))]
        // Using the numerically stable formulation: BCE = max(x, 0) - x*y + log(1 + exp(-abs(x)))

        let max_val = pred.clone().clamp_max(0.0.elem::<B::FloatElem>());
        let term1 = max_val - pred.clone() * target;
        let term2 = (-pred.abs())
            .exp()
            .add_scalar(1.0.elem::<B::FloatElem>())
            .log();

        (term1 + term2).mean()
    }

    /// Calculate IoU (Intersection over Union) loss.
    ///
    /// This implements the IoU loss as:
    /// IoU = (intersection + epsilon) / (union + epsilon)
    /// Loss = 1 - IoU
    ///
    /// # Arguments
    /// * `pred` - Predicted logits with shape [N, C, H, W]
    /// * `target` - Ground truth binary masks with shape [N, C, H, W]
    ///
    /// # Returns
    /// IoU loss tensor
    pub fn iou_loss(&self, pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        // Apply sigmoid to get probabilities
        let pred_sigmoid = burn::tensor::activation::sigmoid(pred);

        // Calculate intersection: sum of element-wise multiplication
        let intersection = (pred_sigmoid.clone() * target.clone()).sum();

        // Calculate union: sum of both tensors minus intersection
        let union = pred_sigmoid.sum() + target.sum() - intersection.clone();

        // Calculate IoU with epsilon for numerical stability
        let iou = (intersection + self.epsilon) / (union + self.epsilon);

        // Return IoU loss (1 - IoU)
        let one = Tensor::ones_like(&iou);
        one - iou
    }
}
