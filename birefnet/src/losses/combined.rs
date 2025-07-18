//! Combined loss function for BiRefNet training.
//!
//! This combines BCE loss and IoU loss as used in the original implementation.

use burn::{
    prelude::*,
    tensor::{backend::Backend, Tensor},
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

/// Combined loss function for BiRefNet training.
///
/// This combines BCE loss and IoU loss as used in the original implementation.
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

        let max_val = pred.clone().clamp_max(0.0);
        let term1 = max_val - pred.clone() * target;
        let term2 = (-pred.abs()).exp().add_scalar(1.0).log();

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
