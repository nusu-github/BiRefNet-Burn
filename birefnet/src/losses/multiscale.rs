//! Multi-scale loss function for hierarchical supervision.
//!
//! This applies the combined loss at multiple scales as used in the original implementation.

use burn::{
    prelude::*,
    tensor::{backend::Backend, Tensor},
};

use super::combined::{CombinedLoss, CombinedLossConfig};

/// Configuration for Multi-scale Loss function.
#[derive(Config, Debug)]
pub struct MultiScaleLossConfig {
    #[config(default = 1.0)]
    pub bce_weight: f32,
    #[config(default = 1.0)]
    pub iou_weight: f32,
    #[config(default = "vec![1.0, 0.8, 0.6, 0.4]")]
    pub scale_weights: Vec<f32>,
    #[config(default = 1e-6)]
    pub epsilon: f32,
}

/// Multi-scale loss function for hierarchical supervision.
///
/// This applies the combined loss at multiple scales as used in the original implementation.
#[derive(Module, Debug)]
pub struct MultiScaleLoss<B: Backend> {
    pub base_loss: CombinedLoss<B>,
    pub scale_weights: Vec<f32>,
}

impl MultiScaleLossConfig {
    /// Initialize a new multi-scale loss function with the given configuration.
    pub fn init<B: Backend>(&self) -> MultiScaleLoss<B> {
        MultiScaleLoss {
            base_loss: CombinedLossConfig::new()
                .with_bce_weight(self.bce_weight)
                .with_iou_weight(self.iou_weight)
                .with_epsilon(self.epsilon)
                .init(),
            scale_weights: self.scale_weights.clone(),
        }
    }
}

impl<B: Backend> Default for MultiScaleLoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> MultiScaleLoss<B> {
    /// Create a new multi-scale loss function with default configuration.
    pub fn new() -> Self {
        MultiScaleLossConfig::new().init()
    }

    /// Create a new multi-scale loss function with custom weights.
    pub fn with_weights(bce_weight: f32, iou_weight: f32, scale_weights: Vec<f32>) -> Self {
        MultiScaleLossConfig::new()
            .with_bce_weight(bce_weight)
            .with_iou_weight(iou_weight)
            .with_scale_weights(scale_weights)
            .init()
    }

    /// Calculate multi-scale loss.
    ///
    /// # Arguments
    /// * `preds` - List of predicted segmentation maps at different scales
    /// * `targets` - List of ground truth segmentation maps at different scales
    ///
    /// # Returns
    /// Combined multi-scale loss tensor
    pub fn forward(&self, preds: Vec<Tensor<B, 4>>, targets: Vec<Tensor<B, 4>>) -> Tensor<B, 1> {
        assert_eq!(
            preds.len(),
            targets.len(),
            "Number of predictions and targets must be equal."
        );
        assert_eq!(
            preds.len(),
            self.scale_weights.len(),
            "Number of predictions and scale_weights must be equal."
        );

        let device = preds[0].device();

        let total_loss = preds
            .into_iter()
            .zip(targets)
            .zip(self.scale_weights.iter())
            .fold(
                None,
                |acc: Option<Tensor<B, 1>>, ((pred, target), &weight)| {
                    let scale_loss = self.base_loss.forward(pred, target);
                    let weighted_loss = scale_loss * weight;
                    match acc {
                        Some(total) => Some(total + weighted_loss),
                        None => Some(weighted_loss),
                    }
                },
            );

        total_loss.unwrap_or_else(|| Tensor::zeros([1], &device))
    }
}
