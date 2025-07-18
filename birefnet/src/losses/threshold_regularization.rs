//! Threshold regularization loss to push predictions towards 0 or 1.

use burn::{
    prelude::*,
    tensor::{backend::Backend, Tensor},
};

/// Configuration for Threshold Regularization Loss.
#[derive(Config, Debug)]
pub struct ThrRegLossConfig {
    #[config(default = 1.0)]
    pub weight: f32,
}

/// Threshold regularization loss to push predictions towards 0 or 1.
#[derive(Module, Debug)]
pub struct ThrRegLoss<B: Backend> {
    pub weight: f32,
    _phantom: std::marker::PhantomData<B>,
}

impl ThrRegLossConfig {
    /// Initialize a new threshold regularization loss with the given configuration.
    pub const fn init<B: Backend>(&self) -> ThrRegLoss<B> {
        ThrRegLoss {
            weight: self.weight,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Default for ThrRegLoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> ThrRegLoss<B> {
    /// Create a new threshold regularization loss with default configuration.
    pub fn new() -> Self {
        ThrRegLossConfig::new().init()
    }

    /// Calculate threshold regularization loss.
    ///
    /// # Arguments
    /// * `pred` - Predicted segmentation map with shape [N, C, H, W]
    /// * `_target` - Unused, kept for API consistency
    ///
    /// # Returns
    /// Threshold regularization loss tensor
    pub fn forward(&self, pred: Tensor<B, 4>, _target: Tensor<B, 4>) -> Tensor<B, 1> {
        // Loss = 1 - (pred^2 + (pred-1)^2)
        // Simplified: 1 - (pred^2 + pred^2 - 2*pred + 1) = 1 - (2*pred^2 - 2*pred + 1) = 2*pred - 2*pred^2 = 2*pred*(1-pred)
        // The original implementation is `torch.mean(1 - ((pred - 0) ** 2 + (pred - 1) ** 2))`
        // which simplifies to `torch.mean(2*pred - 2*pred**2)`
        // Let's stick to the original formula for clarity, but simplified.
        let pred_sq = pred.clone().powf_scalar(2.0);
        let pred_minus_one_sq = (pred - 1.0).powf_scalar(2.0);
        let reg = (pred_sq + pred_minus_one_sq).mean();
        (Tensor::ones_like(&reg) - reg) * self.weight
    }
}
