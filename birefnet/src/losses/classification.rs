//! Classification loss for auxiliary supervision.

use burn::{
    nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig},
    prelude::*,
    tensor::{backend::Backend, Int, Tensor},
};

/// Configuration for Classification Loss.
#[derive(Config, Debug)]
pub struct ClsLossConfig {
    #[config(default = 1.0)]
    pub weight: f32,
}

/// Classification loss for auxiliary supervision.
#[derive(Module, Debug)]
pub struct ClsLoss<B: Backend> {
    pub weight: f32,
    pub ce_loss: CrossEntropyLoss<B>,
}

impl ClsLossConfig {
    /// Initialize a new classification loss with the given configuration.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ClsLoss<B> {
        ClsLoss {
            weight: self.weight,
            ce_loss: CrossEntropyLossConfig::new().init(device),
        }
    }
}

// Note: No Default implementation for ClsLoss since device parameter is required

impl<B: Backend> ClsLoss<B> {
    /// Create a new classification loss with default configuration.
    pub fn new(device: &B::Device) -> Self {
        ClsLossConfig::new().init(device)
    }

    /// Calculate classification loss.
    ///
    /// # Arguments
    /// * `preds` - List of predicted class logits
    /// * `targets` - Ground truth class labels
    ///
    /// # Returns
    /// Classification loss tensor
    pub fn forward(&self, preds: Vec<Tensor<B, 2>>, targets: Tensor<B, 1, Int>) -> Tensor<B, 1> {
        if preds.is_empty() {
            return Tensor::zeros([1], &targets.device());
        }

        let mut total_loss = Tensor::zeros([1], &targets.device());

        for pred in preds.iter() {
            let loss = self.ce_loss.forward(pred.clone(), targets.clone());
            total_loss = total_loss + loss;
        }

        total_loss * self.weight / (preds.len() as f32)
    }
}
