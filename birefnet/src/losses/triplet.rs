//! Triplet loss implementation for embedding learning.
//!
//! This implements a simplified version of triplet loss that can be used
//! for encouraging better feature representations in segmentation tasks.

use burn::{
    prelude::*,
    tensor::{backend::Backend, Tensor},
};

/// Configuration for Triplet Loss function.
#[derive(Config, Debug)]
pub struct TripletLossConfig {
    #[config(default = 1.0)]
    pub margin: f32,
    #[config(default = 1.0)]
    pub weight: f32,
    #[config(default = 1e-6)]
    pub epsilon: f32,
}

/// Triplet loss for encouraging better feature representations.
///
/// This implements a simplified triplet loss that compares positive and negative
/// regions based on the ground truth segmentation mask.
#[derive(Module, Debug)]
pub struct TripletLoss<B: Backend> {
    pub margin: f32,
    pub weight: f32,
    pub epsilon: f32,
    _phantom: std::marker::PhantomData<B>,
}

impl TripletLossConfig {
    /// Initialize a new triplet loss function with the given configuration.
    pub const fn init<B: Backend>(&self) -> TripletLoss<B> {
        TripletLoss {
            margin: self.margin,
            weight: self.weight,
            epsilon: self.epsilon,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Default for TripletLoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> TripletLoss<B> {
    /// Create a new triplet loss function with default configuration.
    pub fn new() -> Self {
        TripletLossConfig::new().init()
    }

    /// Create a new triplet loss function with custom parameters.
    pub fn with_params(margin: f32, weight: f32) -> Self {
        TripletLossConfig::new()
            .with_margin(margin)
            .with_weight(weight)
            .init()
    }

    /// Calculate triplet loss.
    ///
    /// This implementation uses a simplified approach where:
    /// - Anchor: predicted values where ground truth is positive
    /// - Positive: high confidence positive predictions
    /// - Negative: high confidence negative predictions
    ///
    /// # Arguments
    /// * `pred` - Predicted segmentation map with shape [N, C, H, W]
    /// * `target` - Ground truth segmentation map with shape [N, C, H, W]
    ///
    /// # Returns
    /// Triplet loss tensor
    pub fn forward(&self, pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        // Apply sigmoid to get probabilities
        let pred_sigmoid = burn::tensor::activation::sigmoid(pred);

        // Create masks for positive and negative regions
        let positive_mask = target.clone().greater_elem(0.5);
        let negative_mask = target.lower_elem(0.5);

        // Calculate mean predictions for positive and negative regions
        let positive_preds = pred_sigmoid.clone() * positive_mask.clone().float();
        let negative_preds = pred_sigmoid * negative_mask.clone().float();

        // Calculate anchor (average of positive predictions)
        let positive_sum = positive_preds.clone().sum();
        let positive_count = positive_mask.float().sum() + self.epsilon;
        let anchor = positive_sum / positive_count.clone();

        // Calculate positive example (high confidence positive predictions)
        let positive_avg = positive_preds.sum() / positive_count;

        // Calculate negative example (high confidence negative predictions)
        let negative_sum = negative_preds.sum();
        let negative_count = negative_mask.float().sum() + self.epsilon;
        let negative_avg = negative_sum / negative_count;

        // Calculate distances
        let pos_distance = (anchor.clone() - positive_avg).powf_scalar(2.0);
        let neg_distance = (anchor - negative_avg).powf_scalar(2.0);

        // Triplet loss: max(0, margin + pos_distance - neg_distance)
        let triplet_loss = (pos_distance - neg_distance + self.margin).clamp_min(0.0);

        triplet_loss.mean() * self.weight
    }
}
