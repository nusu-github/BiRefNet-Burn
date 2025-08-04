//! Contour loss for boundary refinement.

use burn::{
    prelude::*,
    tensor::{backend::Backend, Tensor},
};

/// Configuration for Contour Loss function.
#[derive(Config, Debug)]
pub struct ContourLossConfig {
    #[config(default = 10.0)]
    pub length_weight: f32,
    #[config(default = 1e-8)]
    pub epsilon: f32,
}

/// Contour loss for boundary refinement.
#[derive(Module, Debug)]
pub struct ContourLoss<B: Backend> {
    pub length_weight: f32,
    pub epsilon: f32,
    _phantom: std::marker::PhantomData<B>,
}

impl ContourLossConfig {
    /// Initialize a new contour loss function with the given configuration.
    pub const fn init<B: Backend>(&self) -> ContourLoss<B> {
        ContourLoss {
            length_weight: self.length_weight,
            epsilon: self.epsilon,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Default for ContourLoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> ContourLoss<B> {
    /// Create a new contour loss function with default configuration.
    pub fn new() -> Self {
        ContourLossConfig::new().init()
    }

    /// Create a new contour loss function with custom weight.
    pub fn with_weight(length_weight: f32) -> Self {
        ContourLossConfig::new()
            .with_length_weight(length_weight)
            .init()
    }

    /// Calculate contour loss.
    ///
    /// # Arguments
    /// * `pred` - Predicted segmentation map with shape [N, C, H, W]
    /// * `target` - Ground truth segmentation map with shape [N, C, H, W]
    ///
    /// # Returns
    /// Contour loss tensor
    pub fn forward(&self, pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        // The original PyTorch implementation computes loss on probabilities
        let pred_prob = burn::tensor::activation::sigmoid(pred);

        // length term - replicating the specific slicing from the original PyTorch code
        // delta_r = pred[:,:,1:,:] - pred[:,:,:-1,:]
        let delta_r = pred_prob.clone().slice(s![.., .., 1.., ..])
            - pred_prob.clone().slice(s![.., .., 0..-1, ..]);

        // delta_c = pred[:,:,:,1:] - pred[:,:,:,:-1]
        let delta_c = pred_prob.clone().slice(s![.., .., .., 1..])
            - pred_prob.clone().slice(s![.., .., .., 0..-1]);

        // These specific slices require the input to be at least 3x3.
        // Panics will occur on smaller inputs, which matches the original PyTorch behavior.
        // delta_r    = delta_r[:,:,1:,:-2]**2
        let delta_r_sq = delta_r.slice(s![.., .., 1.., 0..-2]).powf_scalar(2.0);

        // delta_c    = delta_c[:,:,:-2,1:]**2
        let delta_c_sq = delta_c.slice(s![.., .., ..-2, 1..]).powf_scalar(2.0);

        let delta_pred = (delta_r_sq + delta_c_sq).abs();
        let length = (delta_pred + self.epsilon).sqrt().mean();

        // Region terms
        let c_in = Tensor::ones_like(&pred_prob);
        let c_out = Tensor::zeros_like(&pred_prob);

        // region_in = mean(pred * (target - c_in)^2)
        let region_in = (pred_prob.clone() * (target.clone() - c_in).powf_scalar(2.0)).mean();

        // region_out = mean((1-pred) * (target - c_out)^2)
        let region_out = ((Tensor::ones_like(&pred_prob) - pred_prob)
            * (target - c_out).powf_scalar(2.0))
        .mean();

        let region = region_in + region_out;

        // Total loss = weight * length + region
        length * self.length_weight + region
    }
}
