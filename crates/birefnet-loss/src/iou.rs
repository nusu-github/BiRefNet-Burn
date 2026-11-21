//! Intersection over Union (IoU) loss for segmentation tasks.

use burn::{
    module::{Content, DisplaySettings, ModuleDisplay},
    nn::loss::Reduction,
    prelude::*,
};

use crate::ops::{assert_same_shape, reduce_loss, usize_to_i32};

/// Configuration for [`IoULoss`].
#[derive(Config, Debug, PartialEq)]
pub struct IoULossConfig {
    /// Small epsilon value to avoid division by zero.
    #[config(default = 1e-8)]
    pub eps: f64,
}

impl IoULossConfig {
    /// Initializes an [`IoULoss`] module.
    ///
    /// # Panics
    ///
    /// Panics if `eps` is not finite and strictly positive.
    #[must_use]
    pub fn init(&self) -> IoULoss {
        assert!(
            self.eps.is_finite() && self.eps > 0.0,
            "IoULoss eps must be finite and positive, got {}",
            self.eps
        );
        IoULoss { eps: self.eps }
    }
}

/// Differentiable soft Intersection over Union loss computed as `1 - intersection / (union + eps)`.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct IoULoss {
    eps: f64,
}

impl Default for IoULoss {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleDisplay for IoULoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add("eps", &self.eps).optional()
    }
}

impl IoULoss {
    /// Creates an [`IoULoss`] with default configuration (`eps = 1e-8`).
    #[must_use]
    pub fn new() -> Self {
        IoULossConfig::new().init()
    }

    /// Computes reduced IoU loss over probabilities `predictions` and targets in `[0, 1]`.
    #[must_use]
    pub fn forward(
        &self,
        predictions: Tensor<4>,
        targets: Tensor<4>,
        reduction: Reduction,
    ) -> Tensor<1> {
        let loss = self.forward_no_reduction(predictions, targets);
        reduce_loss(loss, reduction)
    }

    /// Computes per-sample IoU loss `[batch_size]` over probabilities in `[0, 1]`.
    ///
    /// # Panics
    ///
    /// Panics if `predictions` and `targets` have different shapes.
    #[must_use]
    pub fn forward_no_reduction(&self, predictions: Tensor<4>, targets: Tensor<4>) -> Tensor<1> {
        assert_same_shape(&predictions, &targets);

        let batch = usize_to_i32(predictions.dims()[0]);
        let pred_flat = predictions.reshape([batch, -1]);
        let target_flat = targets.reshape([batch, -1]);

        let intersection = (pred_flat.clone() * target_flat.clone()).sum_dim(1);
        let union = pred_flat.sum_dim(1) + target_flat.sum_dim(1) - intersection.clone();
        let iou = intersection / (union + self.eps);

        (iou.ones_like() - iou).squeeze_dim::<1>(1)
    }
}
