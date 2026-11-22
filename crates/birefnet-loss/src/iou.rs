//! Intersection over Union (IoU) loss for segmentation tasks.

use burn::{
    module::{Content, DisplaySettings, ModuleDisplay},
    nn::loss::Reduction,
    prelude::*,
};

use crate::ops::{assert_same_shape, flatten_per_sample, reduce_loss};

/// Configuration for [`IouLoss`].
#[derive(Config, Debug, PartialEq)]
pub struct IouLossConfig {
    /// Smoothing term added to both intersection and union.
    #[config(default = 1e-8)]
    pub eps: f64,
}

impl IouLossConfig {
    /// Initializes an [`IouLoss`] module.
    ///
    /// # Panics
    ///
    /// Panics if `eps` is not finite and strictly positive.
    #[must_use]
    pub fn init(&self) -> IouLoss {
        assert!(
            self.eps.is_finite() && self.eps > 0.0,
            "IouLoss eps must be finite and positive, got {}",
            self.eps
        );
        IouLoss { eps: self.eps }
    }
}

/// Differentiable soft IoU loss `1 - (intersection + eps) / (union + eps)`.
///
/// The symmetric smoothing makes two empty masks a perfect match (loss `0`).
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct IouLoss {
    eps: f64,
}

impl Default for IouLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleDisplay for IouLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add("eps", &self.eps).optional()
    }
}

impl IouLoss {
    /// Creates an [`IouLoss`] with default configuration (`eps = 1e-8`).
    #[must_use]
    pub fn new() -> Self {
        IouLossConfig::new().init()
    }

    /// Computes reduced IoU loss over probabilities `predictions` and targets in `[0, 1]`.
    ///
    /// # Panics
    ///
    /// Panics if `predictions` and `targets` have different shapes.
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

        let pred_flat = flatten_per_sample(predictions);
        let target_flat = flatten_per_sample(targets);

        let intersection = (pred_flat.clone() * target_flat.clone()).sum_dim(1);
        let union = pred_flat.sum_dim(1) + target_flat.sum_dim(1) - intersection.clone();
        let iou = intersection.add_scalar(self.eps) / union.add_scalar(self.eps);

        iou.neg().add_scalar(1.0).squeeze_dim::<1>(1)
    }
}
