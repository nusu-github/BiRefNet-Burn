//! Boundary-aware contour loss combining curve length and internal/external region terms.

use burn::{
    module::{Content, DisplaySettings, ModuleDisplay},
    nn::loss::Reduction,
    prelude::*,
    tensor::s,
};

use crate::ops::{assert_same_shape, reduce_loss, usize_to_i32};

/// Configuration for [`ContourLoss`].
#[derive(Config, Debug, PartialEq)]
pub struct ContourLossConfig {
    /// Non-negative weight applied to the contour length term.
    #[config(default = 10.0)]
    pub weight: f64,
    /// Positive epsilon added inside the gradient magnitude square root.
    #[config(default = 1e-8)]
    pub eps: f64,
}

impl ContourLossConfig {
    /// Initializes a [`ContourLoss`] module.
    ///
    /// # Panics
    ///
    /// Panics if `weight < 0.0` or `eps` is not finite and strictly positive.
    #[must_use]
    pub fn init(&self) -> ContourLoss {
        assert!(
            self.weight.is_finite() && self.weight >= 0.0,
            "ContourLoss weight must be finite and non-negative, got {}",
            self.weight
        );
        assert!(
            self.eps.is_finite() && self.eps > 0.0,
            "ContourLoss eps must be finite and positive, got {}",
            self.eps
        );
        ContourLoss {
            weight: self.weight,
            eps: self.eps,
        }
    }
}

/// Active contour loss operating on probability maps `[B, C, H, W]` in `[0, 1]`.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct ContourLoss {
    weight: f64,
    eps: f64,
}

impl Default for ContourLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleDisplay for ContourLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("weight", &self.weight)
            .add("eps", &self.eps)
            .optional()
    }
}

impl ContourLoss {
    /// Creates a [`ContourLoss`] with default configuration (`weight = 10.0`, `eps = 1e-8`).
    #[must_use]
    pub fn new() -> Self {
        ContourLossConfig::new().init()
    }

    /// Computes reduced contour loss `[1]` from predicted probabilities and targets in `[0, 1]`.
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

    /// Computes per-sample contour loss `[batch_size]` from probabilities and targets in `[0, 1]`.
    ///
    /// # Panics
    ///
    /// Panics if shapes disagree or spatial dimensions are smaller than `3x3`.
    #[must_use]
    pub fn forward_no_reduction(&self, predictions: Tensor<4>, targets: Tensor<4>) -> Tensor<1> {
        assert_same_shape(&predictions, &targets);

        let [batch_size, _, height, width] = predictions.dims();
        assert!(
            height >= 3 && width >= 3,
            "ContourLoss requires input size >= 3x3, got [{height}x{width}]"
        );

        let delta_r = predictions.clone().slice(s![.., .., 1.., ..])
            - predictions.clone().slice(s![.., .., ..-1, ..]);
        let delta_c = predictions.clone().slice(s![.., .., .., 1..])
            - predictions.clone().slice(s![.., .., .., ..-1]);

        let row_grad_sq = delta_r.slice(s![.., .., 1.., ..-2]).powi_scalar(2);
        let col_grad_sq = delta_c.slice(s![.., .., ..-2, 1..]).powi_scalar(2);
        let delta_pred = (row_grad_sq + col_grad_sq).abs();

        let batch = usize_to_i32(batch_size);
        let length = (delta_pred + self.eps)
            .sqrt()
            .reshape([batch, -1])
            .mean_dim(1)
            .squeeze_dim::<1>(1);

        let region_in = (predictions.clone() * (targets.clone() - 1.0).powi_scalar(2))
            .reshape([batch, -1])
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let region_out = ((predictions.ones_like() - predictions) * targets.powi_scalar(2))
            .reshape([batch, -1])
            .mean_dim(1)
            .squeeze_dim::<1>(1);

        length.mul_scalar(self.weight) + region_in + region_out
    }
}
