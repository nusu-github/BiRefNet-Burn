//! Threshold regularization loss encouraging confident binary probabilities (`0` or `1`).

use burn::{
    module::{Content, DisplaySettings, ModuleDisplay},
    nn::loss::Reduction,
    prelude::*,
};

use crate::ops::reduce_loss;

/// Configuration for [`ThresholdRegularizationLoss`].
#[derive(Config, Debug, PartialEq)]
pub struct ThresholdRegularizationLossConfig {
    /// Positive scalar multiplier applied after reduction.
    #[config(default = 1.0)]
    pub weight: f64,
}

impl ThresholdRegularizationLossConfig {
    /// Initializes a [`ThresholdRegularizationLoss`] module.
    ///
    /// # Panics
    ///
    /// Panics if `weight` is not finite and strictly positive.
    #[must_use]
    pub fn init(&self) -> ThresholdRegularizationLoss {
        assert!(
            self.weight.is_finite() && self.weight > 0.0,
            "ThresholdRegularizationLoss weight must be finite and positive, got {}",
            self.weight
        );
        ThresholdRegularizationLoss {
            weight: self.weight,
        }
    }
}

/// Threshold regularization loss `weight * (1 - (p^2 + (p - 1)^2))`.
///
/// Reaches `0` when `p` is `0` or `1` and `0.5 * weight` at `p = 0.5`.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct ThresholdRegularizationLoss {
    weight: f64,
}

impl Default for ThresholdRegularizationLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleDisplay for ThresholdRegularizationLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add("weight", &self.weight).optional()
    }
}

impl ThresholdRegularizationLoss {
    /// Creates a [`ThresholdRegularizationLoss`] with default weight `1.0`.
    #[must_use]
    pub fn new() -> Self {
        ThresholdRegularizationLossConfig::new().init()
    }

    /// Computes reduced threshold regularization loss multiplied by `weight`.
    #[must_use]
    pub fn forward<const D: usize>(
        &self,
        predictions: Tensor<D>,
        reduction: Reduction,
    ) -> Tensor<1> {
        let loss = self.forward_no_reduction(predictions);
        reduce_loss(loss, reduction).mul_scalar(self.weight)
    }

    /// Computes element-wise `1 - (p^2 + (p - 1)^2)`.
    #[must_use]
    pub fn forward_no_reduction<const D: usize>(&self, predictions: Tensor<D>) -> Tensor<D> {
        // 1 - (p^2 + (p - 1)^2) simplifies to 2 * p * (1 - p).
        predictions.clone().neg().add_scalar(1.0) * predictions.mul_scalar(2.0)
    }
}
