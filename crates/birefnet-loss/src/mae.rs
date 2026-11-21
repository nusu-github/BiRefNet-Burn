//! Mean Absolute Error (L1) loss.

use burn::{
    module::{Content, DisplaySettings, ModuleDisplay},
    nn::loss::Reduction,
    prelude::*,
};

use crate::ops::{assert_same_shape, reduce_loss};

/// Configuration for [`MaeLoss`].
#[derive(Config, Debug, PartialEq)]
pub struct MaeLossConfig {
    /// Positive scalar weight multiplier for the reduced loss.
    #[config(default = 1.0)]
    pub weight: f64,
}

impl MaeLossConfig {
    /// Initializes an [`MaeLoss`] module.
    ///
    /// # Panics
    ///
    /// Panics if `weight` is not finite and strictly positive.
    #[must_use]
    pub fn init(&self) -> MaeLoss {
        assert!(
            self.weight.is_finite() && self.weight > 0.0,
            "MaeLoss weight must be finite and positive, got {}",
            self.weight
        );
        MaeLoss {
            weight: self.weight,
        }
    }
}

/// Mean Absolute Error (L1) loss module.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct MaeLoss {
    weight: f64,
}

impl Default for MaeLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleDisplay for MaeLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add("weight", &self.weight).optional()
    }
}

impl MaeLoss {
    /// Creates an [`MaeLoss`] with default weight `1.0`.
    #[must_use]
    pub fn new() -> Self {
        MaeLossConfig::new().init()
    }

    /// Computes reduced MAE loss multiplied by `weight`.
    #[must_use]
    pub fn forward<const D: usize>(
        &self,
        predictions: Tensor<D>,
        targets: Tensor<D>,
        reduction: Reduction,
    ) -> Tensor<1> {
        let loss = self.forward_no_reduction(predictions, targets);
        reduce_loss(loss, reduction).mul_scalar(self.weight)
    }

    /// Computes element-wise `|predictions - targets|`.
    ///
    /// # Panics
    ///
    /// Panics if `predictions` and `targets` have different shapes.
    #[must_use]
    pub fn forward_no_reduction<const D: usize>(
        &self,
        predictions: Tensor<D>,
        targets: Tensor<D>,
    ) -> Tensor<D> {
        assert_same_shape(&predictions, &targets);
        (predictions - targets).abs()
    }
}
