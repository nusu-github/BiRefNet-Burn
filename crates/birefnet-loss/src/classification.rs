//! Auxiliary multi-level classification cross-entropy loss.

use burn::{
    module::{Content, DisplaySettings, ModuleDisplay},
    nn::loss::Reduction,
    prelude::*,
    tensor::activation::log_softmax,
};

use crate::ops::{accumulate_tensor, reduce_loss};

/// Configuration for [`ClassificationLoss`].
#[derive(Config, Debug, PartialEq)]
pub struct ClassificationLossConfig {
    /// Positive scalar weight applied to each level's cross-entropy loss.
    #[config(default = 1.0)]
    pub weight: f64,
}

impl ClassificationLossConfig {
    /// Initializes a [`ClassificationLoss`] module.
    ///
    /// # Panics
    ///
    /// Panics if `weight` is not finite and strictly positive.
    #[must_use]
    pub fn init(&self) -> ClassificationLoss {
        assert!(
            self.weight.is_finite() && self.weight > 0.0,
            "ClassificationLoss weight must be finite and positive, got {}",
            self.weight
        );
        ClassificationLoss {
            weight: self.weight,
        }
    }
}

/// Multi-level cross-entropy classification loss.
///
/// Each prediction level contributes `weight * CE(logits, targets)`; levels are summed.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct ClassificationLoss {
    weight: f64,
}

impl Default for ClassificationLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleDisplay for ClassificationLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content.add("weight", &self.weight).optional()
    }
}

impl ClassificationLoss {
    /// Creates a [`ClassificationLoss`] with default weight `1.0`.
    #[must_use]
    pub fn new() -> Self {
        ClassificationLossConfig::new().init()
    }

    /// Computes reduced cross-entropy loss over levels of `[batch_size, num_classes]` logits.
    ///
    /// `Reduction::Mean` matches PyTorch's `sum(CrossEntropyLoss()(level, targets))`.
    ///
    /// # Panics
    ///
    /// Panics if any prediction batch dimension disagrees with `targets`.
    #[must_use]
    pub fn forward(
        &self,
        predictions: &[Tensor<2>],
        targets: &Tensor<1, Int>,
        reduction: Reduction,
    ) -> Tensor<1> {
        let loss = self.forward_no_reduction(predictions, targets);
        reduce_loss(loss, reduction)
    }

    /// Computes the per-sample weighted cross-entropy summed across levels as `[batch_size]`.
    ///
    /// Returns zeros when `predictions` is empty.
    ///
    /// # Panics
    ///
    /// Panics if any prediction batch dimension disagrees with `targets`.
    #[must_use]
    pub fn forward_no_reduction(
        &self,
        predictions: &[Tensor<2>],
        targets: &Tensor<1, Int>,
    ) -> Tensor<1> {
        let target_batch = targets.dims()[0];
        let target_index = targets.clone().unsqueeze_dim::<2>(1);
        let mut total: Option<Tensor<1>> = None;
        for (idx, pred) in predictions.iter().enumerate() {
            let pred_batch = pred.dims()[0];
            assert_eq!(
                pred_batch, target_batch,
                "prediction[{idx}] batch size ({pred_batch}) must match targets ({target_batch})"
            );
            let nll = log_softmax(pred.clone(), 1)
                .gather(1, target_index.clone())
                .squeeze_dim::<1>(1)
                .neg();
            accumulate_tensor(&mut total, nll.mul_scalar(self.weight));
        }

        total.unwrap_or_else(|| Tensor::zeros([target_batch], &targets.device()))
    }
}
