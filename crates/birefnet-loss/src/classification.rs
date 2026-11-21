//! Auxiliary multi-level classification cross-entropy loss.

use burn::{
    module::{Content, DisplaySettings, ModuleDisplay},
    nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig, Reduction},
    prelude::*,
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
    /// Initializes a [`ClassificationLoss`] module on `device`.
    ///
    /// # Panics
    ///
    /// Panics if `weight` is not finite and strictly positive.
    #[must_use]
    pub fn init(&self, device: &Device) -> ClassificationLoss {
        assert!(
            self.weight.is_finite() && self.weight > 0.0,
            "ClassificationLoss weight must be finite and positive, got {}",
            self.weight
        );
        ClassificationLoss {
            weight: self.weight,
            ce_loss: CrossEntropyLossConfig::new().init(device),
        }
    }
}

/// Multi-level cross-entropy classification loss.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct ClassificationLoss {
    weight: f64,
    ce_loss: CrossEntropyLoss,
}

impl ModuleDisplay for ClassificationLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("weight", &self.weight)
            .add("ce_loss", &self.ce_loss)
            .optional()
    }
}

impl ClassificationLoss {
    /// Creates a [`ClassificationLoss`] on `device` with default weight `1.0`.
    #[must_use]
    pub fn new(device: &Device) -> Self {
        ClassificationLossConfig::new().init(device)
    }

    /// Computes reduced cross-entropy loss across prediction levels `[batch_size, num_classes]`.
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

    /// Computes the summed weighted cross-entropy loss across prediction levels as `[1]`.
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
        if predictions.is_empty() {
            return Tensor::zeros([1], &targets.device());
        }

        let target_batch = targets.dims()[0];
        let mut total: Option<Tensor<1>> = None;
        for (idx, pred) in predictions.iter().enumerate() {
            let pred_batch = pred.dims()[0];
            assert_eq!(
                pred_batch, target_batch,
                "prediction[{idx}] batch size ({pred_batch}) must match targets ({target_batch})"
            );
            let term = self
                .ce_loss
                .forward(pred.clone(), targets.clone())
                .mul_scalar(self.weight);
            accumulate_tensor(&mut total, term);
        }

        total.unwrap_or_else(|| Tensor::zeros([1], &targets.device()))
    }
}
