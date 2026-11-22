//! Structure loss (`F^3-Net`) combining edge-weighted BCE with logits and edge-weighted IoU.

use burn::{
    module::{Content, DisplaySettings, ModuleDisplay},
    nn::{
        PaddingConfig2d,
        loss::Reduction,
        pool::{AvgPool2d, AvgPool2dConfig},
    },
    prelude::*,
    tensor::activation::sigmoid,
};

use crate::ops::{assert_same_shape, bce_with_logits, reduce_loss};

/// Configuration for [`StructureLoss`].
#[derive(Config, Debug, PartialEq)]
pub struct StructureLossConfig {
    /// Positive scalar multiplier applied after reduction.
    #[config(default = 1.0)]
    pub weight: f64,
    /// Non-negative edge enhancement weight factor.
    #[config(default = 5.0)]
    pub edge_factor: f64,
    /// Positive odd pooling kernel size used to detect boundaries.
    #[config(default = 31)]
    pub kernel_size: usize,
    /// Positive epsilon for numerical stability.
    #[config(default = 1e-8)]
    pub eps: f64,
}

impl StructureLossConfig {
    /// Initializes a [`StructureLoss`] module.
    ///
    /// # Panics
    ///
    /// Panics if `weight <= 0.0`, `edge_factor < 0.0`, `kernel_size` is even or zero, or `eps <= 0.0`.
    #[must_use]
    pub fn init(&self) -> StructureLoss {
        assert!(
            self.weight.is_finite() && self.weight > 0.0,
            "StructureLoss weight must be finite and positive, got {}",
            self.weight
        );
        assert!(
            self.edge_factor.is_finite() && self.edge_factor >= 0.0,
            "StructureLoss edge_factor must be finite and non-negative, got {}",
            self.edge_factor
        );
        assert!(
            self.kernel_size > 0 && self.kernel_size % 2 == 1,
            "StructureLoss kernel_size must be positive and odd, got {}",
            self.kernel_size
        );
        assert!(
            self.eps.is_finite() && self.eps > 0.0,
            "StructureLoss eps must be finite and positive, got {}",
            self.eps
        );

        let padding = self.kernel_size / 2;
        let avg_pool = AvgPool2dConfig::new([self.kernel_size, self.kernel_size])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(
                padding, padding, padding, padding,
            ))
            .init();

        StructureLoss {
            weight: self.weight,
            edge_factor: self.edge_factor,
            eps: self.eps,
            avg_pool,
        }
    }
}

/// Edge-weighted structure loss (`wbce + wiou`) operating on raw logits and targets in `[0, 1]`.
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct StructureLoss {
    weight: f64,
    edge_factor: f64,
    eps: f64,
    avg_pool: AvgPool2d,
}

impl Default for StructureLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl ModuleDisplay for StructureLoss {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("weight", &self.weight)
            .add("edge_factor", &self.edge_factor)
            .add("eps", &self.eps)
            .optional()
    }
}

impl StructureLoss {
    /// Creates a [`StructureLoss`] with default configuration.
    #[must_use]
    pub fn new() -> Self {
        StructureLossConfig::new().init()
    }

    /// Computes reduced weighted structure loss `[1]` from raw logits and targets in `[0, 1]`.
    #[must_use]
    pub fn forward(
        &self,
        predictions: Tensor<4>,
        targets: Tensor<4>,
        reduction: Reduction,
    ) -> Tensor<1> {
        let loss = self.forward_no_reduction(predictions, targets);
        reduce_loss(loss, reduction).mul_scalar(self.weight)
    }

    /// Computes per-sample structure loss `[batch_size]` from raw logits and targets in `[0, 1]`.
    ///
    /// # Panics
    ///
    /// Panics if `predictions` and `targets` have different shapes.
    #[must_use]
    pub fn forward_no_reduction(&self, predictions: Tensor<4>, targets: Tensor<4>) -> Tensor<1> {
        assert_same_shape(&predictions, &targets);

        let pooled = self.avg_pool.forward(targets.clone());
        let edge_diff = (pooled - targets.clone()).abs();
        let weights = edge_diff.mul_scalar(self.edge_factor).add_scalar(1.0);

        let wbce =
            self.weighted_bce_with_logits(predictions.clone(), targets.clone(), weights.clone());
        let wiou = self.weighted_iou_loss(sigmoid(predictions), targets, weights);
        wbce + wiou
    }

    fn weighted_bce_with_logits(
        &self,
        logits: Tensor<4>,
        targets: Tensor<4>,
        weights: Tensor<4>,
    ) -> Tensor<1> {
        let weighted_bce = bce_with_logits(logits, targets) * weights.clone();
        let bce_sum = weighted_bce.sum_dims_squeeze::<2, _>(&[2, 3]);
        let weights_sum = weights
            .sum_dims_squeeze::<2, _>(&[2, 3])
            .clamp_min(self.eps);

        (bce_sum / weights_sum).mean_dim(1).squeeze_dim::<1>(1)
    }

    fn weighted_iou_loss(
        &self,
        pred: Tensor<4>,
        targets: Tensor<4>,
        weights: Tensor<4>,
    ) -> Tensor<1> {
        let inter =
            (pred.clone() * targets.clone() * weights.clone()).sum_dims_squeeze::<2, _>(&[2, 3]);
        let union = ((pred + targets) * weights).sum_dims_squeeze::<2, _>(&[2, 3]);

        let denom = (union - inter.clone()).add_scalar(self.eps);
        let iou = inter.add_scalar(self.eps) / denom;

        iou.neg().add_scalar(1.0).mean_dim(1).squeeze_dim::<1>(1)
    }
}
