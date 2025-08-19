//! Structure loss for edge-aware segmentation training.
//!
//! Combines weighted binary cross-entropy and weighted IoU loss with
//! edge-aware weighting to emphasize boundaries in segmentation tasks.
//!
//! The loss is computed as:
//! ```text
//! weit = 1 + 5 * |avg_pool(target) - target|
//! wbce = weighted_bce_with_logits(pred, target, weit)
//! wiou = weighted_iou_loss(sigmoid(pred), target, weit)
//! Loss = mean(wbce + wiou)
//! ```

use burn::{
    config::Config,
    module::{Content, DisplaySettings, Module, ModuleDisplay},
    nn::{
        loss::Reduction,
        pool::{AvgPool2d, AvgPool2dConfig},
        PaddingConfig2d,
    },
    tensor::{activation::sigmoid, backend::Backend, Int, Tensor},
};

/// Configuration for creating a [Structure loss](StructureLoss).
#[derive(Config, Debug)]
pub struct StructureLossConfig {
    /// Weight factor applied to the final loss. Default: 1.0
    #[config(default = 1.0)]
    pub weight: f64,

    /// Edge enhancement factor for weighting. Default: 5.0
    #[config(default = 5.0)]
    pub edge_factor: f64,

    /// Kernel size for average pooling to detect edges. Default: 31
    #[config(default = 31)]
    pub kernel_size: usize,

    /// Small epsilon to avoid division by zero. Default: 1e-8
    #[config(default = 1e-8)]
    pub eps: f64,
}

impl StructureLossConfig {
    /// Initialize [Structure loss](StructureLoss).
    pub fn init(&self) -> StructureLoss {
        self.assertions();

        let padding = self.kernel_size / 2;
        let avg_pool = AvgPool2dConfig::new([self.kernel_size, self.kernel_size])
            .with_strides([1, 1])
            .with_padding(PaddingConfig2d::Explicit(padding, padding))
            .init();

        StructureLoss {
            weight: self.weight,
            edge_factor: self.edge_factor,
            eps: self.eps,
            avg_pool,
        }
    }

    fn assertions(&self) {
        assert!(
            self.weight > 0.0,
            "Weight for StructureLoss must be positive, got {}",
            self.weight
        );
        assert!(
            self.edge_factor >= 0.0,
            "Edge factor for StructureLoss must be non-negative, got {}",
            self.edge_factor
        );
        assert!(
            self.kernel_size > 0 && self.kernel_size % 2 == 1,
            "Kernel size for StructureLoss must be positive and odd, got {}",
            self.kernel_size
        );
        assert!(
            self.eps > 0.0,
            "Epsilon for StructureLoss must be positive, got {}",
            self.eps
        );
    }
}

/// Structure loss for edge-aware segmentation training.
///
/// This loss combines weighted binary cross-entropy and weighted IoU loss
/// with edge-aware weighting to emphasize boundaries in segmentation tasks.
/// The edge weighting is computed using average pooling to detect regions
/// where the target changes rapidly.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct StructureLoss {
    /// Weight factor applied to the final loss.
    pub weight: f64,
    /// Edge enhancement factor for weighting.
    pub edge_factor: f64,
    /// Small epsilon to avoid division by zero.
    pub eps: f64,
    /// Average pooling module for edge detection.
    pub avg_pool: AvgPool2d,
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
            .add("avg_pool", &self.avg_pool)
            .optional()
    }
}

impl StructureLoss {
    /// Create a new structure loss with default configuration.
    pub fn new() -> Self {
        StructureLossConfig::new().init()
    }

    /// Compute the criterion on the input tensor with reduction.
    ///
    /// # Shapes
    ///
    /// - predictions: `[batch_size, channels, height, width]` (logits)
    /// - targets: `[batch_size, channels, height, width]` (binary values)
    /// - output: `[1]`
    pub fn forward<B: Backend>(
        &self,
        predictions: Tensor<B, 4>,
        targets: Tensor<B, 4, Int>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let loss = self.forward_no_reduction(predictions, targets);
        let reduced = match reduction {
            Reduction::Mean | Reduction::Auto => loss.mean(),
            Reduction::Sum => loss.sum(),
        };

        // Apply weight factor
        reduced.mul_scalar(self.weight)
    }

    /// Compute the criterion on the input tensor without reduction.
    ///
    /// # Shapes
    ///
    /// - predictions: `[batch_size, channels, height, width]` (logits)
    /// - targets: `[batch_size, channels, height, width]` (binary values)
    /// - output: `[batch_size]`
    pub fn forward_no_reduction<B: Backend>(
        &self,
        predictions: Tensor<B, 4>,
        targets: Tensor<B, 4, Int>,
    ) -> Tensor<B, 1> {
        self.assertions(&predictions, &targets);

        let targets_float = targets.float();

        // Apply average pooling to detect edges
        let pooled = self.avg_pool.forward(targets_float.clone());

        // Calculate edge weight: weit = 1 + edge_factor * |avg_pool(target) - target|
        let edge_diff = (pooled - targets_float.clone()).abs();
        let weit = edge_diff.mul_scalar(self.edge_factor).add_scalar(1.0);

        // Weighted BCE loss with logits (numerically stable)
        let wbce =
            self.weighted_bce_with_logits(predictions.clone(), targets_float.clone(), weit.clone());

        // Weighted IoU loss
        let pred_sigmoid = sigmoid(predictions);
        let wiou = self.weighted_iou_loss(pred_sigmoid, targets_float, weit);

        // Combine losses for each batch element
        wbce + wiou
    }

    /// Compute weighted binary cross-entropy with logits.
    fn weighted_bce_with_logits<B: Backend>(
        &self,
        logits: Tensor<B, 4>,
        targets: Tensor<B, 4>,
        weights: Tensor<B, 4>,
    ) -> Tensor<B, 1> {
        // Numerically stable BCE with logits: max(x, 0) - x*y + log(1 + exp(-abs(x)))
        let term1 = logits.clone().clamp_min(0.0) - logits.clone() * targets;
        let term2 = (-logits.abs()).exp().add_scalar(1.0).log();
        let bce = (term1 + term2) * weights.clone();

        // Weighted sum over spatial dimensions (H, W) - reduces [B,C,H,W] to [B,C,1,1]
        let bce_sum = bce.sum_dim(3).sum_dim(2);
        let weights_sum = weights.sum_dim(3).sum_dim(2);

        // Weighted average and flatten spatial dims: [B,C,1,1] -> [B,C]
        let [batch_size, channels, _, _] = bce_sum.dims();
        let bce_avg = (bce_sum / weights_sum.clamp_min(self.eps))
            .reshape([batch_size as i32, channels as i32]);

        // Mean over channels: [B,C] -> [B,1] -> [B]
        bce_avg.mean_dim(1).squeeze(1)
    }

    /// Compute weighted IoU loss.
    fn weighted_iou_loss<B: Backend>(
        &self,
        predictions: Tensor<B, 4>,
        targets: Tensor<B, 4>,
        weights: Tensor<B, 4>,
    ) -> Tensor<B, 1> {
        // Weighted intersection and union
        let weighted_pred = predictions.clone() * weights.clone();
        let weighted_target = targets.clone() * weights.clone();
        let weighted_union = (predictions + targets) * weights;

        // Sum over spatial dimensions (H, W) - reduces [B,C,H,W] to [B,C,1,1]
        let inter = (weighted_pred * weighted_target).sum_dim(3).sum_dim(2);
        let union = weighted_union.sum_dim(3).sum_dim(2);

        // Flatten spatial dims: [B,C,1,1] -> [B,C]
        let [batch_size, channels, _, _] = inter.dims();
        let inter_flat = inter.reshape([batch_size as i32, channels as i32]);
        let union_flat = union.reshape([batch_size as i32, channels as i32]);

        // IoU = (inter + eps) / (union - inter + eps)
        let iou = (inter_flat.clone().add_scalar(self.eps))
            / (union_flat - inter_flat.add_scalar(self.eps));

        // IoU loss = 1 - IoU, average over channels: [B,C] -> [B,1] -> [B]
        (iou.ones_like() - iou).mean_dim(1).squeeze(1)
    }

    fn assertions<B: Backend>(&self, predictions: &Tensor<B, 4>, targets: &Tensor<B, 4, Int>) {
        let pred_dims = predictions.dims();
        let target_dims = targets.dims();
        assert_eq!(
            pred_dims, target_dims,
            "Shape of predictions ({pred_dims:?}) must match targets ({target_dims:?})"
        );
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{cast::ToElement, TensorData};

    use super::*;
    use crate::tests::TestBackend;
    #[test]
    fn structure_loss_forward_perfect_match_returns_non_negative() {
        let device = Default::default();
        let loss = StructureLoss::new();

        // Perfect match: predictions and targets are identical
        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.8, 0.9], [0.7, 0.95]]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[1, 1], [1, 1]]]]),
            &device,
        );

        let result = loss.forward(predictions, targets, Reduction::Mean);

        // Should be low loss for good predictions
        assert!(
            result.into_scalar().to_f64() >= 0.0,
            "Loss should be non-negative"
        );
    }

    #[test]
    fn structure_loss_forward_checkerboard_pattern_computes_finite_values() {
        let device = Default::default();
        let loss = StructureLoss::new();

        // Create targets with clear edges (checkerboard pattern)
        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.1, 0.9], [0.9, 0.1]]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[0, 1], [1, 0]]]]),
            &device,
        );

        let result_mean = loss.forward(predictions.clone(), targets.clone(), Reduction::Mean);
        let result_sum = loss.forward(predictions.clone(), targets.clone(), Reduction::Sum);
        let result_no_reduction = loss.forward_no_reduction(predictions, targets);

        // All should be valid finite values
        assert!(result_mean.into_scalar().to_f64().is_finite());
        assert!(result_sum.into_scalar().to_f64().is_finite());
        assert!(result_no_reduction
            .clone()
            .sum()
            .into_scalar()
            .to_f64()
            .is_finite());

        // Should have shape [batch_size] for no reduction
        assert_eq!(result_no_reduction.dims(), [1]);
    }

    #[test]
    fn structure_loss_processes_batches_correctly() {
        let device = Default::default();
        let loss = StructureLoss::new();

        // Batch of 2 samples
        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([
                [[[0.8, 0.2], [0.3, 0.9]]],    // Sample 1
                [[[0.1, 0.95], [0.85, 0.05]]], // Sample 2
            ]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([
                [[[1, 0], [0, 1]]], // Sample 1
                [[[0, 1], [1, 0]]], // Sample 2
            ]),
            &device,
        );

        let result_mean = loss.forward(predictions.clone(), targets.clone(), Reduction::Mean);
        let result_no_reduction = loss.forward_no_reduction(predictions, targets);

        // Check shapes
        assert_eq!(result_mean.dims(), [1]);
        assert_eq!(result_no_reduction.dims(), [2]); // batch_size = 2

        // All values should be finite and non-negative
        assert!(result_mean.into_scalar().to_f64() >= 0.0);
        for i in 0..2 {
            let sample_loss = result_no_reduction
                .clone()
                .select(0, Tensor::from_data([i], &device))
                .into_scalar()
                .to_f64();
            assert!(sample_loss >= 0.0, "Sample {i} loss should be non-negative");
        }
    }

    #[test]
    fn structure_loss_with_custom_weight_and_edge_factor_applies_correctly() {
        let device = Default::default();
        let config = StructureLossConfig::new()
            .with_weight(2.0)
            .with_edge_factor(3.0)
            .with_kernel_size(3)
            .with_eps(1e-6);
        let loss = config.init();

        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.5, 0.5], [0.5, 0.5]]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[1, 0], [0, 1]]]]),
            &device,
        );

        let result = loss.forward(predictions, targets, Reduction::Mean);

        // Should incorporate the weight factor (2.0)
        assert!(result.into_scalar().to_f64() > 0.0);
    }

    #[test]
    fn structure_loss_auto_reduction_equals_mean_reduction() {
        let device = Default::default();
        let loss = StructureLoss::new();

        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.3, 0.7], [0.8, 0.2]]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[0, 1], [1, 0]]]]),
            &device,
        );

        let result_auto = loss.forward(predictions.clone(), targets.clone(), Reduction::Auto);
        let result_mean = loss.forward(predictions, targets, Reduction::Mean);

        let auto_val = result_auto.into_scalar().to_f64();
        let mean_val = result_mean.into_scalar().to_f64();

        assert!((auto_val - mean_val).abs() < 1e-6, "Auto should equal Mean");
    }

    #[test]
    #[should_panic = "Weight for StructureLoss must be positive"]
    fn structure_loss_config_negative_weight_panics() {
        let _loss = StructureLossConfig::new().with_weight(-1.0).init();
    }

    #[test]
    #[should_panic = "Kernel size for StructureLoss must be positive and odd"]
    fn structure_loss_config_even_kernel_size_panics() {
        let _loss = StructureLossConfig::new().with_kernel_size(4).init();
    }

    #[test]
    #[should_panic = "Epsilon for StructureLoss must be positive"]
    fn structure_loss_config_negative_epsilon_panics() {
        let _loss = StructureLossConfig::new().with_eps(-1e-6).init();
    }

    #[test]
    #[should_panic = "Shape of predictions"]
    fn structure_loss_forward_mismatched_shapes_panics() {
        let device = Default::default();
        let loss = StructureLoss::new();

        let predictions =
            Tensor::<TestBackend, 4>::from_data(TensorData::from([[[[1.0, 2.0]]]]), &device);
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[1, 2], [3, 4]]]]),
            &device,
        );

        let _result = loss.forward_no_reduction(predictions, targets);
    }

    #[test]
    fn structure_loss_display_shows_weight_and_edge_factor() {
        let config = StructureLossConfig::new()
            .with_weight(0.5)
            .with_edge_factor(2.0)
            .with_eps(1e-6);
        let loss = config.init();

        let display_str = format!("{loss}");
        assert!(display_str.contains("StructureLoss"));
        assert!(display_str.contains("weight: 0.5"));
        assert!(display_str.contains("edge_factor: 2"));
        assert!(display_str.contains("eps: 0.000001"));
    }
}
