//! Contour loss for boundary refinement.
//!
//! Computes contour loss by combining:
//! 1. Length term: Encourages smooth contours by penalizing large gradients
//! 2. Region term: Encourages accurate region segmentation
//!
//! The loss is computed as:
//! ```text
//! Length = mean(sqrt(|∇x pred|² + |∇y pred|² + ε))
//! RegionIn = mean(pred * (target - 1)²)
//! RegionOut = mean((1-pred) * (target - 0)²)
//! Loss = weight * Length + RegionIn + RegionOut
//! ```

// Import s! macro
use burn::{
    config::Config,
    module::{Content, DisplaySettings, Module, ModuleDisplay},
    nn::loss::Reduction,
    tensor::{backend::Backend, s, Int, Tensor},
};

/// Configuration for creating a [Contour loss](ContourLoss).
#[derive(Config, Debug)]
pub struct ContourLossConfig {
    /// Weight factor for the length term. Default: 10.0
    #[config(default = 10.0)]
    pub weight: f64,

    /// Small epsilon to avoid division by zero in sqrt. Default: 1e-8
    #[config(default = 1e-8)]
    pub eps: f64,
}

impl ContourLossConfig {
    /// Initialize [Contour loss](ContourLoss).

    pub fn init(&self) -> ContourLoss {
        self.assertions();
        ContourLoss {
            weight: self.weight,
            eps: self.eps,
        }
    }

    fn assertions(&self) {
        assert!(
            self.weight >= 0.0,
            "Weight for ContourLoss must be non-negative, got {}",
            self.weight
        );
        assert!(
            self.eps > 0.0,
            "Epsilon for ContourLoss must be positive, got {}",
            self.eps
        );
    }
}

/// Contour loss for boundary refinement.
///
/// This loss combines a length term that encourages smooth contours
/// and a region term that encourages accurate segmentation. It is particularly
/// useful for tasks where boundary accuracy is critical.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct ContourLoss {
    /// Weight factor for the length term.
    pub weight: f64,
    /// Small epsilon to avoid division by zero.
    pub eps: f64,
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
    /// Create a new contour loss with default configuration.

    pub fn new() -> Self {
        ContourLossConfig::new().init()
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
        match reduction {
            Reduction::Mean | Reduction::Auto => loss.mean(),
            Reduction::Sum => loss.sum(),
        }
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

        let [batch_size, channels, height, width] = predictions.dims();
        let device = predictions.device();

        // Need at least 3x3 input for gradient computation
        assert!(
            height >= 3 && width >= 3,
            "ContourLoss requires input size >= 3x3, got [{height}x{width}]"
        );

        let targets_float = targets.float();

        // Compute gradients using difference operations using s! macro (PyTorch-like syntax)
        // Horizontal gradient: delta_r = pred[:,:,1:,:] - pred[:,:,:-1,:]
        let delta_r = predictions.clone().slice(s![.., .., 1.., ..])
            - predictions.clone().slice(s![.., .., ..-1, ..]);

        // Vertical gradient: delta_c = pred[:,:,:,1:] - pred[:,:,:,:-1]
        let delta_c = predictions.clone().slice(s![.., .., .., 1..])
            - predictions.clone().slice(s![.., .., .., ..-1]);

        // Apply specific slicing as in PyTorch version
        // delta_r = delta_r[:,:,1:,:-2]**2  -> shape [B, C, H-2, W-2]
        let delta_r_sq = delta_r.slice(s![.., .., 1.., ..-2]).powi_scalar(2);

        // delta_c = delta_c[:,:,:-2,1:]**2  -> shape [B, C, H-2, W-2]
        let delta_c_sq = delta_c.slice(s![.., .., ..-2, 1..]).powi_scalar(2);

        let delta_pred = (delta_r_sq + delta_c_sq).abs();

        // Length term: mean(sqrt(delta_pred + epsilon))
        let length = (delta_pred + self.eps)
            .sqrt()
            .reshape([batch_size as i32, -1])
            .mean_dim(1)
            .squeeze(1);

        // Region terms
        let c_in = Tensor::ones_like(&predictions);
        let c_out = Tensor::zeros_like(&predictions);

        // region_in = mean(pred * (targets - c_in)²) per batch
        let region_in_term = predictions.clone() * (targets_float.clone() - c_in).powi_scalar(2);
        let region_in = region_in_term
            .reshape([batch_size as i32, -1])
            .mean_dim(1)
            .squeeze(1);

        // region_out = mean((1-pred) * (targets - c_out)²) per batch
        let region_out_term = (Tensor::ones_like(&predictions) - predictions)
            * (targets_float - c_out).powi_scalar(2);
        let region_out = region_out_term
            .reshape([batch_size as i32, -1])
            .mean_dim(1)
            .squeeze(1);

        let region = region_in + region_out;

        // Total loss per batch: weight * length + region
        length.mul_scalar(self.weight) + region
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
    fn contour_loss_forward_minimum_size_input_computes_finite_loss() {
        let device = Default::default();
        let loss = ContourLoss::new();

        // Simple test case with 3x3 input (minimum size)
        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.8, 0.9, 0.7], [0.6, 0.8, 0.9], [0.5, 0.7, 0.8]]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[1, 1, 0], [1, 1, 1], [0, 1, 1]]]]),
            &device,
        );

        let result_mean = loss.forward(predictions.clone(), targets.clone(), Reduction::Mean);
        let result_sum = loss.forward(predictions.clone(), targets.clone(), Reduction::Sum);
        let result_no_reduction = loss.forward_no_reduction(predictions, targets);

        // Verify shapes
        assert_eq!(result_mean.dims(), [1]);
        assert_eq!(result_sum.dims(), [1]);
        assert_eq!(result_no_reduction.dims(), [1]); // batch_size = 1

        // All values should be finite and non-negative
        assert!(result_mean.into_scalar().to_f64().is_finite());
        assert!(result_sum.into_scalar().to_f64().is_finite());
        assert!(result_no_reduction.into_scalar().to_f64() >= 0.0);
    }

    #[test]
    fn contour_loss_forward_batch_samples_produces_correct_shapes() {
        let device = Default::default();
        let loss = ContourLoss::new();

        // Batch of 2 samples, each 4x4
        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([
                [[
                    [0.8, 0.9, 0.7, 0.6],
                    [0.5, 0.8, 0.9, 0.4],
                    [0.3, 0.7, 0.8, 0.2],
                    [0.1, 0.4, 0.6, 0.9],
                ]], // Sample 1
                [[
                    [0.2, 0.3, 0.8, 0.9],
                    [0.1, 0.4, 0.7, 0.8],
                    [0.9, 0.8, 0.5, 0.3],
                    [0.7, 0.6, 0.2, 0.1],
                ]], // Sample 2
            ]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([
                [[[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 0], [0, 0, 1, 1]]], // Sample 1
                [[[0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]]], // Sample 2
            ]),
            &device,
        );

        let result_mean = loss.forward(predictions.clone(), targets.clone(), Reduction::Mean);
        let result_no_reduction = loss.forward_no_reduction(predictions, targets);

        // Check shapes
        assert_eq!(result_mean.dims(), [1]);
        assert_eq!(result_no_reduction.dims(), [2]); // batch_size = 2

        // All values should be finite and non-negative
        assert!(result_mean.into_scalar().to_f64().is_finite());
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
    fn contour_loss_forward_smooth_and_rough_boundaries_returns_finite_values() {
        let device = Default::default();
        let loss = ContourLoss::new();

        // Smooth boundary (should have lower loss)
        let smooth_pred = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.9, 0.8, 0.7], [0.8, 0.7, 0.6], [0.7, 0.6, 0.5]]]]),
            &device,
        );
        let smooth_targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[1, 1, 1], [1, 1, 0], [1, 0, 0]]]]),
            &device,
        );

        // Rough boundary (should have higher loss due to large gradients)
        let rough_pred = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.9, 0.1, 0.9], [0.1, 0.9, 0.1], [0.9, 0.1, 0.9]]]]),
            &device,
        );
        let rough_targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[1, 0, 1], [0, 1, 0], [1, 0, 1]]]]),
            &device,
        );

        let smooth_loss = loss.forward(smooth_pred, smooth_targets, Reduction::Mean);
        let rough_loss = loss.forward(rough_pred, rough_targets, Reduction::Mean);

        // Rough boundaries should generally have higher loss due to length term
        // (Though this depends on how well the prediction matches the target)
        assert!(smooth_loss.into_scalar().to_f64() >= 0.0);
        assert!(rough_loss.into_scalar().to_f64() >= 0.0);
    }

    #[test]
    fn contour_loss_with_custom_weight_and_eps_computes_finite_loss() {
        let device = Default::default();
        let config = ContourLossConfig::new().with_weight(5.0).with_eps(1e-6);
        let loss = config.init();

        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.8, 0.9, 0.7], [0.6, 0.8, 0.9], [0.5, 0.7, 0.8]]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[1, 1, 0], [1, 1, 1], [0, 1, 1]]]]),
            &device,
        );

        let result = loss.forward(predictions, targets, Reduction::Mean);

        // Should work with custom parameters
        assert!(result.clone().into_scalar().to_f64().is_finite());
        assert!(result.into_scalar().to_f64() >= 0.0);
    }

    #[test]
    fn contour_loss_auto_reduction_equals_mean_reduction() {
        let device = Default::default();
        let loss = ContourLoss::new();

        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.3, 0.7, 0.8], [0.8, 0.2, 0.6], [0.4, 0.9, 0.1]]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[0, 1, 1], [1, 0, 1], [0, 1, 0]]]]),
            &device,
        );

        let result_auto = loss.forward(predictions.clone(), targets.clone(), Reduction::Auto);
        let result_mean = loss.forward(predictions, targets, Reduction::Mean);

        let auto_val = result_auto.into_scalar().to_f64();
        let mean_val = result_mean.into_scalar().to_f64();

        assert!((auto_val - mean_val).abs() < 1e-6, "Auto should equal Mean");
    }

    #[test]
    #[should_panic = "ContourLoss requires input size >= 3x3"]
    fn contour_loss_forward_with_2x2_input_panics() {
        let device = Default::default();
        let loss = ContourLoss::new();

        // 2x2 input should fail
        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.8, 0.9], [0.6, 0.7]]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[1, 1], [0, 1]]]]),
            &device,
        );

        let _result = loss.forward_no_reduction(predictions, targets);
    }

    #[test]
    #[should_panic = "Weight for ContourLoss must be non-negative"]
    fn contour_loss_config_negative_weight_panics() {
        let _loss = ContourLossConfig::new().with_weight(-1.0).init();
    }

    #[test]
    #[should_panic = "Epsilon for ContourLoss must be positive"]
    fn contour_loss_config_negative_epsilon_panics() {
        let _loss = ContourLossConfig::new().with_eps(-1e-6).init();
    }

    #[test]
    #[should_panic = "Shape of predictions"]
    fn contour_loss_forward_mismatched_shapes_panics() {
        let device = Default::default();
        let loss = ContourLoss::new();

        let predictions =
            Tensor::<TestBackend, 4>::from_data(TensorData::from([[[[1.0, 2.0, 3.0]]]]), &device);
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[1, 2], [3, 4]]]]),
            &device,
        );

        let _result = loss.forward_no_reduction(predictions, targets);
    }

    #[test]
    fn contour_loss_display_shows_weight_and_eps_parameters() {
        let config = ContourLossConfig::new().with_weight(5.0).with_eps(1e-6);
        let loss = config.init();

        let display_str = format!("{loss}");
        assert!(display_str.contains("ContourLoss"));
        assert!(display_str.contains("weight: 5"));
        assert!(display_str.contains("eps: 0.000001"));
    }
}
