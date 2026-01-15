//! Intersection over Union (IoU) loss.
//!
//! Computes IoU loss for segmentation tasks. For each sample in the batch,
//! calculates IoU = intersection / union, then returns (1 - IoU) as loss.
//!
//! The IoU loss is computed as:
//! ```text
//! IoU = (pred ∩ target) / (pred ∪ target)
//! Loss = 1 - IoU
//! ```

use burn::{
    config::Config,
    module::{Content, DisplaySettings, Module, ModuleDisplay},
    nn::loss::Reduction,
    tensor::{Int, Tensor, backend::Backend},
};

/// Configuration for creating an [IoU loss](IoULoss).
#[derive(Config, Debug)]
pub struct IoULossConfig {
    /// Small epsilon value to avoid division by zero. Default: 1e-8
    #[config(default = 1e-8)]
    pub eps: f64,
}

impl IoULossConfig {
    /// Initialize [IoU loss](IoULoss).
    pub fn init(&self) -> IoULoss {
        self.assertions();
        IoULoss { eps: self.eps }
    }

    fn assertions(&self) {
        assert!(
            self.eps > 0.0,
            "Epsilon for IoULoss must be positive, got {}",
            self.eps
        );
    }
}

/// Intersection over Union (IoU) loss.
///
/// Calculates IoU loss for binary segmentation tasks.
/// Supports batch processing and reduction options.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct IoULoss {
    /// Small epsilon value to avoid division by zero.
    pub eps: f64,
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
    /// Create a new IoU loss with default configuration.
    pub fn new() -> Self {
        IoULossConfig::new().init()
    }

    /// Compute the criterion on the input tensor with reduction.
    ///
    /// # Shapes
    ///
    /// - predictions: `[batch_size, channels, height, width]`
    /// - targets: `[batch_size, channels, height, width]`
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
    /// - predictions: `[batch_size, channels, height, width]`
    /// - targets: `[batch_size, channels, height, width]`
    /// - output: `[batch_size]`
    pub fn forward_no_reduction<B: Backend>(
        &self,
        predictions: Tensor<B, 4>,
        targets: Tensor<B, 4, Int>,
    ) -> Tensor<B, 1> {
        self.assertions(&predictions, &targets);

        let targets_float = targets.float();
        let [batch_size, ..] = predictions.dims();

        // Vectorized computation for all batch elements
        let eps = self.eps;

        // Flatten spatial dimensions for each batch element: [B, C*H*W]
        let pred_flat = predictions.reshape([batch_size as i32, -1]);
        let target_flat = targets_float.reshape([batch_size as i32, -1]);

        // Compute intersection: sum over spatial dimensions [B]
        let intersection = (pred_flat.clone() * target_flat.clone()).sum_dim(1);

        // Compute union: pred + target - intersection [B]
        let union = pred_flat.sum_dim(1) + target_flat.sum_dim(1) - intersection.clone();

        // Compute IoU with epsilon for numerical stability [B, 1]
        let iou = intersection / (union + eps);

        // IoU loss is (1 - IoU) [B, 1] -> [B]
        let loss = Tensor::ones_like(&iou) - iou;
        loss.squeeze::<1>()
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
    use burn::tensor::{TensorData, Tolerance, Transaction};

    use super::*;
    use crate::tests::TestBackend;
    #[test]
    fn iou_loss_forward_perfect_overlap_returns_zero_loss() {
        let device = Default::default();
        let loss = IoULoss::new();

        // Perfect overlap: IoU = 1.0, Loss = 0.0
        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[1.0, 1.0], [1.0, 1.0]]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[1, 1], [1, 1]]]]),
            &device,
        );

        let result_mean = loss.forward(predictions.clone(), targets.clone(), Reduction::Mean);
        let result_no_reduction = loss.forward_no_reduction(predictions, targets);

        let [result_mean_data, result_no_reduction_data] = Transaction::default()
            .register(result_mean)
            .register(result_no_reduction)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        // Perfect IoU should give loss ≈ 0
        let expected_mean = TensorData::from([0.0]);
        result_mean_data.assert_approx_eq::<f32>(&expected_mean, Tolerance::relative(1e-6));

        let expected_no_reduction = TensorData::from([0.0]);
        result_no_reduction_data
            .assert_approx_eq::<f32>(&expected_no_reduction, Tolerance::relative(1e-6));
    }

    #[test]
    fn iou_loss_forward_no_overlap_returns_one_loss() {
        let device = Default::default();
        let loss = IoULoss::new();

        // No overlap: IoU = 0.0, Loss = 1.0
        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[1.0, 1.0], [0.0, 0.0]]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[0, 0], [1, 1]]]]),
            &device,
        );

        let result_mean = loss.forward(predictions.clone(), targets.clone(), Reduction::Mean);
        let result_no_reduction = loss.forward_no_reduction(predictions, targets);

        let [result_mean_data, result_no_reduction_data] = Transaction::default()
            .register(result_mean)
            .register(result_no_reduction)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        // No overlap should give loss = 1.0
        let expected_mean = TensorData::from([1.0]);
        result_mean_data.assert_approx_eq::<f32>(&expected_mean, Tolerance::relative(1e-6));

        let expected_no_reduction = TensorData::from([1.0]);
        result_no_reduction_data
            .assert_approx_eq::<f32>(&expected_no_reduction, Tolerance::relative(1e-6));
    }

    #[test]
    fn iou_loss_forward_partial_overlap_computes_correct_loss() {
        let device = Default::default();
        let loss = IoULoss::new();

        // Partial overlap: pred=[1,1,0,0], target=[1,0,1,0]
        // Intersection = 1, Union = 1+1+1 = 3, IoU = 1/3, Loss = 2/3
        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[1.0, 1.0], [0.0, 0.0]]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[1, 0], [1, 0]]]]),
            &device,
        );

        let result = loss.forward(predictions, targets, Reduction::Mean);

        // IoU = 1/3, Loss = 1 - 1/3 = 2/3
        let expected = TensorData::from([2.0 / 3.0]);
        result
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::relative(1e-6));
    }

    #[test]
    fn iou_loss_forward_batch_samples_processes_correctly() {
        let device = Default::default();
        let loss = IoULoss::new();

        // Batch of 2: one perfect, one no overlap
        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([
                [[[1.0, 1.0], [1.0, 1.0]]], // Perfect overlap
                [[[1.0, 1.0], [0.0, 0.0]]], // No overlap
            ]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([
                [[[1, 1], [1, 1]]], // Perfect overlap
                [[[0, 0], [1, 1]]], // No overlap
            ]),
            &device,
        );

        let result_mean = loss.forward(predictions.clone(), targets.clone(), Reduction::Mean);
        let result_sum = loss.forward(predictions.clone(), targets.clone(), Reduction::Sum);
        let result_no_reduction = loss.forward_no_reduction(predictions, targets);

        let [result_mean_data, result_sum_data, result_no_reduction_data] = Transaction::default()
            .register(result_mean)
            .register(result_sum)
            .register(result_no_reduction)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        // Loss: [0.0, 1.0], Mean = 0.5, Sum = 1.0
        let expected_mean = TensorData::from([0.5]);
        result_mean_data.assert_approx_eq::<f32>(&expected_mean, Tolerance::relative(1e-6));

        let expected_sum = TensorData::from([1.0]);
        result_sum_data.assert_approx_eq::<f32>(&expected_sum, Tolerance::relative(1e-6));

        let expected_no_reduction = TensorData::from([0.0, 1.0]);
        result_no_reduction_data
            .assert_approx_eq::<f32>(&expected_no_reduction, Tolerance::relative(1e-6));
    }

    #[test]
    fn iou_loss_with_epsilon_handles_zero_division() {
        let device = Default::default();
        let config = IoULossConfig::new().with_eps(1e-6);
        let loss = config.init();

        // All zeros case - should not crash due to division by zero
        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[0.0, 0.0], [0.0, 0.0]]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[0, 0], [0, 0]]]]),
            &device,
        );

        let result = loss.forward(predictions, targets, Reduction::Mean);

        // With epsilon, IoU = 0/(0+eps) = 0, Loss = 1
        let expected = TensorData::from([1.0]);
        result
            .into_data()
            .assert_approx_eq::<f32>(&expected, Tolerance::relative(1e-5));
    }

    #[test]
    fn iou_loss_auto_reduction_equals_mean_reduction() {
        let device = Default::default();
        let loss = IoULoss::new();

        let predictions = Tensor::<TestBackend, 4>::from_data(
            TensorData::from([[[[1.0, 0.5], [0.8, 0.2]]]]),
            &device,
        );
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[1, 0], [1, 0]]]]),
            &device,
        );

        let result_auto = loss.forward(predictions.clone(), targets.clone(), Reduction::Auto);
        let result_mean = loss.forward(predictions, targets, Reduction::Mean);

        let [result_auto_data, result_mean_data] = Transaction::default()
            .register(result_auto)
            .register(result_mean)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        result_auto_data.assert_approx_eq::<f32>(&result_mean_data, Tolerance::default());
    }

    #[test]
    #[should_panic = "Epsilon for IoULoss must be positive"]
    fn iou_loss_config_negative_epsilon_panics() {
        let _loss = IoULossConfig::new().with_eps(-1e-6).init();
    }

    #[test]
    #[should_panic = "Shape of predictions"]
    fn iou_loss_forward_mismatched_shapes_panics() {
        let device = Default::default();
        let loss = IoULoss::new();

        let predictions =
            Tensor::<TestBackend, 4>::from_data(TensorData::from([[[[1.0, 2.0]]]]), &device);
        let targets = Tensor::<TestBackend, 4, Int>::from_data(
            TensorData::from([[[[1, 2], [3, 4]]]]),
            &device,
        );

        let _result = loss.forward_no_reduction(predictions, targets);
    }

    #[test]
    fn iou_loss_display_shows_eps_parameter() {
        let config = IoULossConfig::new().with_eps(1e-6);
        let loss = config.init();

        assert_eq!(format!("{loss}"), "IoULoss {eps: 0.000001}");
    }
}
