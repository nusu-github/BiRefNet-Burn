//! Threshold regularization loss to encourage binary predictions.
//!
//! This loss function encourages predictions to be close to either 0 or 1,
//! helping with binary segmentation tasks by discouraging intermediate values.
//!
//! The loss is computed as:
//! ```text
//! Loss = 1 - mean((pred - 0)² + (pred - 1)²)
//! ```
//!
//! This formulation gives maximum reward (loss = 0) when predictions are exactly 0 or 1,
//! and maximum penalty when predictions are around 0.5.

use burn::{
    config::Config,
    module::{Content, DisplaySettings, Module, ModuleDisplay},
    nn::loss::Reduction,
    tensor::{Tensor, backend::Backend},
};

/// Configuration for creating a [Threshold Regularization loss](ThresholdRegularizationLoss).
#[derive(Config, Debug)]
pub struct ThresholdRegularizationLossConfig {
    /// Weight factor applied to the regularization loss. Default: 1.0
    #[config(default = 1.0)]
    pub weight: f64,
}

impl ThresholdRegularizationLossConfig {
    /// Initialize [Threshold Regularization loss](ThresholdRegularizationLoss).
    pub fn init(&self) -> ThresholdRegularizationLoss {
        self.assertions();
        ThresholdRegularizationLoss {
            weight: self.weight,
        }
    }

    fn assertions(&self) {
        assert!(
            self.weight > 0.0,
            "Weight for ThresholdRegularizationLoss must be positive, got {}",
            self.weight
        );
    }
}

/// Threshold regularization loss to encourage binary predictions.
///
/// This loss function encourages predictions to be close to either 0 or 1,
/// which is particularly useful for binary segmentation tasks where
/// intermediate values are not desired.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct ThresholdRegularizationLoss {
    /// Weight factor applied to the regularization loss.
    pub weight: f64,
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
    /// Create a new threshold regularization loss with default configuration.
    pub fn new() -> Self {
        ThresholdRegularizationLossConfig::new().init()
    }

    /// Compute the criterion on the input tensor with reduction.
    ///
    /// # Shapes
    ///
    /// - predictions: `[...dims]` (any shape)
    /// - output: `[1]`
    pub fn forward<const D: usize, B: Backend>(
        &self,
        predictions: Tensor<B, D>,
        reduction: Reduction,
    ) -> Tensor<B, 1> {
        let loss = self.forward_no_reduction(predictions);
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
    /// - predictions: `[...dims]` (any shape)
    /// - output: `[...dims]` (same shape as input)
    pub fn forward_no_reduction<const D: usize, B: Backend>(
        &self,
        predictions: Tensor<B, D>,
    ) -> Tensor<B, D> {
        // Calculate threshold regularization: 1 - ((pred - 0)² + (pred - 1)²)
        // This encourages predictions to be close to 0 or 1 (binary values)
        // Maximum loss (0) when pred = 0 or pred = 1
        // Minimum loss (-1) when pred = 0.5
        let device = predictions.device();
        let dims = predictions.dims();

        let pred_sq = predictions.clone().powi_scalar(2.0); // (pred - 0)²
        let pred_minus_one = predictions - 1.0;
        let pred_minus_one_sq = pred_minus_one.powi_scalar(2.0); // (pred - 1)²

        // 1 - (pred² + (pred-1)²)
        Tensor::ones(dims, &device) - (pred_sq + pred_minus_one_sq)
    }
}

#[cfg(test)]
mod tests {
    use burn::{
        backend::Cpu,
        tensor::{TensorData, Tolerance, Transaction, ops::FloatElem},
    };

    use super::*;

    type TestBackend = Cpu;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn thr_reg_loss_forward_binary_values_returns_zero_loss() {
        let device = Default::default();
        let loss = ThresholdRegularizationLoss::new();

        // Perfect binary values (0 and 1) should give maximum reward (loss = 0)
        let predictions = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.0, 1.0], [1.0, 0.0]]),
            &device,
        );

        let result_mean = loss.forward(predictions.clone(), Reduction::Mean);
        let result_no_reduction = loss.forward_no_reduction(predictions);

        let [result_mean_data, result_no_reduction_data] = Transaction::default()
            .register(result_mean)
            .register(result_no_reduction)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        // For pred=0: 1 - (0² + (0-1)²) = 1 - (0 + 1) = 0
        // For pred=1: 1 - (1² + (1-1)²) = 1 - (1 + 0) = 0
        let expected_mean = TensorData::from([0.0]);
        result_mean_data.assert_approx_eq::<FT>(&expected_mean, Tolerance::default());

        let expected_no_reduction = TensorData::from([[0.0, 0.0], [0.0, 0.0]]);
        result_no_reduction_data
            .assert_approx_eq::<FT>(&expected_no_reduction, Tolerance::default());
    }

    #[test]
    fn thr_reg_loss_forward_intermediate_values_gives_maximum_penalty() {
        let device = Default::default();
        let loss = ThresholdRegularizationLoss::new();

        // Intermediate value (0.5) should give maximum penalty
        let predictions = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.5, 0.5], [0.5, 0.5]]),
            &device,
        );

        let result = loss.forward(predictions, Reduction::Mean);

        // For pred=0.5: 1 - (0.5² + (0.5-1)²) = 1 - (0.25 + 0.25) = 1 - 0.5 = 0.5
        let expected = TensorData::from([0.5]);
        result
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn thr_reg_loss_forward_mixed_binary_and_intermediate_computes_correct_mean() {
        let device = Default::default();
        let loss = ThresholdRegularizationLoss::new();

        // Mix of binary and intermediate values
        let predictions = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.0, 1.0], [0.5, 0.25]]),
            &device,
        );

        let result_no_reduction = loss.forward_no_reduction(predictions.clone());
        let result_mean = loss.forward(predictions, Reduction::Mean);

        let [result_no_reduction_data, result_mean_data] = Transaction::default()
            .register(result_no_reduction)
            .register(result_mean)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        // pred=0.0: 1 - (0² + 1²) = 0
        // pred=1.0: 1 - (1² + 0²) = 0
        // pred=0.5: 1 - (0.25 + 0.25) = 0.5
        // pred=0.25: 1 - (0.0625 + 0.5625) = 0.375
        let expected_no_reduction = TensorData::from([[0.0, 0.0], [0.5, 0.375]]);
        result_no_reduction_data
            .assert_approx_eq::<FT>(&expected_no_reduction, Tolerance::default());

        // Mean: (0 + 0 + 0.5 + 0.375) / 4 = 0.21875
        let expected_mean = TensorData::from([0.21875]);
        result_mean_data.assert_approx_eq::<FT>(&expected_mean, Tolerance::default());
    }

    #[test]
    fn thr_reg_loss_auto_mean_and_sum_reductions_work_correctly() {
        let device = Default::default();
        let loss = ThresholdRegularizationLoss::new();

        let predictions = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[0.5, 0.5], [0.5, 0.5]]),
            &device,
        );

        let result_auto = loss.forward(predictions.clone(), Reduction::Auto);
        let result_mean = loss.forward(predictions.clone(), Reduction::Mean);
        let result_sum = loss.forward(predictions, Reduction::Sum);

        let [result_auto_data, result_mean_data, result_sum_data] = Transaction::default()
            .register(result_auto)
            .register(result_mean)
            .register(result_sum)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        // Auto should equal Mean
        result_auto_data.assert_approx_eq::<FT>(&result_mean_data, Tolerance::default());

        // Sum should be Mean * num_elements = 0.5 * 4 = 2.0
        let expected_sum = TensorData::from([2.0]);
        result_sum_data.assert_approx_eq::<FT>(&expected_sum, Tolerance::default());
    }

    #[test]
    fn thr_reg_loss_with_custom_weight_multiplies_default_result() {
        let device = Default::default();
        let config = ThresholdRegularizationLossConfig::new().with_weight(2.0);
        let loss = config.init();

        let predictions =
            Tensor::<TestBackend, 2>::from_data(TensorData::from([[0.5, 0.5]]), &device);

        let result = loss.forward(predictions.clone(), Reduction::Mean);

        // Compare with default weight
        let default_loss = ThresholdRegularizationLoss::new();
        let default_result = default_loss.forward(predictions, Reduction::Mean);
        let expected = default_result * 2.0;

        let [result_data, expected_data] = Transaction::default()
            .register(result)
            .register(expected)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        result_data.assert_approx_eq::<FT>(&expected_data, Tolerance::default());
    }

    #[test]
    fn thr_reg_loss_forward_different_tensor_dimensions_works() {
        let device = Default::default();
        let loss = ThresholdRegularizationLoss::new();

        // Test 1D tensor
        let pred_1d =
            Tensor::<TestBackend, 1>::from_data(TensorData::from([0.0, 0.5, 1.0]), &device);
        let result_1d = loss.forward(pred_1d, Reduction::Mean);

        // (0 + 0.5 + 0) / 3 = 0.1667
        let expected_1d = TensorData::from([1.0 / 6.0]);
        result_1d
            .into_data()
            .assert_approx_eq::<FT>(&expected_1d, Tolerance::relative(1e-4));

        // Test 3D tensor
        let pred_3d = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([[[0.0, 1.0]], [[0.5, 0.25]]]),
            &device,
        );
        let result_3d = loss.forward(pred_3d, Reduction::Mean);

        // (0 + 0 + 0.5 + 0.375) / 4 = 0.21875
        let expected_3d = TensorData::from([0.21875]);
        result_3d
            .into_data()
            .assert_approx_eq::<FT>(&expected_3d, Tolerance::default());
    }

    #[test]
    #[should_panic = "Weight for ThresholdRegularizationLoss must be positive"]
    fn thr_reg_loss_config_negative_weight_panics() {
        let _loss = ThresholdRegularizationLossConfig::new()
            .with_weight(-1.0)
            .init();
    }

    #[test]
    fn thr_reg_loss_forward_values_outside_range_handles_correctly() {
        let device = Default::default();
        let loss = ThresholdRegularizationLoss::new();

        // Test values outside [0,1] range
        let predictions = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[-0.5, 1.5], [2.0, -1.0]]),
            &device,
        );

        let result = loss.forward(predictions, Reduction::Mean);

        // pred=-0.5: 1 - (0.25 + 2.25) = -1.5
        // pred=1.5: 1 - (2.25 + 0.25) = -1.5
        // pred=2.0: 1 - (4 + 1) = -4
        // pred=-1.0: 1 - (1 + 4) = -4
        // Mean = (-1.5 - 1.5 - 4 - 4) / 4 = -2.75
        let expected = TensorData::from([-2.75]);
        result
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn thr_reg_loss_display_shows_weight_parameter() {
        let config = ThresholdRegularizationLossConfig::new().with_weight(0.5);
        let loss = config.init();

        assert_eq!(
            format!("{loss}"),
            "ThresholdRegularizationLoss {weight: 0.5}"
        );
    }
}
