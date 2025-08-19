//! Mean Absolute Error (L1) loss.
//!
//! Creates a criterion that measures the mean absolute error (MAE) between each element in
//! the input and target tensors.
//!
//! The unreduced loss can be described as:
//! `L = {l_1, ..., l_N}` where `l_n = |x_n - y_n|`
//!
//! When reduction is applied:
//! - `'mean'`: `mean(L)`
//! - `'sum'`: `sum(L)`
//! - `'none'`: returns `L` (no reduction)

use burn::{
    config::Config,
    module::{Content, DisplaySettings, Module, ModuleDisplay},
    nn::loss::Reduction,
    tensor::{backend::Backend, Tensor},
};

/// Configuration for creating a [Mean Absolute Error loss](MaeLoss).
#[derive(Config, Debug)]
pub struct MaeLossConfig {
    /// Weight factor for the loss. Default: 1.0
    #[config(default = 1.0)]
    pub weight: f64,
}

impl MaeLossConfig {
    /// Initialize [Mean Absolute Error loss](MaeLoss).
    pub fn init(&self) -> MaeLoss {
        self.assertions();
        MaeLoss {
            weight: self.weight,
        }
    }

    fn assertions(&self) {
        assert!(
            self.weight > 0.0,
            "Weight for MaeLoss must be positive, got {}",
            self.weight
        );
    }
}

/// Mean Absolute Error (L1) loss.
///
/// Calculates the mean absolute error between predictions and targets.
/// Supports arbitrary tensor dimensions and reduction options.
#[derive(Module, Clone, Debug)]
#[module(custom_display)]
pub struct MaeLoss {
    /// Weight factor applied to the loss.
    pub weight: f64,
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
    /// Create a new MAE loss with default configuration.
    pub fn new() -> Self {
        MaeLossConfig::new().init()
    }

    /// Compute the criterion on the input tensor with reduction.
    ///
    /// # Shapes
    ///
    /// - predictions: `[...dims]` (any shape)
    /// - targets: `[...dims]` (same shape as predictions)
    /// - output: `[1]`
    pub fn forward<const D: usize, B: Backend>(
        &self,
        predictions: Tensor<B, D>,
        targets: Tensor<B, D>,
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
    /// - predictions: `[...dims]` (any shape)
    /// - targets: `[...dims]` (same shape as predictions)
    /// - output: `[...dims]` (same shape as input)
    pub fn forward_no_reduction<const D: usize, B: Backend>(
        &self,
        predictions: Tensor<B, D>,
        targets: Tensor<B, D>,
    ) -> Tensor<B, D> {
        self.assertions(&predictions, &targets);

        // Compute absolute difference: |pred - target|
        (predictions - targets).abs()
    }

    fn assertions<const D: usize, B: Backend>(
        &self,
        predictions: &Tensor<B, D>,
        targets: &Tensor<B, D>,
    ) {
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
    use burn::{
        backend::NdArray,
        tensor::{ops::FloatElem, TensorData, Tolerance, Transaction},
    };

    use super::*;

    type TestBackend = NdArray;
    type FT = FloatElem<TestBackend>;

    #[test]
    fn mae_loss_forward_identical_tensors_returns_zero() {
        let device = Default::default();
        let loss = MaeLoss::new();

        let pred = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
            &device,
        );
        let target = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
            &device,
        );

        let result = loss.forward(pred.clone(), target.clone(), Reduction::Mean);
        let result_no_reduction = loss.forward_no_reduction(pred, target);

        let [result_data, result_no_reduction_data] = Transaction::default()
            .register(result)
            .register(result_no_reduction)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let expected = TensorData::from([0.0]);
        result_data.assert_approx_eq::<FT>(&expected, Tolerance::default());

        let expected_no_reduction = TensorData::from([[0.0, 0.0], [0.0, 0.0]]);
        result_no_reduction_data
            .assert_approx_eq::<FT>(&expected_no_reduction, Tolerance::default());
    }

    #[test]
    fn mae_loss_forward_different_tensors_computes_correct_mean() {
        let device = Default::default();
        let loss = MaeLoss::new();

        let pred = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[2.0, 3.0], [4.0, 5.0]]),
            &device,
        );
        let target = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 1.0], [1.0, 1.0]]),
            &device,
        );

        let result_mean = loss.forward(pred.clone(), target.clone(), Reduction::Mean);
        let result_sum = loss.forward(pred.clone(), target.clone(), Reduction::Sum);
        let result_no_reduction = loss.forward_no_reduction(pred, target);

        let [result_mean_data, result_sum_data, result_no_reduction_data] = Transaction::default()
            .register(result_mean)
            .register(result_sum)
            .register(result_no_reduction)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        // |2-1| + |3-1| + |4-1| + |5-1| = 1 + 2 + 3 + 4 = 10
        // Mean = 10/4 = 2.5
        let expected_mean = TensorData::from([2.5]);
        result_mean_data.assert_approx_eq::<FT>(&expected_mean, Tolerance::default());

        // Sum = 10
        let expected_sum = TensorData::from([10.0]);
        result_sum_data.assert_approx_eq::<FT>(&expected_sum, Tolerance::default());

        // No reduction: element-wise absolute differences
        let expected_no_reduction = TensorData::from([[1.0, 2.0], [3.0, 4.0]]);
        result_no_reduction_data
            .assert_approx_eq::<FT>(&expected_no_reduction, Tolerance::default());
    }

    #[test]
    fn mae_loss_with_custom_weight_multiplies_result() {
        let device = Default::default();
        let config = MaeLossConfig::new().with_weight(2.0);
        let loss = config.init();

        let pred = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[2.0, 1.0], [3.0, 0.0]]),
            &device,
        );
        let target = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 1.0], [1.0, 1.0]]),
            &device,
        );

        let result = loss.forward(pred, target, Reduction::Mean);

        // |2-1| + |1-1| + |3-1| + |0-1| = 1 + 0 + 2 + 1 = 4
        // Mean = 4/4 = 1, Weight = 2.0, Total = 2.0
        let expected = TensorData::from([2.0]);
        result
            .into_data()
            .assert_approx_eq::<FT>(&expected, Tolerance::default());
    }

    #[test]
    fn mae_loss_forward_different_tensor_dimensions_works() {
        let device = Default::default();
        let loss = MaeLoss::new();

        // Test 1D tensors
        let pred_1d =
            Tensor::<TestBackend, 1>::from_data(TensorData::from([2.0, 4.0, 6.0]), &device);
        let target_1d =
            Tensor::<TestBackend, 1>::from_data(TensorData::from([1.0, 2.0, 3.0]), &device);

        let result_1d = loss.forward(pred_1d, target_1d, Reduction::Mean);

        // |2-1| + |4-2| + |6-3| = 1 + 2 + 3 = 6, Mean = 6/3 = 2.0
        let expected_1d = TensorData::from([2.0]);
        result_1d
            .into_data()
            .assert_approx_eq::<FT>(&expected_1d, Tolerance::default());

        // Test 3D tensors
        let pred_3d = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([[[2.0, 4.0]], [[6.0, 8.0]]]),
            &device,
        );
        let target_3d = Tensor::<TestBackend, 3>::from_data(
            TensorData::from([[[1.0, 2.0]], [[3.0, 4.0]]]),
            &device,
        );

        let result_3d = loss.forward(pred_3d, target_3d, Reduction::Mean);

        // |2-1| + |4-2| + |6-3| + |8-4| = 1 + 2 + 3 + 4 = 10, Mean = 10/4 = 2.5
        let expected_3d = TensorData::from([2.5]);
        result_3d
            .into_data()
            .assert_approx_eq::<FT>(&expected_3d, Tolerance::default());
    }

    #[test]
    fn mae_loss_auto_reduction_equals_mean_reduction() {
        let device = Default::default();
        let loss = MaeLoss::new();

        let pred = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[2.0, 4.0], [6.0, 8.0]]),
            &device,
        );
        let target = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
            &device,
        );

        let result_auto = loss.forward(pred.clone(), target.clone(), Reduction::Auto);
        let result_mean = loss.forward(pred, target, Reduction::Mean);

        let [result_auto_data, result_mean_data] = Transaction::default()
            .register(result_auto)
            .register(result_mean)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        result_auto_data.assert_approx_eq::<FT>(&result_mean_data, Tolerance::default());
    }

    #[test]
    #[should_panic = "Weight for MaeLoss must be positive"]
    fn mae_loss_config_negative_weight_panics() {
        let _loss = MaeLossConfig::new().with_weight(-1.0).init();
    }

    #[test]
    #[should_panic = "Shape of predictions"]
    fn mae_loss_forward_mismatched_shapes_panics() {
        let device = Default::default();
        let loss = MaeLoss::new();

        let pred = Tensor::<TestBackend, 2>::from_data(TensorData::from([[1.0, 2.0]]), &device);
        let target = Tensor::<TestBackend, 2>::from_data(
            TensorData::from([[1.0, 2.0], [3.0, 4.0]]),
            &device,
        );

        let _result = loss.forward_no_reduction(pred, target);
    }

    #[test]
    fn mae_loss_display_shows_weight_parameter() {
        let config = MaeLossConfig::new().with_weight(0.5);
        let loss = config.init();

        assert_eq!(format!("{loss}"), "MaeLoss {weight: 0.5}");
    }
}
