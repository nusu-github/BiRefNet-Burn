//! Mean Squared Error (MSE) metric for BiRefNet.

use core::marker::PhantomData;
use std::sync::Arc;

use burn::{
    config::Config,
    tensor::{Tensor, backend::Backend, cast::ToElement, s},
    train::metric::{
        Metric, MetricMetadata, Numeric, NumericEntry,
        state::{FormatOptions, NumericMetricState},
    },
};

use super::input::MSEInput;

/// Configuration for the MSE metric.
#[derive(Config, Debug)]
pub struct MSEMetricConfig {
    /// Name of the metric (default: "MSE").
    #[config(default = "String::from(\"MSE\")")]
    name: String,
}

/// MSE metric.
#[derive(Default, Clone)]
pub struct MSEMetric<B: Backend> {
    state: NumericMetricState,
    name: Arc<String>,
    _backend: PhantomData<B>,
}

impl<B: Backend> MSEMetric<B> {
    /// Creates a new MSE metric.
    pub fn new() -> Self {
        Self {
            state: NumericMetricState::default(),
            name: Arc::new("MSE".to_owned()),
            _backend: PhantomData,
        }
    }

    /// Creates a new MSE metric with a custom name.
    pub fn with_name(name: String) -> Self {
        Self {
            state: NumericMetricState::default(),
            name: Arc::new(name),
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Metric for MSEMetric<B> {
    type Input = MSEInput<B>;

    fn name(&self) -> Arc<String> {
        self.name.clone()
    }

    fn update(
        &mut self,
        input: &Self::Input,
        _metadata: &MetricMetadata,
    ) -> burn::train::metric::SerializedEntry {
        let [batch_size, ..] = input.predictions.dims();

        let mut total_mse = 0.0;

        // Process each item in the batch
        for b in 0..batch_size {
            let pred: Tensor<B, 3> = input
                .predictions
                .clone()
                .slice(s![b..=b, .., .., ..])
                .squeeze();
            let gt: Tensor<B, 3> = input.targets.clone().slice(s![b..=b, .., .., ..]).squeeze();

            let mse = calculate_mse_single(pred, gt);
            total_mse += mse;
        }

        let avg_mse = total_mse / batch_size as f64;
        self.state.update(
            avg_mse,
            batch_size,
            FormatOptions::new(self.name.clone()).precision(5),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl<B: Backend> Numeric for MSEMetric<B> {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}

/// Calculates MSE for a single prediction-target pair.
///
/// Implements the _prepare_data logic from Python BiRefNet:
/// - gt = gt > 128 (threshold at 128)
/// - pred = pred / 255 (normalize to [0, 1])
/// - if pred.max() != pred.min(): pred = (pred - pred.min()) / (pred.max() - pred.min())
///
/// # Arguments
/// * `predictions` - Predictions with shape `[channels, height, width]` or `[height, width]`.
/// * `targets` - Ground truth with shape `[channels, height, width]` or `[height, width]`.
///
/// # Returns
/// The MSE value.
fn calculate_mse_single<B: Backend, const D: usize>(
    predictions: Tensor<B, D>,
    targets: Tensor<B, D>,
) -> f64 {
    // Prepare data following Python _prepare_data function
    let (pred, gt) = prepare_data(predictions, targets);

    // Calculate MSE: np.mean((pred - gt) ** 2)
    let diff = pred - gt;
    let squared_diff = diff.powf_scalar(2.0);
    let mse = squared_diff.mean();

    mse.into_scalar().to_f64()
}

/// Prepares prediction and ground truth data following Python _prepare_data logic.
fn prepare_data<B: Backend, const D: usize>(
    pred: Tensor<B, D>,
    gt: Tensor<B, D>,
) -> (Tensor<B, D>, Tensor<B, D>) {
    // gt = gt > 128 (binary ground truth)
    let gt_binary = gt.greater_elem(128).float();

    // pred = pred / 255 (normalize predictions to [0, 1])
    let pred_norm = pred.div_scalar(255.0);

    // if pred.max() != pred.min(): pred = (pred - pred.min()) / (pred.max() - pred.min())
    let pred_min = pred_norm.clone().min();
    let pred_max = pred_norm.clone().max();
    let range = pred_max - pred_min.clone();

    let pred_final = if range.clone().greater_elem(1e-8).into_scalar().to_bool() {
        // Normalize to [0, 1] if there's variation
        let pred_min_scalar = pred_min.into_scalar().to_f64();
        let range_scalar = range.into_scalar().to_f64();
        (pred_norm - pred_min_scalar) / range_scalar
    } else {
        // Use as-is if all values are the same
        pred_norm
    };

    (pred_final, gt_binary)
}

/// Public function for external use.
pub fn calculate_mse<B: Backend>(predictions: Tensor<B, 2>, targets: Tensor<B, 2>) -> f64 {
    calculate_mse_single(predictions, targets)
}
