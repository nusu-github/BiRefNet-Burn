//! MAE (Mean Absolute Error) metric implementation for BiRefNet.
//!
//! This module implements the Mean Absolute Error metric used in BiRefNet
//! for evaluating segmentation performance.

use core::marker::PhantomData;

use burn::{
    prelude::*,
    tensor::{backend::Backend, cast::ToElement, Tensor},
    train::metric::{
        state::{FormatOptions, NumericMetricState},
        Metric, MetricEntry, MetricMetadata, Numeric,
    },
};

use super::input::MAEInput;

// --- MAE Metric ---

#[derive(Config, Debug)]
pub struct MAEMetricConfig {
    #[config(default = true)]
    pub apply_sigmoid: bool,
}

pub struct MAEMetric<B: Backend> {
    state: NumericMetricState,
    apply_sigmoid: bool,
    _b: PhantomData<B>,
}

impl MAEMetricConfig {
    pub fn init<B: Backend>(&self) -> MAEMetric<B> {
        MAEMetric {
            state: NumericMetricState::default(),
            apply_sigmoid: self.apply_sigmoid,
            _b: PhantomData,
        }
    }
}

impl<B: Backend> Default for MAEMetric<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> MAEMetric<B> {
    pub fn new() -> Self {
        MAEMetricConfig::new().init()
    }
}

impl<B: Backend> Metric for MAEMetric<B> {
    type Input = MAEInput<B>;

    fn name(&self) -> String {
        "MAE".to_owned()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let [batch_size, ..] = item.predictions.dims();

        let mut total_mae = 0.0;

        // Process each item in the batch
        for b in 0..batch_size {
            let pred: Tensor<B, 3> = item
                .predictions
                .clone()
                .slice(s![b..=b, .., .., ..])
                .squeeze(0);
            let gt: Tensor<B, 3> = item.targets.clone().slice(s![b..=b, .., .., ..]).squeeze(0);

            let mae = calculate_mae_single(pred, gt);
            total_mae += mae;
        }

        let avg_mae = total_mae / batch_size as f64;
        self.state.update(
            avg_mae,
            batch_size,
            FormatOptions::new(self.name()).precision(5),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl<B: Backend> Numeric for MAEMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

/// Calculates MAE for a single prediction-target pair.
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
/// The MAE value.
fn calculate_mae_single<B: Backend, const D: usize>(
    predictions: Tensor<B, D>,
    targets: Tensor<B, D>,
) -> f64 {
    // Prepare data following Python _prepare_data function
    let (pred, gt) = prepare_data(predictions, targets);

    // Calculate MAE: np.mean(np.abs(pred - gt))
    let abs_error = (pred - gt).abs().mean();
    abs_error.into_scalar().to_f64()
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
pub fn calculate_mae<B: Backend>(predictions: Tensor<B, 2>, targets: Tensor<B, 2>) -> f64 {
    calculate_mae_single(predictions, targets)
}
