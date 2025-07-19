//! Metrics for BiRefNet training and evaluation.
//!
//! This module implements the evaluation metrics used in BiRefNet,
//! including F-measure, MAE (Mean Absolute Error), and other segmentation metrics.
//!
//! The implementation follows the original PyTorch BiRefNet evaluation metrics.

use burn::tensor::cast::ToElement;
use burn::{
    prelude::*,
    tensor::{backend::Backend, ElementConversion, Tensor},
    train::metric::{
        state::{FormatOptions, NumericMetricState},
        Metric, MetricEntry, MetricMetadata, Numeric,
    },
};
use std::marker::PhantomData;
// --- Input Structs for Metrics ---

pub struct FMeasureInput<B: Backend> {
    pub predictions: Tensor<B, 4>,
    pub targets: Tensor<B, 4>,
}

pub struct MAEInput<B: Backend> {
    pub predictions: Tensor<B, 4>,
    pub targets: Tensor<B, 4>,
}

pub struct IoUInput<B: Backend> {
    pub predictions: Tensor<B, 4>,
    pub targets: Tensor<B, 4>,
}

pub struct BiRefNetLossInput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub batch_size: usize,
}

// --- F-measure Metric ---

#[derive(Config, Debug)]
pub struct FMeasureMetricConfig {
    #[config(default = 0.5)]
    pub threshold: f32,
}

#[derive(Debug, Clone)]
pub struct FMeasureMetric<B: Backend> {
    state: FMeasureState,
    threshold: f32,
    _b: PhantomData<B>,
}

#[derive(Debug, Clone, Default)]
struct FMeasureState {
    true_positives: f32,
    false_positives: f32,
    false_negatives: f32,
    count: usize,
}

impl FMeasureMetricConfig {
    pub fn init<B: Backend>(&self) -> FMeasureMetric<B> {
        FMeasureMetric {
            state: FMeasureState::default(),
            threshold: self.threshold,
            _b: PhantomData,
        }
    }
}

impl<B: Backend> Default for FMeasureMetric<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> FMeasureMetric<B> {
    pub fn new() -> Self {
        FMeasureMetricConfig::new().init()
    }

    fn update_stats(&mut self, predictions: &Tensor<B, 4>, targets: &Tensor<B, 4>) {
        let preds_sigmoid = burn::tensor::activation::sigmoid(predictions.clone());
        let preds_binary = preds_sigmoid.greater_elem(self.threshold).int();
        let targets_binary = targets.clone().greater_elem(0.5).int();

        let tp = (preds_binary.clone() * targets_binary.clone())
            .sum()
            .into_scalar()
            .elem::<f32>();
        let pred_positives = preds_binary.sum().into_scalar().elem::<f32>();
        let actual_positives = targets_binary.sum().into_scalar().elem::<f32>();

        let fp = pred_positives - tp;
        let fn_val = actual_positives - tp;

        self.state.true_positives += tp;
        self.state.false_positives += fp;
        self.state.false_negatives += fn_val;
        self.state.count += predictions.dims()[0];
    }

    fn fmeasure_value(&self) -> f32 {
        if self.state.count == 0 {
            return 0.0;
        }
        let tp = self.state.true_positives;
        let fp = self.state.false_positives;
        let fn_val = self.state.false_negatives;
        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fn_val > 0.0 {
            tp / (tp + fn_val)
        } else {
            0.0
        };
        if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        }
    }
}

impl<B: Backend> Metric for FMeasureMetric<B> {
    type Input = FMeasureInput<B>;

    fn name(&self) -> String {
        "F-measure".to_string()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        self.update_stats(&item.predictions, &item.targets);
        let value = self.fmeasure_value();
        MetricEntry::new(
            self.name(),
            format!("{:.5}", value),
            format!("{:.5}", value),
        )
    }

    fn clear(&mut self) {
        self.state = FMeasureState::default();
    }
}

impl<B: Backend> Numeric for FMeasureMetric<B> {
    fn value(&self) -> f64 {
        self.fmeasure_value() as f64
    }
}

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
        "MAE".to_string()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let batch_size = item.predictions.dims()[0];
        let preds_processed = if self.apply_sigmoid {
            burn::tensor::activation::sigmoid(item.predictions.clone())
        } else {
            item.predictions.clone()
        };
        let abs_error = (preds_processed - item.targets.clone()).abs().mean();
        let error_value = abs_error.into_scalar().elem::<f64>();
        self.state.update(
            error_value,
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

// --- IoU Metric ---

#[derive(Config, Debug)]
pub struct IoUMetricConfig {
    #[config(default = 0.5)]
    pub threshold: f32,
    #[config(default = 1e-6)]
    pub epsilon: f32,
}

#[derive(Debug, Clone)]
pub struct IoUMetric<B: Backend> {
    state: IoUState<B>,
    threshold: f32,
    epsilon: f32,
}

#[derive(Debug, Clone, Default)]
struct IoUState<B: Backend> {
    sum_intersection: f32,
    sum_union: f32,
    count: usize,
    _b: PhantomData<B>,
}

impl IoUMetricConfig {
    pub fn init<B: Backend>(&self) -> IoUMetric<B> {
        IoUMetric {
            state: IoUState::default(),
            threshold: self.threshold,
            epsilon: self.epsilon,
        }
    }
}

impl<B: Backend> Default for IoUMetric<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> IoUMetric<B> {
    pub fn new() -> Self {
        IoUMetricConfig::new().init()
    }

    fn update_stats(&mut self, predictions: &Tensor<B, 4>, targets: &Tensor<B, 4>) {
        let preds_sigmoid = burn::tensor::activation::sigmoid(predictions.clone());
        let preds_binary = preds_sigmoid.greater_elem(self.threshold).int();
        let targets_binary = targets.clone().greater_elem(0.5).int();
        let intersection = (preds_binary.clone() * targets_binary.clone())
            .sum()
            .into_scalar()
            .elem::<f32>();
        let union = (preds_binary.sum() + targets_binary.sum())
            .into_scalar()
            .elem::<f32>()
            - intersection;
        self.state.sum_intersection += intersection;
        self.state.sum_union += union;
        self.state.count += predictions.dims()[0];
    }

    fn iou_value(&self) -> f32 {
        if self.state.count == 0 {
            return 0.0;
        }
        let intersection = self.state.sum_intersection;
        let union = self.state.sum_union;
        if union > 0.0 {
            (intersection + self.epsilon) / (union + self.epsilon)
        } else {
            1.0
        }
    }
}

impl<B: Backend> Metric for IoUMetric<B> {
    type Input = IoUInput<B>;

    fn name(&self) -> String {
        "IoU".to_string()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        self.update_stats(&item.predictions, &item.targets);
        let value = self.iou_value();
        MetricEntry::new(
            self.name(),
            format!("{:.5}", value),
            format!("{:.5}", value),
        )
    }

    fn clear(&mut self) {
        self.state = IoUState::default();
    }
}

impl<B: Backend> Numeric for IoUMetric<B> {
    fn value(&self) -> f64 {
        self.iou_value() as f64
    }
}

// --- Loss Metric ---

#[derive(Default)]
pub struct LossMetric<B: Backend> {
    state: NumericMetricState,
    _b: PhantomData<B>,
}

impl<B: Backend> LossMetric<B> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<B: Backend> Metric for LossMetric<B> {
    type Input = BiRefNetLossInput<B>;

    fn name(&self) -> String {
        "Loss".to_string()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let loss = item.loss.clone().into_scalar().elem::<f64>();
        self.state.update(
            loss,
            item.batch_size,
            FormatOptions::new(self.name()).precision(5),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl<B: Backend> Numeric for LossMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

// --- Convenience Functions for Examples ---

/// Calculate IoU metric using a simple function interface.
pub fn calculate_iou<B: Backend>(
    predictions: Tensor<B, 4>,
    targets: Tensor<B, 4>,
    threshold: f32,
) -> f32
where
    B::FloatElem: From<f32> + Into<f32>,
{
    let preds_sigmoid = burn::tensor::activation::sigmoid(predictions);
    let preds_binary = preds_sigmoid.greater_elem(threshold).float();
    let targets_binary = targets.greater_elem(0.5).float();

    let intersection: f32 = (preds_binary.clone() * targets_binary.clone())
        .sum()
        .into_scalar()
        .into();
    let union: f32 = (preds_binary.sum() + targets_binary.sum())
        .into_scalar()
        .into()
        - intersection;

    if union > 0.0 {
        intersection / union
    } else {
        1.0
    }
}

/// Calculate F-measure metric using a simple function interface.
pub fn calculate_f_measure<B: Backend>(
    predictions: Tensor<B, 4>,
    targets: Tensor<B, 4>,
    threshold: f32,
) -> f32
where
    B::FloatElem: From<f32> + Into<f32>,
{
    let preds_sigmoid = burn::tensor::activation::sigmoid(predictions);
    let preds_binary = preds_sigmoid.greater_elem(threshold).float();
    let targets_binary = targets.greater_elem(0.5).float();

    let tp: f32 = (preds_binary.clone() * targets_binary.clone())
        .sum()
        .into_scalar()
        .to_f32();
    let pred_positives: f32 = preds_binary.sum().into_scalar().to_f32();
    let actual_positives: f32 = targets_binary.sum().into_scalar().to_f32();

    let fp = pred_positives - tp;
    let fn_val = actual_positives - tp;

    let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
    let recall = if tp + fn_val > 0.0 {
        tp / (tp + fn_val)
    } else {
        0.0
    };

    if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    }
}

/// Calculate MAE metric using a simple function interface.
pub fn calculate_mae<B: Backend>(
    predictions: Tensor<B, 4>,
    targets: Tensor<B, 4>,
    apply_sigmoid: bool,
) -> f32
where
    B::FloatElem: From<f32> + Into<f32>,
{
    let preds_processed = if apply_sigmoid {
        burn::tensor::activation::sigmoid(predictions)
    } else {
        predictions
    };

    let mae = (preds_processed - targets).abs().mean();
    mae.into_scalar().into()
}

/// Calculate all metrics at once.
pub fn calculate_all_metrics<B: Backend>(
    predictions: Tensor<B, 4>,
    targets: Tensor<B, 4>,
    threshold: f32,
) -> (f32, f32, f32)
where
    B::FloatElem: From<f32> + Into<f32>,
{
    let iou = calculate_iou(predictions.clone(), targets.clone(), threshold);
    let f_measure = calculate_f_measure(predictions.clone(), targets.clone(), threshold);
    let mae = calculate_mae(predictions, targets, true);

    (iou, f_measure, mae)
}

/// Metrics aggregator for batch processing.
#[derive(Debug, Clone)]
pub struct MetricsAggregator {
    iou_sum: f32,
    f_measure_sum: f32,
    mae_sum: f32,
    count: usize,
}

impl MetricsAggregator {
    /// Create a new metrics aggregator.
    pub const fn new() -> Self {
        Self {
            iou_sum: 0.0,
            f_measure_sum: 0.0,
            mae_sum: 0.0,
            count: 0,
        }
    }

    /// Add a batch of metrics.
    pub fn add_batch<B: Backend>(
        &mut self,
        predictions: Tensor<B, 4>,
        targets: Tensor<B, 4>,
        threshold: f32,
    ) where
        B::FloatElem: From<f32> + Into<f32>,
    {
        let (iou, f_measure, mae) = calculate_all_metrics(predictions, targets, threshold);

        self.iou_sum += iou;
        self.f_measure_sum += f_measure;
        self.mae_sum += mae;
        self.count += 1;
    }

    /// Get the average metrics.
    pub fn get_averages(&self) -> (f32, f32, f32) {
        if self.count == 0 {
            return (0.0, 0.0, 0.0);
        }

        let count = self.count as f32;
        (
            self.iou_sum / count,
            self.f_measure_sum / count,
            self.mae_sum / count,
        )
    }

    /// Reset the aggregator.
    pub const fn reset(&mut self) {
        self.iou_sum = 0.0;
        self.f_measure_sum = 0.0;
        self.mae_sum = 0.0;
        self.count = 0;
    }
}

impl Default for MetricsAggregator {
    fn default() -> Self {
        Self::new()
    }
}
