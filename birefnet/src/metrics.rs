//! Metrics for BiRefNet training and evaluation.
//!
//! This module implements the evaluation metrics used in BiRefNet,
//! including F-measure, MAE (Mean Absolute Error), and other segmentation metrics.
//!
//! The implementation follows the original PyTorch BiRefNet evaluation metrics.

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

    fn clear(&mut self) {
        self.state.reset();
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

    fn clear(&mut self) {
        self.state = IoUState::default();
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

    fn clear(&mut self) {
        self.state.reset();
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let loss = item.loss.clone().into_scalar().elem::<f64>();
        self.state.update(
            loss,
            item.batch_size,
            FormatOptions::new(self.name()).precision(5),
        )
    }
}

impl<B: Backend> Numeric for LossMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}
