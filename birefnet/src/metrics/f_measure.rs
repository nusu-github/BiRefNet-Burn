//! F-measure metric implementation for BiRefNet.
//!
//! This module implements the F-measure (F1-score) metric used in BiRefNet
//! for evaluating segmentation performance.

use burn::{
    prelude::*,
    tensor::{backend::Backend, ElementConversion, Tensor},
    train::metric::{Metric, MetricEntry, MetricMetadata, Numeric},
};
use std::marker::PhantomData;

use crate::metrics::input::FMeasureInput;

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

    fn update_stats(&mut self, predictions: Tensor<B, 4>, targets: Tensor<B, 4>) {
        let preds_sigmoid = burn::tensor::activation::sigmoid(predictions.clone());
        let preds_binary = preds_sigmoid.greater_elem(self.threshold).int();
        let targets_binary = targets.greater_elem(0.5).int();

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
        self.update_stats(item.predictions.clone(), item.targets.clone());
        let value = self.fmeasure_value();
        MetricEntry::new(self.name(), format!("{value:.5}"), format!("{value:.5}"))
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

/// Calculate F-measure metric using a simple function interface.
pub fn calculate_f_measure<B: Backend>(
    predictions: Tensor<B, 4>,
    targets: Tensor<B, 4>,
    threshold: f32,
) -> f32 {
    let preds_sigmoid = burn::tensor::activation::sigmoid(predictions);
    let preds_binary = preds_sigmoid.greater_elem(threshold).float();
    let targets_binary = targets.greater_elem(0.5).float();

    let tp = (preds_binary.clone() * targets_binary.clone())
        .sum()
        .into_scalar()
        .elem::<f32>();
    let pred_positives = preds_binary.sum().into_scalar().elem::<f32>();
    let actual_positives = targets_binary.sum().into_scalar().elem::<f32>();

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
