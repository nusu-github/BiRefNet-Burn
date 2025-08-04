//! IoU (Intersection over Union) metric implementation for BiRefNet.
//!
//! This module implements the Intersection over Union metric used in BiRefNet
//! for evaluating segmentation performance.

use burn::{
    prelude::*,
    tensor::{backend::Backend, ElementConversion, Tensor},
    train::metric::{Metric, MetricEntry, MetricMetadata, Numeric},
};
use std::marker::PhantomData;

use crate::metrics::input::IoUInput;

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

    fn update_stats(&mut self, predictions: Tensor<B, 4>, targets: Tensor<B, 4>) {
        let preds_sigmoid = burn::tensor::activation::sigmoid(predictions.clone());
        let preds_binary = preds_sigmoid.greater_elem(self.threshold).int();
        let targets_binary = targets.greater_elem(0.5).int();
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
        self.update_stats(item.predictions.clone(), item.targets.clone());
        let value = self.iou_value();
        MetricEntry::new(self.name(), format!("{value:.5}"), format!("{value:.5}"))
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

/// Calculate IoU metric using a simple function interface.
pub fn calculate_iou<B: Backend>(
    predictions: Tensor<B, 4>,
    targets: Tensor<B, 4>,
    threshold: f32,
) -> f32 {
    let preds_sigmoid = burn::tensor::activation::sigmoid(predictions);
    let preds_binary = preds_sigmoid.greater_elem(threshold).float();
    let targets_binary = targets.greater_elem(0.5).float();

    let intersection = (preds_binary.clone() * targets_binary.clone())
        .sum()
        .into_scalar()
        .elem::<f32>();
    let union = (preds_binary.sum() + targets_binary.sum())
        .into_scalar()
        .elem::<f32>()
        - intersection;

    if union > 0.0 {
        intersection / union
    } else {
        1.0
    }
}
