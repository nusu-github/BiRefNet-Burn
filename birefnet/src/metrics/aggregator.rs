//! Metrics aggregator for batch processing in BiRefNet.
//!
//! This module provides the MetricsAggregator struct which allows
//! for efficient accumulation and averaging of metrics across batches.

use burn::tensor::{backend::Backend, Tensor};

use crate::metrics::utils::calculate_all_metrics;

/// Metrics aggregator for batch processing.
#[derive(Debug, Clone)]
pub struct MetricsAggregator<B: Backend> {
    iou_sum: f32,
    f_measure_sum: f32,
    mae_sum: f32,
    count: usize,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> MetricsAggregator<B> {
    /// Create a new metrics aggregator.
    pub const fn new() -> Self {
        Self {
            iou_sum: 0.0,
            f_measure_sum: 0.0,
            mae_sum: 0.0,
            count: 0,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Add a batch of metrics.
    pub fn update(&mut self, predictions: Tensor<B, 4>, targets: Tensor<B, 4>, threshold: f32) {
        let all_metrics = calculate_all_metrics(predictions, targets, threshold);

        self.iou_sum += all_metrics.iou;
        self.f_measure_sum += all_metrics.f_measure;
        self.mae_sum += all_metrics.mae;
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

impl<B: Backend> Default for MetricsAggregator<B> {
    fn default() -> Self {
        Self::new()
    }
}
