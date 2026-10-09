//! Metrics aggregator for batch processing in BiRefNet.
//!
//! This module provides the MetricsAggregator struct which allows
//! for efficient accumulation and averaging of metrics across batches.

use burn::tensor::Tensor;

use super::utils::calculate_all_metrics;

/// Metrics aggregator for batch processing.
#[derive(Debug, Clone)]
pub struct MetricsAggregator {
    iou_sum: f64,
    f_measure_sum: f64,
    mae_sum: f64,
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
    pub fn update(&mut self, predictions: Tensor<4>, targets: Tensor<4>, threshold: f64) {
        let all_metrics = calculate_all_metrics(predictions, targets, threshold);

        self.iou_sum += all_metrics.iou;
        self.f_measure_sum += all_metrics.f_measure;
        self.mae_sum += all_metrics.mae;
        self.count += 1;
    }

    /// Get the average metrics.
    pub fn get_averages(&self) -> (f64, f64, f64) {
        if self.count == 0 {
            return (0.0, 0.0, 0.0);
        }

        let count = self.count as f64;
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
