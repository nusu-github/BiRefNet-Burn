//! Batch metric aggregator accumulating evaluation results across dataset batches.

use burn::tensor::Tensor;

use crate::{
    common::usize_to_f64,
    utils::{AllMetricsResult, calculate_all_metrics},
};

/// Stateful aggregator tracking running averages across evaluation batches.
#[derive(Debug, Clone, Default)]
pub struct MetricsAggregator {
    iou_sum: f64,
    f_measure_sum: f64,
    mae_sum: f64,
    mse_sum: f64,
    s_measure_sum: f64,
    e_measure_sum: f64,
    weighted_f_measure_sum: f64,
    e_measure_curve_sum: Vec<f64>,
    count: usize,
}

impl MetricsAggregator {
    /// Creates an empty [`MetricsAggregator`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            iou_sum: 0.0,
            f_measure_sum: 0.0,
            mae_sum: 0.0,
            mse_sum: 0.0,
            s_measure_sum: 0.0,
            e_measure_sum: 0.0,
            weighted_f_measure_sum: 0.0,
            e_measure_curve_sum: Vec::new(),
            count: 0,
        }
    }

    /// Evaluates one `[B, 1, H, W]` batch and accumulates sample-weighted sums.
    pub fn update(&mut self, predictions: &Tensor<4>, targets: &Tensor<4>, dilation_ratio: f64) {
        let [batch_size, ..] = predictions.dims();
        if batch_size == 0 {
            return;
        }

        let metrics = calculate_all_metrics(predictions, targets, dilation_ratio);
        let weight = usize_to_f64(batch_size);

        self.iou_sum += metrics.iou * weight;
        self.f_measure_sum += metrics.f_measure * weight;
        self.mae_sum += metrics.mae * weight;
        self.mse_sum += metrics.mse * weight;
        self.s_measure_sum += metrics.s_measure * weight;
        self.e_measure_sum += metrics.e_measure_adaptive * weight;
        self.weighted_f_measure_sum += metrics.weighted_f_measure * weight;

        if self.e_measure_curve_sum.is_empty() {
            self.e_measure_curve_sum = vec![0.0; metrics.e_measure_curve.len()];
        }
        for (dst, src) in self
            .e_measure_curve_sum
            .iter_mut()
            .zip(metrics.e_measure_curve.iter())
        {
            *dst += *src * weight;
        }

        self.count += batch_size;
    }

    /// Returns `(mean_biou, mean_f_measure, mean_mae)` across all accumulated samples.
    #[must_use]
    pub fn averages(&self) -> (f64, f64, f64) {
        if self.count == 0 {
            return (0.0, 0.0, 0.0);
        }

        let count = usize_to_f64(self.count);
        (
            self.iou_sum / count,
            self.f_measure_sum / count,
            self.mae_sum / count,
        )
    }

    /// Returns full averaged [`AllMetricsResult`] across all accumulated samples.
    #[must_use]
    pub fn results(&self) -> AllMetricsResult {
        if self.count == 0 {
            return AllMetricsResult {
                iou: 0.0,
                f_measure: 0.0,
                mae: 0.0,
                mse: 0.0,
                s_measure: 0.0,
                e_measure_adaptive: 0.0,
                e_measure_curve: vec![0.0; 256],
                weighted_f_measure: 0.0,
            };
        }

        let count = usize_to_f64(self.count);
        AllMetricsResult {
            iou: self.iou_sum / count,
            f_measure: self.f_measure_sum / count,
            mae: self.mae_sum / count,
            mse: self.mse_sum / count,
            s_measure: self.s_measure_sum / count,
            e_measure_adaptive: self.e_measure_sum / count,
            e_measure_curve: self
                .e_measure_curve_sum
                .iter()
                .map(|val| val / count)
                .collect(),
            weighted_f_measure: self.weighted_f_measure_sum / count,
        }
    }

    /// Clears all accumulated metric sums and sample counts.
    pub fn reset(&mut self) {
        self.iou_sum = 0.0;
        self.f_measure_sum = 0.0;
        self.mae_sum = 0.0;
        self.mse_sum = 0.0;
        self.s_measure_sum = 0.0;
        self.e_measure_sum = 0.0;
        self.weighted_f_measure_sum = 0.0;
        self.e_measure_curve_sum.clear();
        self.count = 0;
    }
}
