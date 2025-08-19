//! F-measure metric implementation for BiRefNet.
//!
//! This module implements the F-measure metric used in BiRefNet
//! for evaluating segmentation performance, including both adaptive
//! and changeable F-measure calculations as per the original Python implementation.

use core::marker::PhantomData;

use burn::{
    prelude::*,
    tensor::{backend::Backend, cast::ToElement, Tensor},
    train::metric::{
        state::{FormatOptions, NumericMetricState},
        Metric, MetricEntry, MetricMetadata, Numeric,
    },
};

use super::input::FMeasureInput;

// --- F-measure Metric ---

/// Configuration for the F-measure metric.
#[derive(Config, Debug)]
pub struct FMeasureMetricConfig {
    /// Beta parameter for F-measure (default: 0.3 as in Python implementation).
    #[config(default = 0.3)]
    pub beta: f64,
}

/// F-measure metric.
#[derive(Default)]
pub struct FMeasureMetric<B: Backend> {
    state: NumericMetricState,
    beta: f64,
    adaptive_fms: Vec<f64>,
    precision_curves: Vec<Vec<f64>>,
    recall_curves: Vec<Vec<f64>>,
    changeable_fms: Vec<Vec<f64>>,
    _b: PhantomData<B>,
}

impl<B: Backend> FMeasureMetric<B> {
    /// Creates a new F-measure metric with default beta=0.3.
    pub fn new() -> Self {
        Self {
            state: NumericMetricState::default(),
            beta: 0.3,
            adaptive_fms: Vec::new(),
            precision_curves: Vec::new(),
            recall_curves: Vec::new(),
            changeable_fms: Vec::new(),
            _b: PhantomData,
        }
    }

    /// Creates a new F-measure metric with custom beta parameter.
    pub fn with_beta(beta: f64) -> Self {
        Self {
            state: NumericMetricState::default(),
            beta,
            adaptive_fms: Vec::new(),
            precision_curves: Vec::new(),
            recall_curves: Vec::new(),
            changeable_fms: Vec::new(),
            _b: PhantomData,
        }
    }

    /// Calculates the current average adaptive F-measure.
    fn adaptive_fm_value(&self) -> f64 {
        if self.adaptive_fms.is_empty() {
            return 0.0;
        }
        self.adaptive_fms.iter().sum::<f64>() / self.adaptive_fms.len() as f64
    }
}

impl<B: Backend> Metric for FMeasureMetric<B> {
    type Input = FMeasureInput<B>;

    fn name(&self) -> String {
        "F-measure".to_owned()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let [batch_size, ..] = item.predictions.dims();

        // Process each item in the batch
        for b in 0..batch_size {
            let pred: Tensor<B, 3> = item
                .predictions
                .clone()
                .slice(s![b..=b, .., .., ..])
                .squeeze(0);
            let gt: Tensor<B, 3> = item.targets.clone().slice(s![b..=b, .., .., ..]).squeeze(0);

            // Calculate adaptive F-measure
            let adaptive_fm = calculate_adaptive_fm(pred.clone(), gt.clone(), self.beta);
            self.adaptive_fms.push(adaptive_fm);

            // Calculate precision-recall curves and changeable F-measure
            let (precisions, recalls, changeable_fm) = calculate_pr_curves(pred, gt, self.beta);
            self.precision_curves.push(precisions);
            self.recall_curves.push(recalls);
            self.changeable_fms.push(changeable_fm);
        }

        let adaptive_value = self.adaptive_fm_value();
        self.state.update(
            adaptive_value,
            batch_size,
            FormatOptions::new(self.name()).precision(5),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
        self.adaptive_fms.clear();
        self.precision_curves.clear();
        self.recall_curves.clear();
        self.changeable_fms.clear();
    }
}

impl<B: Backend> Numeric for FMeasureMetric<B> {
    fn value(&self) -> f64 {
        self.adaptive_fm_value()
    }
}

/// Calculates adaptive F-measure for a single prediction-target pair.
///
/// Implements adaptive F-measure calculation from Python BiRefNet:
/// 1. Calculate adaptive threshold as min(2 * mean(pred), 1.0)
/// 2. Binarize prediction using adaptive threshold
/// 3. Calculate precision and recall on foreground regions
/// 4. Calculate F-measure using beta parameter
fn calculate_adaptive_fm<B: Backend, const D: usize>(
    predictions: Tensor<B, D>,
    targets: Tensor<B, D>,
    beta: f64,
) -> f64 {
    // Prepare data following Python _prepare_data function
    let (pred, gt) = prepare_data(predictions, targets);

    // Calculate adaptive threshold: min(2 * pred.mean(), 1.0)
    let adaptive_threshold = (2.0 * pred.clone().mean().into_scalar().to_f64()).min(1.0);

    // Binarize prediction using adaptive threshold
    let binary_pred = pred.greater_equal_elem(adaptive_threshold);

    // Calculate area intersection (true positives on foreground regions)
    let gt_bool = gt.greater_elem(0.5);
    let intersection_count = binary_pred
        .clone()
        .bool_and(gt_bool.clone())
        .int()
        .sum()
        .into_scalar()
        .to_f64();

    if intersection_count == 0.0 {
        return 0.0;
    }

    // Calculate precision and recall
    let pred_positive_count = binary_pred.int().sum().into_scalar().to_f64().max(1.0);
    let gt_positive_count = gt_bool.int().sum().into_scalar().to_f64().max(1.0);

    let precision = intersection_count / pred_positive_count;
    let recall = intersection_count / gt_positive_count;

    // Calculate F-measure with beta weighting: (1 + beta) * precision * recall / (beta * precision + recall)
    (1.0 + beta) * precision * recall / beta.mul_add(precision, recall)
}

/// Calculates precision-recall curves and changeable F-measure.
///
/// Implements the histogram-based PR calculation from Python BiRefNet:
/// - Create histograms for foreground and background pixels across 256 thresholds
/// - Calculate cumulative true positives and false positives
/// - Compute precision, recall, and F-measure curves
fn calculate_pr_curves<B: Backend, const D: usize>(
    predictions: Tensor<B, D>,
    targets: Tensor<B, D>,
    beta: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Prepare data following Python _prepare_data function
    let (pred, gt) = prepare_data(predictions, targets);

    // Convert predictions to uint8 range for histogram calculation
    let pred_u8 = (pred * 255.0).clamp(0.0, 255.0);
    let gt_bool = gt.greater_elem(0.5);

    let mut precisions = Vec::with_capacity(256);
    let mut recalls = Vec::with_capacity(256);
    let mut changeable_fms = Vec::with_capacity(256);

    // Calculate metrics for each threshold (from 255 down to 0)
    for threshold in (0..256).rev() {
        let pred_thresh = pred_u8.clone().greater_equal_elem(f64::from(threshold));

        let tp = pred_thresh
            .clone()
            .bool_and(gt_bool.clone())
            .int()
            .sum()
            .into_scalar()
            .to_f64();
        let pred_positives = pred_thresh.int().sum().into_scalar().to_f64().max(1.0);
        let gt_positives = gt_bool.clone().int().sum().into_scalar().to_f64().max(1.0);

        let precision = tp / pred_positives;
        let recall = tp / gt_positives;

        // Calculate F-measure with beta parameter
        let f_measure = if precision + recall > 0.0 {
            (1.0 + beta) * precision * recall / beta.mul_add(precision, recall)
        } else {
            0.0
        };

        precisions.push(precision);
        recalls.push(recall);
        changeable_fms.push(f_measure);
    }

    (precisions, recalls, changeable_fms)
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
pub fn calculate_f_measure<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    beta: f64,
) -> f64 {
    calculate_adaptive_fm(predictions, targets, beta)
}
