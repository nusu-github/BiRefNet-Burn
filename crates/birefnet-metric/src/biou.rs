//! BIoU (Boundary IoU) metric implementation for BiRefNet.
//!
//! This module implements the Boundary Intersection over Union metric used in BiRefNet
//! for evaluating boundary-focused segmentation performance.

use core::marker::PhantomData;

use burn::{
    prelude::*,
    tensor::{backend::Backend, cast::ToElement, Tensor},
    train::metric::{
        state::{FormatOptions, NumericMetricState},
        Metric, MetricEntry, MetricMetadata, Numeric,
    },
};

use super::input::BIoUInput;

// --- BIoU Metric ---

/// Configuration for the BIoU metric.
#[derive(Config, Debug)]
pub struct BIoUMetricConfig {
    /// Dilation ratio for boundary extraction (default: 0.02).
    #[config(default = 0.02)]
    pub dilation_ratio: f64,
}

/// BIoU metric.
#[derive(Default)]
pub struct BIoUMetric<B: Backend> {
    state: NumericMetricState,
    dilation_ratio: f64,
    _b: PhantomData<B>,
}

impl<B: Backend> BIoUMetric<B> {
    /// Creates a new BIoU metric.

    pub fn new() -> Self {
        Self {
            state: NumericMetricState::default(),
            dilation_ratio: 0.02,
            _b: PhantomData,
        }
    }

    /// Creates a new BIoU metric with custom configuration.

    pub fn with_config(config: BIoUMetricConfig) -> Self {
        Self {
            state: NumericMetricState::default(),
            dilation_ratio: config.dilation_ratio,
            _b: PhantomData,
        }
    }
}

impl<B: Backend> Metric for BIoUMetric<B> {
    type Input = BIoUInput<B>;

    fn name(&self) -> String {
        "BIoU".to_owned()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let [batch_size, ..] = item.predictions.dims();

        let mut total_biou_curves = Vec::new();

        // Process each item in the batch
        for b in 0..batch_size {
            let pred: Tensor<B, 3> = item
                .predictions
                .clone()
                .slice(s![b..=b, .., .., ..])
                .squeeze(0);
            let gt: Tensor<B, 3> = item.targets.clone().slice(s![b..=b, .., .., ..]).squeeze(0);

            let biou_curve = calculate_biou_curve(pred, gt, self.dilation_ratio);
            total_biou_curves.push(biou_curve);
        }

        // Average the curves across the batch
        let avg_biou = average_biou_curves(total_biou_curves);

        self.state.update(
            avg_biou,
            batch_size,
            FormatOptions::new(self.name()).precision(5),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl<B: Backend> Numeric for BIoUMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

/// Calculates BIoU curve for a single prediction-target pair.
///
/// Implements the boundary IoU calculation from Python BiRefNet:
/// 1. Apply _prepare_data to normalize inputs
/// 2. Extract boundaries using mask_to_boundary
/// 3. Calculate IoU curves across thresholds
///
/// # Arguments
/// * `predictions` - Predictions with shape `[channels, height, width]` or `[height, width]`.
/// * `targets` - Ground truth with shape `[channels, height, width]` or `[height, width]`.
/// * `dilation_ratio` - Dilation ratio for boundary extraction.
///
/// # Returns
/// Vector of IoU values across thresholds (256 values).
fn calculate_biou_curve<B: Backend, const D: usize>(
    predictions: Tensor<B, D>,
    targets: Tensor<B, D>,
    dilation_ratio: f64,
) -> Vec<f64> {
    // Prepare data following Python _prepare_data function
    let (pred, gt) = prepare_data(predictions, targets);

    // Convert predictions to boundary representation
    let pred_boundary = mask_to_boundary(pred.mul_scalar(255.0), dilation_ratio);
    let gt_boundary = mask_to_boundary(gt.mul_scalar(255.0), dilation_ratio);

    // Calculate BIoU curves across 256 thresholds
    calculate_biou_histogram_curves(pred_boundary, gt_boundary)
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

/// Extracts boundary from mask using morphological operations.
///
/// Implements mask_to_boundary function from Python BiRefNet:
/// 1. Calculate dilation based on image diagonal and dilation_ratio
/// 2. Pad image to handle border effects
/// 3. Apply erosion to get inner mask
/// 4. Subtract to get boundary
fn mask_to_boundary<B: Backend, const D: usize>(
    mask: Tensor<B, D>,
    dilation_ratio: f64,
) -> Tensor<B, D> {
    let shape = mask.dims();
    let device = mask.device();

    // Calculate dilation based on image diagonal
    let h = shape[shape.len() - 2] as f64;
    let w = shape[shape.len() - 1] as f64;
    let img_diag = h.hypot(w);
    let dilation = (dilation_ratio * img_diag).round() as usize;
    let dilation = dilation.max(1);

    // TODO: Implement proper morphological operations for boundary extraction
    // Current: Simplified implementation using convolution approximation
    // Should implement: Proper erosion with disk kernel as in Python version

    // For now, use a simple approximation
    // Create a kernel for erosion (approximating disk kernel)
    let kernel_size = (dilation * 2 + 1).min(7); // Limit kernel size
    let kernel_data: Vec<f64> = (0..kernel_size * kernel_size).map(|_| 1.0).collect();
    let kernel_norm = 1.0 / (kernel_size * kernel_size) as f64;

    let kernel: Vec<f64> = kernel_data.iter().map(|&x| x * kernel_norm).collect();
    let kernel_tensor =
        Tensor::<B, 2>::from_data(kernel.as_slice(), &device).reshape([kernel_size, kernel_size]);

    // Apply erosion approximation using convolution
    let eroded = approximate_erosion(mask.clone(), kernel_tensor, dilation);

    // Boundary = original - eroded
    mask - eroded
}

/// Approximates erosion using convolution (placeholder implementation).
fn approximate_erosion<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    _kernel: Tensor<B, 2>,
    _iterations: usize,
) -> Tensor<B, D> {
    // TODO: Implement proper erosion
    // For now, just return a slightly smaller version by applying a threshold
    input.mul_scalar(0.95)
}

/// Calculates BIoU histogram curves across 256 thresholds.
///
/// Implements the histogram-based curve calculation from Python BIoUMeasure:
/// - Create 256 threshold bins from 0 to 255
/// - Calculate true positives and false positives for each threshold
/// - Compute IoU = TP / (T + FP) where T is total ground truth positives
fn calculate_biou_histogram_curves<B: Backend, const D: usize>(
    pred_boundary: Tensor<B, D>,
    gt_boundary: Tensor<B, D>,
) -> Vec<f64> {
    // Convert to uint8 range for histogram calculation
    let pred_u8 = pred_boundary.mul_scalar(255.0).clamp(0.0, 255.0);
    let gt_binary = gt_boundary.greater_elem(128);

    // Calculate histograms for 256 bins (0-255)
    let mut ious = Vec::with_capacity(256);

    for threshold in 0..256 {
        let pred_thresh = pred_u8.clone().greater_elem(f64::from(threshold));
        let tp = pred_thresh.clone().bool_and(gt_binary.clone()).int().sum();
        let fp = pred_thresh
            .bool_and(gt_binary.clone().bool_not())
            .int()
            .sum();
        let t = gt_binary.clone().int().sum();

        let tp_val = tp.into_scalar().to_f64();
        let fp_val = fp.into_scalar().to_f64();
        let t_val = t.into_scalar().to_f64().max(1.0); // Avoid division by zero

        let iou = tp_val / (t_val + fp_val);
        ious.push(iou);
    }

    ious
}

/// Averages BIoU curves across a batch.
fn average_biou_curves(curves: Vec<Vec<f64>>) -> f64 {
    if curves.is_empty() {
        return 0.0;
    }

    let curve_len = curves[0].len();
    let mut avg_curve = vec![0.0; curve_len];

    for curve in &curves {
        for (i, &val) in curve.iter().enumerate() {
            avg_curve[i] += val;
        }
    }

    let batch_size = curves.len() as f64;
    for val in &mut avg_curve {
        *val /= batch_size;
    }

    // Return the mean of the averaged curve
    avg_curve.iter().sum::<f64>() / curve_len as f64
}

/// Public function for external use.
pub fn calculate_biou<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    dilation_ratio: f64,
) -> f64 {
    let curve = calculate_biou_curve(predictions, targets, dilation_ratio);
    curve.iter().sum::<f64>() / curve.len() as f64
}
