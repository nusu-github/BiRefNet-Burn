//! Enhanced-alignment Measure (E-measure) metric for BiRefNet.
//!
//! E-measure evaluates the alignment between prediction and ground truth,
//! considering both local and global similarities.

use std::sync::Arc;

use burn::tensor::TensorReadError;
use burn::{
    config::Config,
    prelude::*,
    tensor::{Bool, Tensor},
    train::metric::{
        Metric, MetricAttributes, MetricMetadata, Numeric, NumericAttributes, NumericEntry,
        SerializedEntry,
        state::{FormatOptions, NumericMetricState},
    },
};

/// Configuration for the E-measure metric.
#[derive(Config, Debug)]
pub struct EMeasureMetricConfig {
    /// Name of the metric (default: "E-measure").
    #[config(default = "String::from(\"E_measure\")")]
    name: String,
}

/// E-measure metric input.
#[derive(Debug, Clone)]
pub struct EMeasureInput {
    /// Predictions with shape `[batch_size, height, width]`.
    pub predictions: Tensor<3>,
    /// Ground truth with shape `[batch_size, height, width]`.
    pub targets: Tensor<3>,
}

impl EMeasureInput {
    /// Creates a new E-measure input.
    pub const fn new(predictions: Tensor<3>, targets: Tensor<3>) -> Self {
        Self {
            predictions,
            targets,
        }
    }
}

/// E-measure metric state.
#[derive(Default, Clone)]
pub struct EMeasureState {
    adaptive_ems: Vec<f64>,
    changeable_ems: Vec<Vec<f64>>,
}

/// E-measure metric.
#[derive(Clone)]
pub struct EMeasureMetric {
    state: EMeasureState,
    numeric_state: NumericMetricState,
    name: Arc<String>,
}

impl Default for EMeasureMetric {
    fn default() -> Self {
        Self {
            state: EMeasureState::default(),
            numeric_state: NumericMetricState::default(),
            name: Arc::new("E_measure".to_owned()),
        }
    }
}

impl EMeasureMetric {
    /// Creates a new E-measure metric.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new E-measure metric with custom configuration.
    pub fn with_config(config: EMeasureMetricConfig) -> Self {
        Self {
            state: EMeasureState::default(),
            numeric_state: NumericMetricState::default(),
            name: Arc::new(config.name),
        }
    }
}

impl Metric for EMeasureMetric {
    type Input = EMeasureInput;

    fn name(&self) -> Arc<String> {
        self.name.clone()
    }

    fn update(
        &mut self,
        input: &Self::Input,
        _metadata: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        let [batch_size, ..] = input.predictions.dims();

        for b in 0..batch_size {
            let pred = input
                .predictions
                .clone()
                .slice(s![b..=b, .., ..])
                .squeeze_dims::<2>(&[0, 1]);
            let gt = input
                .targets
                .clone()
                .slice(s![b..=b, .., ..])
                .squeeze_dims::<2>(&[0, 1]);

            let (adaptive_em, changeable_em) = calculate_e_measure(pred, gt);
            self.state.adaptive_ems.push(adaptive_em);
            self.state
                .changeable_ems
                .push(changeable_em.into_iter().collect());
        }

        // Update numeric state with adaptive E-measure
        let avg_adaptive_em =
            self.state.adaptive_ems.iter().sum::<f64>() / self.state.adaptive_ems.len() as f64;
        self.numeric_state.update(avg_adaptive_em, batch_size);
        Ok(self
            .numeric_state
            .compute_update(FormatOptions::new(self.name.clone()).precision(5)))
    }

    fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
        Ok(self
            .numeric_state
            .compute_final(FormatOptions::new(self.name.clone()).precision(5)))
    }

    fn attributes(&self) -> MetricAttributes {
        NumericAttributes {
            unit: None,
            higher_is_better: true,
        }
        .into()
    }

    fn clear(&mut self) {
        self.state = EMeasureState::default();
        self.numeric_state.reset();
    }
}

impl Numeric for EMeasureMetric {
    fn value(&self) -> Option<NumericEntry> {
        // Use the dedicated function to get E-measure results
        let (adaptive_em, _changeable_ems) = get_e_measure_results(&self.state);
        Some(NumericEntry::Value(adaptive_em))
    }

    fn running_value(&self) -> Option<NumericEntry> {
        Some(self.numeric_state.running_value())
    }

    fn final_value(&self) -> NumericEntry {
        self.numeric_state.final_value()
    }
}

/// Calculates E-measure for a single prediction-target pair.
///
/// # Arguments
/// * `predictions` - Predictions with shape `[height, width]`.
/// * `targets` - Ground truth with shape `[height, width]`.
///
/// # Returns
/// A tuple of (adaptive_em, changeable_em_curve).
pub fn calculate_e_measure(predictions: Tensor<2>, targets: Tensor<2>) -> (f64, Vec<f64>) {
    // Prepare data
    let gt = targets.div_scalar(255.0).greater_equal_elem(0.5);

    // Normalize predictions if needed
    let min_val = predictions.clone().min();
    let max_val = predictions.clone().max();
    let range = max_val - min_val.clone();
    let epsilon = 1e-8;

    let pred = if range.clone().greater_elem(epsilon).into_scalar::<bool>() {
        let min_scalar = min_val.into_scalar::<f64>();
        let range_scalar = range.into_scalar::<f64>();
        predictions.sub_scalar(min_scalar).div_scalar(range_scalar)
    } else {
        predictions
    };

    let [height, width] = gt.dims();

    let gt_size = (height * width) as f64;
    let gt_fg_numel = gt.clone().float().sum().into_scalar::<f64>();

    // Calculate adaptive E-measure
    let adaptive_threshold = get_adaptive_threshold(pred.clone());

    let adaptive_em: f64 = calculate_em_with_threshold(
        pred.clone(),
        gt.clone(),
        adaptive_threshold,
        gt_fg_numel,
        gt_size,
    );

    // Calculate changeable E-measure curve
    let changeable_em = calculate_em_with_cumsumhistogram(pred, gt, gt_fg_numel, gt_size);

    (adaptive_em, changeable_em)
}

fn get_adaptive_threshold(pred: Tensor<2>) -> f64 {
    let mean_val = pred.mean().into_scalar::<f64>();
    (2.0 * mean_val).min(1.0)
}

fn calculate_em_with_threshold(
    pred: Tensor<2>,
    gt: Tensor<2, Bool>,
    threshold: f64,
    gt_fg_numel: f64,
    gt_size: f64,
) -> f64 {
    let binarized_pred = pred.greater_equal_elem(threshold);

    let fg_fg_numel: f64 = binarized_pred
        .clone()
        .bool_and(gt.clone())
        .float()
        .sum()
        .into_scalar::<f64>();
    let fg_bg_numel: f64 = binarized_pred
        .bool_and(gt.bool_not())
        .float()
        .sum()
        .into_scalar::<f64>();

    let fg_numel = fg_fg_numel + fg_bg_numel;
    let bg_numel = gt_size - fg_numel;

    let enhanced_matrix_sum = if gt_fg_numel == 0.0 {
        bg_numel
    } else if gt_fg_numel == gt_size {
        fg_numel
    } else {
        let (parts_numel, combinations) = generate_parts_numel_combinations(
            fg_fg_numel,
            fg_bg_numel,
            fg_numel,
            bg_numel,
            gt_fg_numel,
            gt_size,
        );

        let mut results_parts = 0.0;
        for (part_numel, (pred_val, gt_val)) in parts_numel.iter().zip(combinations.iter()) {
            let align_matrix_value =
                2.0 * (pred_val * gt_val) / (pred_val * pred_val + gt_val * gt_val + 1e-8);
            let enhanced_matrix_value = (align_matrix_value + 1.0).powi(2) / 4.0;
            results_parts += enhanced_matrix_value * part_numel;
        }
        results_parts
    };

    enhanced_matrix_sum / (gt_size - 1.0 + 1e-8)
}

fn calculate_em_with_cumsumhistogram(
    pred: Tensor<2>,
    gt: Tensor<2, Bool>,
    gt_fg_numel: f64,
    gt_size: f64,
) -> Vec<f64> {
    // Scale predictions to 0-255 range
    let pred_scaled = (pred * 255.0).int();

    // Create histogram bins
    let num_bins = 256;
    let mut changeable_ems = vec![0.0; num_bins];

    // For each threshold
    for (threshold, changeable_em) in changeable_ems.iter_mut().enumerate() {
        let binarized_pred = pred_scaled.clone().greater_equal_elem(threshold as i32);

        let fg_fg_numel: f64 = binarized_pred
            .clone()
            .bool_and(gt.clone())
            .float()
            .sum()
            .into_scalar::<f64>();
        let fg_bg_numel: f64 = binarized_pred
            .bool_and(gt.clone().bool_not())
            .float()
            .sum()
            .into_scalar::<f64>();

        let fg_numel = fg_fg_numel + fg_bg_numel;
        let bg_numel = gt_size - fg_numel;

        let enhanced_matrix_sum = if gt_fg_numel == 0.0 {
            bg_numel
        } else if gt_fg_numel == gt_size {
            fg_numel
        } else {
            let (parts_numel, combinations) = generate_parts_numel_combinations(
                fg_fg_numel,
                fg_bg_numel,
                fg_numel,
                bg_numel,
                gt_fg_numel,
                gt_size,
            );

            let mut results_parts = 0.0;
            for (part_numel, (pred_val, gt_val)) in parts_numel.iter().zip(combinations.iter()) {
                let align_matrix_value =
                    2.0 * (pred_val * gt_val) / (pred_val * pred_val + gt_val * gt_val + 1e-8);
                let enhanced_matrix_value = (align_matrix_value + 1.0).powi(2) / 4.0;
                results_parts += enhanced_matrix_value * part_numel;
            }
            results_parts
        };

        *changeable_em = enhanced_matrix_sum / (gt_size - 1.0 + 1e-8);
    }

    changeable_ems
}

fn generate_parts_numel_combinations(
    fg_fg_numel: f64,
    fg_bg_numel: f64,
    pred_fg_numel: f64,
    pred_bg_numel: f64,
    gt_fg_numel: f64,
    gt_size: f64,
) -> (Vec<f64>, Vec<(f64, f64)>) {
    let bg_fg_numel = gt_fg_numel - fg_fg_numel;
    let bg_bg_numel = pred_bg_numel - bg_fg_numel;

    let parts_numel = vec![fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel];

    let mean_pred_value = pred_fg_numel / gt_size;
    let mean_gt_value = gt_fg_numel / gt_size;

    let demeaned_pred_fg_value = 1.0 - mean_pred_value;
    let demeaned_pred_bg_value = 0.0 - mean_pred_value;
    let demeaned_gt_fg_value = 1.0 - mean_gt_value;
    let demeaned_gt_bg_value = 0.0 - mean_gt_value;

    let combinations = vec![
        (demeaned_pred_fg_value, demeaned_gt_fg_value),
        (demeaned_pred_fg_value, demeaned_gt_bg_value),
        (demeaned_pred_bg_value, demeaned_gt_fg_value),
        (demeaned_pred_bg_value, demeaned_gt_bg_value),
    ];

    (parts_numel, combinations)
}

/// Gets the E-measure results.
pub fn get_e_measure_results(state: &EMeasureState) -> (f64, Vec<f64>) {
    let adaptive_em = state.adaptive_ems.iter().sum::<f64>() / state.adaptive_ems.len() as f64;

    // Average changeable EMs across all samples
    let num_bins = if state.changeable_ems.is_empty() {
        256
    } else {
        state.changeable_ems[0].len()
    };
    let mut avg_changeable_em = vec![0.0; num_bins];

    for changeable in &state.changeable_ems {
        for (i, &val) in changeable.iter().enumerate() {
            avg_changeable_em[i] += val;
        }
    }

    for val in &mut avg_changeable_em {
        *val /= state.changeable_ems.len() as f64;
    }

    (adaptive_em, avg_changeable_em)
}
