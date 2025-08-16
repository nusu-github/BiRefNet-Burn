//! Enhanced-alignment Measure (E-measure) metric for BiRefNet.
//!
//! E-measure evaluates the alignment between prediction and ground truth,
//! considering both local and global similarities.

use core::marker::PhantomData;

use burn::{
    config::Config,
    prelude::*,
    tensor::{backend::Backend, cast::ToElement, Bool, Tensor},
    train::metric::{
        state::{FormatOptions, NumericMetricState},
        Metric, MetricEntry, MetricMetadata, Numeric,
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
pub struct EMeasureInput<B: Backend> {
    /// Predictions with shape `[batch_size, height, width]`.
    pub predictions: Tensor<B, 3>,
    /// Ground truth with shape `[batch_size, height, width]`.
    pub targets: Tensor<B, 3>,
}

impl<B: Backend> EMeasureInput<B> {
    /// Creates a new E-measure input.
    pub const fn new(predictions: Tensor<B, 3>, targets: Tensor<B, 3>) -> Self {
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
pub struct EMeasureMetric<B: Backend> {
    state: EMeasureState,
    numeric_state: NumericMetricState,
    name: String,
    _backend: PhantomData<B>,
}

impl<B: Backend> Default for EMeasureMetric<B> {
    fn default() -> Self {
        Self {
            state: EMeasureState::default(),
            numeric_state: NumericMetricState::default(),
            name: "E_measure".to_owned(),
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> EMeasureMetric<B> {
    /// Creates a new E-measure metric.

    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new E-measure metric with custom configuration.

    pub fn with_config(config: EMeasureMetricConfig) -> Self {
        Self {
            state: EMeasureState::default(),
            numeric_state: NumericMetricState::default(),
            name: config.name,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Metric for EMeasureMetric<B> {
    type Input = EMeasureInput<B>;

    fn name(&self) -> String {
        self.name.to_string()
    }

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let [batch_size, ..] = input.predictions.dims();

        for b in 0..batch_size {
            let pred = input
                .predictions
                .clone()
                .slice(s![b..=b, .., ..])
                .squeeze(0);
            let gt = input.targets.clone().slice(s![b..=b, .., ..]).squeeze(0);

            let (adaptive_em, changeable_em) = calculate_e_measure(pred, gt);
            self.state.adaptive_ems.push(adaptive_em);
            self.state
                .changeable_ems
                .push(changeable_em.into_iter().collect());
        }

        // Update numeric state with adaptive E-measure
        let avg_adaptive_em =
            self.state.adaptive_ems.iter().sum::<f64>() / self.state.adaptive_ems.len() as f64;
        self.numeric_state.update(
            avg_adaptive_em,
            batch_size,
            FormatOptions::new(self.name.to_string()).precision(5),
        )
    }

    fn clear(&mut self) {
        self.state = EMeasureState::default();
        self.numeric_state.reset();
    }
}

impl<B: Backend> Numeric for EMeasureMetric<B> {
    fn value(&self) -> f64 {
        // Use the dedicated function to get E-measure results
        let (adaptive_em, _changeable_ems) = get_e_measure_results(&self.state);
        adaptive_em
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
pub fn calculate_e_measure<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
) -> (f64, Vec<f64>) {
    // Prepare data
    let gt = targets.div_scalar(255.0).greater_equal_elem(0.5);

    // Normalize predictions if needed
    let min_val = predictions.clone().min();
    let max_val = predictions.clone().max();
    let range = max_val - min_val.clone();
    let epsilon = 1e-8;

    let pred = if range.clone().greater_elem(epsilon).into_scalar().to_bool() {
        let min_scalar = min_val.into_scalar();
        let range_scalar = range.into_scalar();
        predictions.sub_scalar(min_scalar).div_scalar(range_scalar)
    } else {
        predictions
    };

    let [height, width] = gt.dims();
    let device = gt.device();

    // Create gt_size tensor and extract scalar
    let gt_size_f64 = (height * width) as f64;
    let gt_size = Tensor::<B, 1>::from_data([gt_size_f64], &device).into_scalar();
    let gt_fg_numel = gt.clone().float().sum().into_scalar();

    // Calculate adaptive E-measure
    let adaptive_threshold_f64 = get_adaptive_threshold(pred.clone());
    let adaptive_threshold =
        Tensor::<B, 1>::from_data([adaptive_threshold_f64], &device).into_scalar();

    let adaptive_em: f64 = calculate_em_with_threshold(
        pred.clone(),
        gt.clone(),
        adaptive_threshold,
        gt_fg_numel,
        gt_size,
    );

    // Calculate changeable E-measure curve
    let changeable_em =
        calculate_em_with_cumsumhistogram(pred, gt, gt_fg_numel.to_f64(), gt_size.to_f64());

    (adaptive_em, changeable_em)
}

fn get_adaptive_threshold<B: Backend>(pred: Tensor<B, 2>) -> f64 {
    let mean_val = pred.mean().into_scalar().to_f64();
    (2.0 * mean_val).min(1.0)
}

fn calculate_em_with_threshold<B: Backend>(
    pred: Tensor<B, 2>,
    gt: Tensor<B, 2, Bool>,
    threshold: B::FloatElem,
    gt_fg_numel: B::FloatElem,
    gt_size: B::FloatElem,
) -> f64 {
    let binarized_pred = pred.greater_equal_elem(threshold);

    let fg_fg_numel: f64 = binarized_pred
        .clone()
        .bool_and(gt.clone())
        .float()
        .sum()
        .into_scalar()
        .to_f64();
    let fg_bg_numel: f64 = binarized_pred
        .bool_and(gt.bool_not())
        .float()
        .sum()
        .into_scalar()
        .to_f64();

    let fg_numel = fg_fg_numel + fg_bg_numel;
    let bg_numel = gt_size.to_f64() - fg_numel;

    let enhanced_matrix_sum = if gt_fg_numel.to_f64() == 0.0 {
        bg_numel
    } else if gt_fg_numel.to_f64() == gt_size.to_f64() {
        fg_numel
    } else {
        let (parts_numel, combinations) = generate_parts_numel_combinations(
            fg_fg_numel,
            fg_bg_numel,
            fg_numel,
            bg_numel,
            gt_fg_numel.to_f64(),
            gt_size.to_f64(),
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

    enhanced_matrix_sum / (gt_size.to_f64() - 1.0 + 1e-8)
}

fn calculate_em_with_cumsumhistogram<B: Backend>(
    pred: Tensor<B, 2>,
    gt: Tensor<B, 2, Bool>,
    gt_fg_numel: f64,
    gt_size: f64,
) -> Vec<f64> {
    // Scale predictions to 0-255 range
    let pred_scaled = (pred * 255.0).int();

    // Create histogram bins
    let num_bins = 256;
    let mut changeable_ems = vec![0.0; num_bins];

    // For each threshold
    for threshold in 0..num_bins {
        let binarized_pred = pred_scaled.clone().greater_equal_elem(threshold as i32);

        let fg_fg_numel: f64 = binarized_pred
            .clone()
            .bool_and(gt.clone())
            .float()
            .sum()
            .into_scalar()
            .to_f64();
        let fg_bg_numel: f64 = binarized_pred
            .bool_and(gt.clone().bool_not())
            .float()
            .sum()
            .into_scalar()
            .to_f64();

        let fg_numel = fg_fg_numel + fg_bg_numel;
        let bg_numel = gt_size.to_f64() - fg_numel;

        let enhanced_matrix_sum = if gt_fg_numel.to_f64() == 0.0 {
            bg_numel
        } else if gt_fg_numel.to_f64() == gt_size.to_f64() {
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

        changeable_ems[threshold] = enhanced_matrix_sum / (gt_size.to_f64() - 1.0 + 1e-8);
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
