//! Enhanced-alignment Measure (`EMeasure`) evaluation metric for `BiRefNet`.

use std::{fmt, sync::Arc};

use burn::{
    prelude::*,
    tensor::TensorReadError,
    train::metric::{
        Metric, MetricAttributes, MetricMetadata, Numeric, NumericAttributes, NumericEntry,
        SerializedEntry,
        state::{FormatOptions, NumericMetricState},
    },
};

use crate::{
    common::{extract_sample_2d, prepare_data, usize_to_f64},
    input::EMeasureInput,
};

/// Configuration for [`EMeasureMetric`].
#[derive(Config, Debug, Default)]
pub struct EMeasureMetricConfig {
    /// Display name for the metric.
    #[config(default = "String::from(\"E_measure\")")]
    pub name: String,
}

impl EMeasureMetricConfig {
    /// Initializes an [`EMeasureMetric`] from this configuration.
    #[must_use]
    pub fn init(&self) -> EMeasureMetric {
        EMeasureMetric {
            state: EMeasureState::default(),
            numeric_state: NumericMetricState::default(),
            name: Arc::new(self.name.clone()),
        }
    }
}

/// Accumulated sample-level E-measure results across evaluation batches.
#[derive(Debug, Default, Clone)]
pub struct EMeasureState {
    adaptive_ems: Vec<f64>,
    changeable_ems: Vec<Vec<f64>>,
}

impl EMeasureState {
    /// Returns `(mean_adaptive_em, mean_changeable_em_curve)` over accumulated samples.
    #[must_use]
    pub fn results(&self) -> (f64, Vec<f64>) {
        e_measure_results(self)
    }
}

/// Enhanced-alignment Measure (`EMeasure`) metric combining adaptive and thresholded curves.
#[derive(Clone)]
pub struct EMeasureMetric {
    state: EMeasureState,
    numeric_state: NumericMetricState,
    name: Arc<String>,
}

impl fmt::Debug for EMeasureMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EMeasureMetric")
            .field("name", &self.name)
            .field("state", &self.state)
            .field("value", &self.value())
            .finish_non_exhaustive()
    }
}

impl Default for EMeasureMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl EMeasureMetric {
    /// Creates a new [`EMeasureMetric`] with default name `"E_measure"`.
    #[must_use]
    pub fn new() -> Self {
        EMeasureMetricConfig::new().init()
    }

    /// Creates a new [`EMeasureMetric`] from `config`.
    #[must_use]
    pub fn with_config(config: &EMeasureMetricConfig) -> Self {
        config.init()
    }

    /// Returns a reference to the accumulated [`EMeasureState`].
    #[must_use]
    pub const fn state(&self) -> &EMeasureState {
        &self.state
    }
}

impl Metric for EMeasureMetric {
    type Input = EMeasureInput;

    fn name(&self) -> Arc<String> {
        Arc::clone(&self.name)
    }

    fn update(
        &mut self,
        item: &Self::Input,
        _metadata: &MetricMetadata,
    ) -> Result<SerializedEntry, TensorReadError> {
        let [batch_size, ..] = item.predictions.dims();
        if batch_size > 0 {
            let mut batch_adaptive_sum = 0.0;
            for b in 0..batch_size {
                let pred = extract_sample_2d(&item.predictions, b);
                let gt = extract_sample_2d(&item.targets, b);

                let (adaptive_em, changeable_em) = calculate_e_measure(pred, gt);
                self.state.adaptive_ems.push(adaptive_em);
                self.state.changeable_ems.push(changeable_em);
                batch_adaptive_sum += adaptive_em;
            }

            let batch_avg = batch_adaptive_sum / usize_to_f64(batch_size);
            self.numeric_state.update(batch_avg, batch_size);
        }

        Ok(self
            .numeric_state
            .compute_update(FormatOptions::new(Arc::clone(&self.name)).precision(5)))
    }

    fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
        Ok(self
            .numeric_state
            .compute_final(FormatOptions::new(Arc::clone(&self.name)).precision(5)))
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
        Some(self.numeric_state.current_value())
    }

    fn running_value(&self) -> Option<NumericEntry> {
        Some(self.numeric_state.running_value())
    }

    fn final_value(&self) -> NumericEntry {
        self.numeric_state.final_value()
    }
}

/// Calculates `(adaptive_em, changeable_em_curve)` for a 2D `[H, W]` pair.
#[must_use]
pub fn calculate_e_measure(predictions: Tensor<2>, targets: Tensor<2>) -> (f64, Vec<f64>) {
    let (pred, gt_float) = prepare_data(predictions, targets);
    let gt = gt_float.greater_elem(0.5);

    let [height, width] = gt.dims();
    let gt_size = usize_to_f64(height * width);
    let gt_fg_numel = gt.clone().float().sum().into_scalar::<f64>();

    let adaptive_threshold = (2.0 * pred.clone().mean().into_scalar::<f64>()).min(1.0);
    let adaptive_em = calculate_em_with_threshold(
        pred.clone(),
        gt.clone(),
        adaptive_threshold,
        gt_fg_numel,
        gt_size,
    );
    let changeable_em = calculate_em_with_threshold_curve(pred, &gt, gt_fg_numel, gt_size);

    (adaptive_em, changeable_em)
}

fn enhanced_alignment_score(tp_count: f64, fp_count: f64, gt_fg_numel: f64, gt_size: f64) -> f64 {
    let predicted_positive = tp_count + fp_count;
    let predicted_negative = gt_size - predicted_positive;

    let enhanced_matrix_sum = if gt_fg_numel == 0.0 {
        predicted_negative
    } else if (gt_fg_numel - gt_size).abs() <= f64::EPSILON {
        predicted_positive
    } else {
        let (parts_numel, combinations) = generate_parts_numel_combinations(
            tp_count,
            fp_count,
            predicted_positive,
            predicted_negative,
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

fn calculate_em_with_threshold(
    pred: Tensor<2>,
    gt: Tensor<2, Bool>,
    threshold: f64,
    gt_fg_numel: f64,
    gt_size: f64,
) -> f64 {
    let binarized_pred = pred.greater_equal_elem(threshold);
    let tp_count = binarized_pred
        .clone()
        .bool_and(gt.clone())
        .float()
        .sum()
        .into_scalar::<f64>();
    let fp_count = binarized_pred
        .bool_and(gt.bool_not())
        .float()
        .sum()
        .into_scalar::<f64>();

    enhanced_alignment_score(tp_count, fp_count, gt_fg_numel, gt_size)
}

fn calculate_em_with_threshold_curve(
    pred: Tensor<2>,
    gt: &Tensor<2, Bool>,
    gt_fg_numel: f64,
    gt_size: f64,
) -> Vec<f64> {
    let pred_scaled = (pred * 255.0).clamp(0.0, 255.0);
    let gt_not = gt.clone().bool_not();

    let mut changeable_ems = Vec::with_capacity(256);
    for threshold in 0..256 {
        let binarized_pred = pred_scaled.clone().greater_equal_elem(f64::from(threshold));
        let tp_count = binarized_pred
            .clone()
            .bool_and(gt.clone())
            .float()
            .sum()
            .into_scalar::<f64>();
        let fp_count = binarized_pred
            .bool_and(gt_not.clone())
            .float()
            .sum()
            .into_scalar::<f64>();

        changeable_ems.push(enhanced_alignment_score(
            tp_count,
            fp_count,
            gt_fg_numel,
            gt_size,
        ));
    }

    changeable_ems
}

fn generate_parts_numel_combinations(
    true_positives: f64,
    false_positives: f64,
    pred_positive: f64,
    pred_negative: f64,
    gt_fg_numel: f64,
    gt_size: f64,
) -> ([f64; 4], [(f64, f64); 4]) {
    let missed_fg = gt_fg_numel - true_positives;
    let true_bg = pred_negative - missed_fg;

    let ordered_parts = [true_positives, false_positives, missed_fg, true_bg];

    let mean_pred_value = pred_positive / gt_size;
    let mean_gt_value = gt_fg_numel / gt_size;

    let pred_hi = 1.0 - mean_pred_value;
    let pred_lo = -mean_pred_value;
    let target_hi = 1.0 - mean_gt_value;
    let target_lo = -mean_gt_value;

    let combinations = [
        (pred_hi, target_hi),
        (pred_hi, target_lo),
        (pred_lo, target_hi),
        (pred_lo, target_lo),
    ];

    (ordered_parts, combinations)
}

/// Computes `(mean_adaptive_em, mean_changeable_em_curve)` from `state`.
#[must_use]
pub fn e_measure_results(state: &EMeasureState) -> (f64, Vec<f64>) {
    if state.adaptive_ems.is_empty() {
        return (0.0, vec![0.0; 256]);
    }

    let sample_count = usize_to_f64(state.adaptive_ems.len());
    let adaptive_em = state.adaptive_ems.iter().sum::<f64>() / sample_count;

    let num_bins = state.changeable_ems.first().map_or(256, Vec::len);
    let mut avg_changeable_em = vec![0.0; num_bins];
    for changeable in &state.changeable_ems {
        for (idx, &val) in changeable.iter().enumerate() {
            avg_changeable_em[idx] += val;
        }
    }

    for val in &mut avg_changeable_em {
        *val /= sample_count;
    }

    (adaptive_em, avg_changeable_em)
}
