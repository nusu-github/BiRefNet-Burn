//! F-measure evaluation metric for `BiRefNet`.

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
    common::{extract_sample_3d, prepare_data, usize_to_f64},
    input::FMeasureInput,
};

/// Configuration for [`FMeasureMetric`].
#[derive(Config, Debug)]
pub struct FMeasureMetricConfig {
    /// Display name for the metric.
    #[config(default = "String::from(\"F-measure\")")]
    pub name: String,
    /// Beta squared weight (`F_beta`, default `0.3`).
    #[config(default = 0.3)]
    pub beta: f64,
}

impl Default for FMeasureMetricConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl FMeasureMetricConfig {
    /// Initializes an [`FMeasureMetric`] from this configuration.
    #[must_use]
    pub fn init(&self) -> FMeasureMetric {
        FMeasureMetric {
            state: NumericMetricState::default(),
            beta: self.beta,
            name: Arc::new(self.name.clone()),
            adaptive_fms: Vec::new(),
            precision_curves: Vec::new(),
            recall_curves: Vec::new(),
            changeable_fms: Vec::new(),
        }
    }
}

/// Adaptive and threshold-curve F-measure metric.
#[derive(Clone)]
pub struct FMeasureMetric {
    state: NumericMetricState,
    beta: f64,
    name: Arc<String>,
    adaptive_fms: Vec<f64>,
    precision_curves: Vec<Vec<f64>>,
    recall_curves: Vec<Vec<f64>>,
    changeable_fms: Vec<Vec<f64>>,
}

impl fmt::Debug for FMeasureMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FMeasureMetric")
            .field("name", &self.name)
            .field("beta", &self.beta)
            .field("adaptive_fms", &self.adaptive_fms)
            .field("precision_curves_len", &self.precision_curves.len())
            .field("recall_curves_len", &self.recall_curves.len())
            .field("changeable_fms_len", &self.changeable_fms.len())
            .finish_non_exhaustive()
    }
}

impl Default for FMeasureMetric {
    fn default() -> Self {
        Self::new()
    }
}

/// Borrowed slice tuple of `(precision_curves, recall_curves, changeable_fm_curves)`.
pub type FMeasureCurves<'a> = (&'a [Vec<f64>], &'a [Vec<f64>], &'a [Vec<f64>]);

impl FMeasureMetric {
    /// Creates a new [`FMeasureMetric`] with default `beta = 0.3`.
    #[must_use]
    pub fn new() -> Self {
        FMeasureMetricConfig::new().init()
    }

    /// Creates a new [`FMeasureMetric`] from `config`.
    #[must_use]
    pub fn with_config(config: &FMeasureMetricConfig) -> Self {
        config.init()
    }

    /// Creates a new [`FMeasureMetric`] with a custom `beta` parameter.
    #[must_use]
    pub fn with_beta(beta: f64) -> Self {
        FMeasureMetricConfig::new().with_beta(beta).init()
    }

    /// Returns the mean adaptive F-measure across all recorded samples.
    #[must_use]
    pub fn adaptive_fm(&self) -> f64 {
        if self.adaptive_fms.is_empty() {
            return 0.0;
        }
        self.adaptive_fms.iter().sum::<f64>() / usize_to_f64(self.adaptive_fms.len())
    }

    /// Returns stored per-sample precision, recall, and changeable F-measure curves.
    #[must_use]
    pub fn curves(&self) -> FMeasureCurves<'_> {
        (
            &self.precision_curves,
            &self.recall_curves,
            &self.changeable_fms,
        )
    }
}

impl Metric for FMeasureMetric {
    type Input = FMeasureInput;

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
                let pred = extract_sample_3d(&item.predictions, b);
                let gt = extract_sample_3d(&item.targets, b);

                let adaptive_fm = calculate_adaptive_fm(pred.clone(), gt.clone(), self.beta);
                self.adaptive_fms.push(adaptive_fm);
                batch_adaptive_sum += adaptive_fm;

                let (precisions, recalls, changeable_fm) = calculate_pr_curves(pred, gt, self.beta);
                self.precision_curves.push(precisions);
                self.recall_curves.push(recalls);
                self.changeable_fms.push(changeable_fm);
            }

            let batch_avg = batch_adaptive_sum / usize_to_f64(batch_size);
            self.state.update(batch_avg, batch_size);
        }

        Ok(self
            .state
            .compute_update(FormatOptions::new(Arc::clone(&self.name)).precision(5)))
    }

    fn compute(&mut self) -> Result<SerializedEntry, TensorReadError> {
        Ok(self
            .state
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
        self.state.reset();
        self.adaptive_fms.clear();
        self.precision_curves.clear();
        self.recall_curves.clear();
        self.changeable_fms.clear();
    }
}

impl Numeric for FMeasureMetric {
    fn value(&self) -> Option<NumericEntry> {
        Some(self.state.current_value())
    }

    fn running_value(&self) -> Option<NumericEntry> {
        Some(self.state.running_value())
    }

    fn final_value(&self) -> NumericEntry {
        self.state.final_value()
    }
}

fn calculate_adaptive_fm<const D: usize>(
    predictions: Tensor<D>,
    targets: Tensor<D>,
    beta: f64,
) -> f64 {
    let (pred, gt) = prepare_data(predictions, targets);

    let adaptive_threshold = (2.0 * pred.clone().mean().into_scalar::<f64>()).min(1.0);
    let binary_pred = pred.greater_equal_elem(adaptive_threshold);

    let gt_bool = gt.greater_elem(0.5);
    let intersection_count = binary_pred
        .clone()
        .bool_and(gt_bool.clone())
        .int()
        .sum()
        .into_scalar::<f64>();

    if intersection_count == 0.0 {
        return 0.0;
    }

    let pred_positive_count = binary_pred.int().sum().into_scalar::<f64>().max(1.0);
    let gt_positive_count = gt_bool.int().sum().into_scalar::<f64>().max(1.0);

    let precision = intersection_count / pred_positive_count;
    let recall = intersection_count / gt_positive_count;

    (1.0 + beta) * precision * recall / beta.mul_add(precision, recall)
}

fn calculate_pr_curves<const D: usize>(
    predictions: Tensor<D>,
    targets: Tensor<D>,
    beta: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (pred, gt) = prepare_data(predictions, targets);
    let pred_u8 = (pred * 255.0).clamp(0.0, 255.0);
    let gt_bool = gt.greater_elem(0.5);
    let gt_positives = gt_bool.clone().int().sum().into_scalar::<f64>().max(1.0);

    let mut precisions = Vec::with_capacity(256);
    let mut recalls = Vec::with_capacity(256);
    let mut changeable_fms = Vec::with_capacity(256);

    for threshold in (0..256).rev() {
        let pred_thresh = pred_u8.clone().greater_equal_elem(f64::from(threshold));

        let tp = pred_thresh
            .clone()
            .bool_and(gt_bool.clone())
            .int()
            .sum()
            .into_scalar::<f64>();
        let pred_positives = pred_thresh.int().sum().into_scalar::<f64>().max(1.0);

        let precision = tp / pred_positives;
        let recall = tp / gt_positives;
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

/// Computes adaptive F-measure for a single 2D `[H, W]` prediction-target pair.
#[must_use]
pub fn calculate_f_measure(predictions: Tensor<2>, targets: Tensor<2>, beta: f64) -> f64 {
    calculate_adaptive_fm(predictions, targets, beta)
}
