//! Boundary Intersection over Union (`BIoU`) metric for `BiRefNet`.

use std::{fmt, sync::Arc};

use birefnet_util::{StructuringElement, erosion};
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
    input::BIoUInput,
};

/// Configuration for [`BIoUMetric`].
#[derive(Config, Debug)]
pub struct BIoUMetricConfig {
    /// Display name for the metric.
    #[config(default = "String::from(\"BIoU\")")]
    pub name: String,
    /// Boundary width fraction relative to image diagonal (`0.02` by default).
    #[config(default = 0.02)]
    pub dilation_ratio: f64,
}

impl Default for BIoUMetricConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl BIoUMetricConfig {
    /// Initializes a [`BIoUMetric`] from this configuration.
    #[must_use]
    pub fn init(&self) -> BIoUMetric {
        BIoUMetric {
            state: NumericMetricState::default(),
            dilation_ratio: self.dilation_ratio,
            name: Arc::new(self.name.clone()),
        }
    }
}

/// Boundary Intersection over Union (`BIoU`) metric.
#[derive(Clone)]
pub struct BIoUMetric {
    state: NumericMetricState,
    dilation_ratio: f64,
    name: Arc<String>,
}

impl fmt::Debug for BIoUMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BIoUMetric")
            .field("name", &self.name)
            .field("dilation_ratio", &self.dilation_ratio)
            .field("value", &self.value())
            .finish_non_exhaustive()
    }
}

impl Default for BIoUMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl BIoUMetric {
    /// Creates a new [`BIoUMetric`] with default `dilation_ratio = 0.02`.
    #[must_use]
    pub fn new() -> Self {
        BIoUMetricConfig::new().init()
    }

    /// Creates a new [`BIoUMetric`] from `config`.
    #[must_use]
    pub fn with_config(config: &BIoUMetricConfig) -> Self {
        config.init()
    }
}

impl Metric for BIoUMetric {
    type Input = BIoUInput;

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
            let mut total_biou_curves = Vec::with_capacity(batch_size);
            for b in 0..batch_size {
                let pred = extract_sample_3d(&item.predictions, b);
                let gt = extract_sample_3d(&item.targets, b);
                total_biou_curves.push(calculate_biou_curve(pred, gt, self.dilation_ratio));
            }
            let avg_biou = average_biou_curves(&total_biou_curves);
            self.state.update(avg_biou, batch_size);
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
    }
}

impl Numeric for BIoUMetric {
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

fn calculate_biou_curve(
    predictions: Tensor<3>,
    targets: Tensor<3>,
    dilation_ratio: f64,
) -> Vec<f64> {
    let (pred, gt) = prepare_data(predictions, targets);

    let pred_boundary = mask_to_boundary(pred, dilation_ratio);
    let gt_boundary = mask_to_boundary(gt, dilation_ratio);

    calculate_biou_histogram_curves(pred_boundary, gt_boundary)
}

fn mask_to_boundary(mask: Tensor<3>, dilation_ratio: f64) -> Tensor<3> {
    let [_c, h, w] = mask.dims();
    let device = mask.device();

    let img_diag = usize_to_f64(h).hypot(usize_to_f64(w));
    let dilation_f64 = (dilation_ratio.max(0.0) * img_diag).round();
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "dilation_f64 is non-negative and clamped to [1.0, img_diag]"
    )]
    let dilation = (dilation_f64.clamp(1.0, img_diag.max(1.0)) as usize).max(1);

    let mask_4d = mask.clone().unsqueeze_dim(0);
    let disk_kernel = StructuringElement::disk(dilation, &device);
    let eroded = erosion(mask_4d, &disk_kernel).squeeze_dim::<3>(0);

    mask - eroded
}

fn calculate_biou_histogram_curves(pred_boundary: Tensor<3>, gt_boundary: Tensor<3>) -> Vec<f64> {
    let pred_u8 = pred_boundary.mul_scalar(255.0).clamp(0.0, 255.0);
    let gt_binary = gt_boundary.greater_elem(0.5);
    let gt_total = gt_binary.clone().int().sum().into_scalar::<f64>();
    let gt_not = gt_binary.clone().bool_not();

    let mut ious = Vec::with_capacity(255);
    for threshold in 0..255 {
        let pred_thresh = pred_u8.clone().greater_elem(f64::from(threshold));
        let tp = pred_thresh
            .clone()
            .bool_and(gt_binary.clone())
            .int()
            .sum()
            .into_scalar::<f64>();
        let fp = pred_thresh
            .bool_and(gt_not.clone())
            .int()
            .sum()
            .into_scalar::<f64>();

        let denom = (gt_total + fp).max(1.0);
        ious.push(tp / denom);
    }

    ious
}

fn average_biou_curves(curves: &[Vec<f64>]) -> f64 {
    if curves.is_empty() {
        return 0.0;
    }

    let curve_len = curves[0].len();
    if curve_len == 0 {
        return 0.0;
    }

    let mut avg_curve = vec![0.0; curve_len];
    for curve in curves {
        for (idx, &val) in curve.iter().enumerate() {
            avg_curve[idx] += val;
        }
    }

    let batch_size = usize_to_f64(curves.len());
    avg_curve.iter().map(|val| val / batch_size).sum::<f64>() / usize_to_f64(curve_len)
}

/// Computes mean Boundary IoU across 255 thresholds for a single 2D `[H, W]` pair.
#[must_use]
pub fn calculate_biou(predictions: Tensor<2>, targets: Tensor<2>, dilation_ratio: f64) -> f64 {
    let curve = calculate_biou_curve(
        predictions.unsqueeze_dim(0),
        targets.unsqueeze_dim(0),
        dilation_ratio,
    );
    curve.iter().sum::<f64>() / usize_to_f64(curve.len())
}
