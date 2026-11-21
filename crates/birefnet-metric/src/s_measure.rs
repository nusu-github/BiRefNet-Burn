//! Structure Measure (`SMeasure`) evaluation metric for `BiRefNet`.

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
    input::SMeasureInput,
};

/// Configuration for [`SMeasureMetric`].
#[derive(Config, Debug)]
pub struct SMeasureMetricConfig {
    /// Display name for the metric.
    #[config(default = "String::from(\"S_measure\")")]
    pub name: String,
    /// Object-vs-region blending weight `alpha` (default `0.5`).
    #[config(default = 0.5)]
    pub alpha: f64,
}

impl Default for SMeasureMetricConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl SMeasureMetricConfig {
    /// Initializes an [`SMeasureMetric`] from this configuration.
    #[must_use]
    pub fn init(&self) -> SMeasureMetric {
        SMeasureMetric {
            state: NumericMetricState::default(),
            name: Arc::new(self.name.clone()),
            alpha: self.alpha,
        }
    }
}

/// Structure Measure (`SMeasure`) metric evaluating object- and region-level structural similarity.
#[derive(Clone)]
pub struct SMeasureMetric {
    state: NumericMetricState,
    name: Arc<String>,
    alpha: f64,
}

impl fmt::Debug for SMeasureMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SMeasureMetric")
            .field("name", &self.name)
            .field("alpha", &self.alpha)
            .field("value", &self.value())
            .finish_non_exhaustive()
    }
}

impl Default for SMeasureMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl SMeasureMetric {
    /// Creates a new [`SMeasureMetric`] with default `alpha = 0.5`.
    #[must_use]
    pub fn new() -> Self {
        SMeasureMetricConfig::new().init()
    }

    /// Creates a new [`SMeasureMetric`] from `config`.
    #[must_use]
    pub fn with_config(config: &SMeasureMetricConfig) -> Self {
        config.init()
    }
}

impl Metric for SMeasureMetric {
    type Input = SMeasureInput;

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
            let mut total_sm = 0.0;
            for b in 0..batch_size {
                let pred = extract_sample_2d(&item.predictions, b);
                let gt = extract_sample_2d(&item.targets, b);
                total_sm += calculate_s_measure(pred, gt, self.alpha);
            }
            let avg_sm = total_sm / usize_to_f64(batch_size);
            self.state.update(avg_sm, batch_size);
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

impl Numeric for SMeasureMetric {
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

/// Computes Structure Measure (`SMeasure`) for a single 2D `[H, W]` prediction-target pair.
#[must_use]
pub fn calculate_s_measure(predictions: Tensor<2>, targets: Tensor<2>, alpha: f64) -> f64 {
    let (pred, gt) = prepare_data(predictions, targets);

    let y = gt.clone().mean().into_scalar::<f64>();
    if y == 0.0 {
        1.0 - pred.mean().into_scalar::<f64>()
    } else if (y - 1.0).abs() <= f64::EPSILON {
        pred.mean().into_scalar::<f64>()
    } else {
        let object_score = calculate_object_score(pred.clone(), gt.clone());
        let region_score = calculate_region_score(pred, gt);
        alpha
            .mul_add(object_score, (1.0 - alpha) * region_score)
            .max(0.0)
    }
}

fn calculate_object_score(pred: Tensor<2>, gt: Tensor<2>) -> f64 {
    let fg = pred.clone() * gt.clone();
    let bg = (pred.sub_scalar(1.0).neg()) * (gt.clone().sub_scalar(1.0).neg());

    let fg_ratio = gt.clone().mean().into_scalar::<f64>();
    let fg_score = calculate_s_object(&fg, gt.clone());
    let bg_score = calculate_s_object(&bg, gt.sub_scalar(1.0).neg());

    fg_ratio.mul_add(fg_score, (1.0 - fg_ratio) * bg_score)
}

fn calculate_s_object(pred: &Tensor<2>, gt: Tensor<2>) -> f64 {
    let mask = gt.equal_elem(1.0);
    let masked_pred = pred
        .clone()
        .mask_where(mask.clone().bool_not(), pred.zeros_like());

    let count = mask.clone().float().sum().into_scalar::<f64>();
    if count == 0.0 {
        return 0.0;
    }

    let mean_fg = masked_pred.sum().into_scalar::<f64>() / count;
    let masked_or_mean = pred
        .clone()
        .mask_where(mask.clone().bool_not(), pred.full_like(mean_fg));
    let diff = masked_or_mean - mean_fg;
    let variance = diff
        .powf_scalar(2.0)
        .mask_where(mask.bool_not(), pred.zeros_like())
        .sum()
        .into_scalar::<f64>()
        / (count - 1.0).max(1.0);
    let sigma_x = variance.sqrt();

    2.0 * mean_fg / (mean_fg.mul_add(mean_fg, 1.0) + sigma_x + 1e-8)
}

fn calculate_region_score(pred: Tensor<2>, gt: Tensor<2>) -> f64 {
    let [height, width] = gt.dims();
    let (cx, cy) = calculate_centroid(gt.clone());
    let weights = calculate_region_weights(cx, cy, height, width);

    let mut total_score = 0.0;
    if cx > 0 && cy > 0 {
        let score = calculate_ssim(
            pred.clone().slice([0..cy, 0..cx]),
            gt.clone().slice([0..cy, 0..cx]),
        );
        total_score += weights.0 * score;
    }
    if cx < width && cy > 0 {
        let score = calculate_ssim(
            pred.clone().slice([0..cy, cx..width]),
            gt.clone().slice([0..cy, cx..width]),
        );
        total_score += weights.1 * score;
    }
    if cx > 0 && cy < height {
        let score = calculate_ssim(
            pred.clone().slice([cy..height, 0..cx]),
            gt.clone().slice([cy..height, 0..cx]),
        );
        total_score += weights.2 * score;
    }
    if cx < width && cy < height {
        let score = calculate_ssim(
            pred.slice([cy..height, cx..width]),
            gt.slice([cy..height, cx..width]),
        );
        total_score += weights.3 * score;
    }

    total_score
}

fn round_coord_to_usize(coord: f64, limit: usize) -> usize {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "centroid coordinate is clamped into [0, limit]"
    )]
    let rounded = (coord + 0.5).clamp(0.0, usize_to_f64(limit)) as usize;
    rounded.min(limit)
}

fn calculate_centroid(gt: Tensor<2>) -> (usize, usize) {
    let [height, width] = gt.dims();
    let device = gt.device();

    let mask = gt.equal_elem(1.0);
    let count = mask.clone().float().sum().into_scalar::<f64>();
    if count == 0.0 {
        return (width / 2, height / 2);
    }

    #[expect(
        clippy::cast_possible_wrap,
        reason = "image dimensions fit well inside signed 64-bit range"
    )]
    let (h_i64, w_i64) = (height as i64, width as i64);
    let y_coords: Tensor<2> = Tensor::<1, Int>::arange(0..h_i64, &device)
        .float()
        .unsqueeze_dim(1);
    let x_coords: Tensor<2> = Tensor::<1, Int>::arange(0..w_i64, &device)
        .float()
        .unsqueeze_dim(0);

    let y_grid = y_coords.expand([height, width]);
    let x_grid = x_coords.expand([height, width]);

    let mask_f = mask.float();
    let cy = (y_grid * mask_f.clone()).sum().into_scalar::<f64>() / count;
    let cx = (x_grid * mask_f).sum().into_scalar::<f64>() / count;

    (
        round_coord_to_usize(cx, width),
        round_coord_to_usize(cy, height),
    )
}

fn calculate_region_weights(
    cx: usize,
    cy: usize,
    height: usize,
    width: usize,
) -> (f64, f64, f64, f64) {
    let area = usize_to_f64(height * width);
    let w1 = usize_to_f64(cx * cy) / area;
    let w2 = usize_to_f64(cy * (width - cx)) / area;
    let w3 = usize_to_f64((height - cy) * cx) / area;
    let w4 = 1.0 - w1 - w2 - w3;
    (w1, w2, w3, w4)
}

fn calculate_ssim(pred: Tensor<2>, gt: Tensor<2>) -> f64 {
    let [height, width] = pred.dims();
    let pixel_count = usize_to_f64(height * width);
    if pixel_count == 0.0 {
        return 1.0;
    }

    let pred_mean = pred.clone().mean().into_scalar::<f64>();
    let gt_mean = gt.clone().mean().into_scalar::<f64>();

    let pred_centered = pred - pred_mean;
    let gt_centered = gt - gt_mean;

    let denom = (pixel_count - 1.0).max(1.0);
    let pred_var = pred_centered
        .clone()
        .powf_scalar(2.0)
        .sum()
        .into_scalar::<f64>()
        / denom;
    let gt_var = gt_centered
        .clone()
        .powf_scalar(2.0)
        .sum()
        .into_scalar::<f64>()
        / denom;
    let cross_cov = (pred_centered * gt_centered).sum().into_scalar::<f64>() / denom;

    let numer = 4.0 * pred_mean * gt_mean * cross_cov;
    let combined_denom = pred_mean.mul_add(pred_mean, gt_mean * gt_mean) * (pred_var + gt_var);

    if numer != 0.0 {
        numer / (combined_denom + 1e-8)
    } else if combined_denom == 0.0 {
        1.0
    } else {
        0.0
    }
}
