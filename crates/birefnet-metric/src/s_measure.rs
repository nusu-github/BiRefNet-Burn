//! Structure Measure (S-measure) metric for BiRefNet.
//!
//! S-measure evaluates the structural similarity between prediction and ground truth,
//! considering both object-level and region-level information.

use core::marker::PhantomData;

use burn::{
    config::Config,
    prelude::*,
    tensor::{backend::Backend, cast::ToElement, Int, Tensor},
    train::metric::{
        state::{FormatOptions, NumericMetricState},
        Metric, MetricEntry, MetricMetadata, Numeric,
    },
};

/// Configuration for the S-measure metric.
#[derive(Config, Debug)]
pub struct SMeasureMetricConfig {
    /// Name of the metric (default: "S-measure").
    #[config(default = "String::from(\"S_measure\")")]
    name: String,
    /// Alpha value for weighting object and region scores (default: 0.5).
    #[config(default = 0.5)]
    alpha: f64,
}

/// S-measure metric input.
#[derive(Debug, Clone)]
pub struct SMeasureInput<B: Backend> {
    /// Predictions with shape `[batch_size, height, width]`.
    pub predictions: Tensor<B, 3>,
    /// Ground truth with shape `[batch_size, height, width]`.
    pub targets: Tensor<B, 3>,
}

impl<B: Backend> SMeasureInput<B> {
    /// Creates a new S-measure input.
    pub const fn new(predictions: Tensor<B, 3>, targets: Tensor<B, 3>) -> Self {
        Self {
            predictions,
            targets,
        }
    }
}

/// S-measure metric.
pub struct SMeasureMetric<B: Backend> {
    state: NumericMetricState,
    name: String,
    alpha: f64,
    _backend: PhantomData<B>,
}

impl<B: Backend> Default for SMeasureMetric<B> {
    fn default() -> Self {
        Self {
            state: NumericMetricState::default(),
            name: "S_measure".to_owned(),
            alpha: 0.5,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> SMeasureMetric<B> {
    /// Creates a new S-measure metric.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new S-measure metric with custom configuration.
    pub fn with_config(config: SMeasureMetricConfig) -> Self {
        Self {
            state: NumericMetricState::default(),
            name: config.name,
            alpha: config.alpha,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Metric for SMeasureMetric<B> {
    type Input = SMeasureInput<B>;

    fn name(&self) -> String {
        self.name.to_string()
    }

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let [batch_size, ..] = input.predictions.dims();

        let mut total_sm = 0.0;

        for b in 0..batch_size {
            let pred = input
                .predictions
                .clone()
                .slice(s![b..=b, .., ..])
                .squeeze(0);
            let gt = input.targets.clone().slice(s![b..=b, .., ..]).squeeze(0);

            let sm = calculate_s_measure(pred, gt, self.alpha);
            total_sm += sm;
        }

        let avg_sm = total_sm / batch_size as f64;
        self.state.update(
            avg_sm,
            batch_size,
            FormatOptions::new(self.name.to_string()).precision(5),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl<B: Backend> Numeric for SMeasureMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

/// Calculates S-measure for a single prediction-target pair.
///
/// # Arguments
/// * `predictions` - Predictions with shape `[height, width]`.
/// * `targets` - Ground truth with shape `[height, width]`.
/// * `alpha` - Weight for combining object and region scores.
///
/// # Returns
/// The S-measure value.
pub fn calculate_s_measure<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    alpha: f64,
) -> f64 {
    // Prepare data
    let gt = targets.div_scalar(255.0).greater_equal_elem(0.5).float();

    // Normalize predictions if needed
    let min_val = predictions.clone().min();
    let max_val = predictions.clone().max();
    let range = max_val - min_val.clone();
    let epsilon = 1e-8;

    let pred = if range.clone().greater_elem(epsilon).into_scalar().to_bool() {
        let min_scalar = min_val.into_scalar().to_f64();
        let range_scalar = range.into_scalar().to_f64();
        predictions.sub_scalar(min_scalar).div_scalar(range_scalar)
    } else {
        predictions
    };

    // Calculate mean of ground truth
    let y = gt.clone().mean().into_scalar().to_f64();

    if y == 0.0 {
        // All background
        1.0 - pred.mean().into_scalar().to_f64()
    } else if y == 1.0 {
        // All foreground
        pred.mean().into_scalar().to_f64()
    } else {
        // Mixed case
        let object_score = calculate_object_score(pred.clone(), gt.clone());
        let region_score = calculate_region_score(pred, gt);
        let score = alpha.mul_add(object_score, (1.0 - alpha) * region_score);
        score.max(0.0)
    }
}

fn calculate_object_score<B: Backend>(pred: Tensor<B, 2>, gt: Tensor<B, 2>) -> f64 {
    let fg = pred.clone() * gt.clone();
    let bg = (pred.sub_scalar(1.0).neg()) * (gt.clone().sub_scalar(1.0).neg());

    let u = gt.clone().mean().into_scalar().to_f64();

    let fg_score = calculate_s_object(fg, gt.clone());
    let bg_score = calculate_s_object(bg, gt.sub_scalar(1.0).neg());

    u.mul_add(fg_score, (1.0 - u) * bg_score)
}

fn calculate_s_object<B: Backend>(pred: Tensor<B, 2>, gt: Tensor<B, 2>) -> f64 {
    let mask = gt.equal_elem(1.0);
    let masked_pred = pred
        .clone()
        .mask_where(mask.clone().bool_not(), pred.zeros_like());

    let count = mask.clone().float().sum().into_scalar().to_f64();
    if count == 0.0 {
        return 0.0;
    }

    let x = masked_pred.sum().into_scalar().to_f64() / count;

    // Calculate standard deviation
    let mean_tensor = pred.clone().mask_where(
        mask.clone().bool_not(),
        Tensor::from_data([x], &pred.device()),
    );
    let diff = mean_tensor - x;
    let variance = diff
        .powf_scalar(2.0)
        .mask_where(mask.bool_not(), pred.zeros_like())
        .sum()
        .into_scalar()
        .to_f64()
        / (count - 1.0).max(1.0);
    let sigma_x = variance.sqrt();

    // S-object score formula
    2.0 * x / (x.mul_add(x, 1.0) + sigma_x + 1e-8)
}

fn calculate_region_score<B: Backend>(pred: Tensor<B, 2>, gt: Tensor<B, 2>) -> f64 {
    let [height, width] = gt.dims();

    // Calculate centroid
    let (cx, cy) = calculate_centroid(gt.clone());

    // Divide into 4 regions
    let weights = calculate_region_weights(cx, cy, height, width);

    // Calculate SSIM for each region
    let mut total_score = 0.0;

    // Top-left
    if cx > 0 && cy > 0 {
        let pred_tl = pred.clone().slice([0..cy, 0..cx]);
        let gt_tl = gt.clone().slice([0..cy, 0..cx]);
        let score = calculate_ssim(pred_tl, gt_tl);
        total_score += weights.0 * score;
    }

    // Top-right
    if cx < width && cy > 0 {
        let pred_tr = pred.clone().slice([0..cy, cx..width]);
        let gt_tr = gt.clone().slice([0..cy, cx..width]);
        let score = calculate_ssim(pred_tr, gt_tr);
        total_score += weights.1 * score;
    }

    // Bottom-left
    if cx > 0 && cy < height {
        let pred_bl = pred.clone().slice([cy..height, 0..cx]);
        let gt_bl = gt.clone().slice([cy..height, 0..cx]);
        let score = calculate_ssim(pred_bl, gt_bl);
        total_score += weights.2 * score;
    }

    // Bottom-right
    if cx < width && cy < height {
        let pred_br = pred.slice([cy..height, cx..width]);
        let gt_br = gt.slice([cy..height, cx..width]);
        let score = calculate_ssim(pred_br, gt_br);
        total_score += weights.3 * score;
    }

    total_score
}

fn calculate_centroid<B: Backend>(gt: Tensor<B, 2>) -> (usize, usize) {
    let [height, width] = gt.dims();
    let device = gt.device();

    let mask = gt.equal_elem(1.0);
    let count = mask.clone().float().sum().into_scalar();

    if count.to_f64() == 0.0 {
        // Return center if no foreground
        (width / 2, height / 2)
    } else {
        // Create coordinate grids
        let y_coords: Tensor<B, 2> = Tensor::<B, 1, Int>::arange(0..height as i64, &device)
            .float()
            .unsqueeze_dim(1);
        let x_coords: Tensor<B, 2> = Tensor::<B, 1, Int>::arange(0..width as i64, &device)
            .float()
            .unsqueeze_dim(0);

        let y_grid = y_coords.expand([height, width]);
        let x_grid = x_coords.expand([height, width]);

        // Calculate weighted average
        let cy = (y_grid * mask.clone().float()).sum().into_scalar().to_f64() / count.to_f64();
        let cx = (x_grid * mask.float()).sum().into_scalar().to_f64() / count.to_f64();

        ((cx + 0.5) as usize, (cy + 0.5) as usize)
    }
}

fn calculate_region_weights(
    cx: usize,
    cy: usize,
    height: usize,
    width: usize,
) -> (f64, f64, f64, f64) {
    let area = (height * width) as f64;

    let w1 = (cx * cy) as f64 / area;
    let w2 = (cy * (width - cx)) as f64 / area;
    let w3 = ((height - cy) * cx) as f64 / area;
    let w4 = 1.0 - w1 - w2 - w3;

    (w1, w2, w3, w4)
}

fn calculate_ssim<B: Backend>(pred: Tensor<B, 2>, gt: Tensor<B, 2>) -> f64 {
    let [h, w] = pred.dims();
    let n = (h * w) as f64;

    if n == 0.0 {
        return 1.0;
    }

    let x = pred.clone().mean().into_scalar().to_f64();
    let y = gt.clone().mean().into_scalar().to_f64();

    let pred_centered = pred - x;
    let gt_centered = gt - y;

    let sigma_x_sq = (pred_centered
        .clone()
        .powf_scalar(2.0)
        .sum()
        .into_scalar()
        .to_f64()
        / (n - 1.0).max(1.0))
    .to_f64();
    let sigma_y_sq = (gt_centered
        .clone()
        .powf_scalar(2.0)
        .sum()
        .into_scalar()
        .to_f64()
        / (n - 1.0).max(1.0))
    .to_f64();
    let sigma_xy = (pred_centered * gt_centered).sum().into_scalar().to_f64() / (n - 1.0).max(1.0);

    let alpha = 4.0 * x * y * sigma_xy;
    let beta = x.mul_add(x, y * y) * (sigma_x_sq + sigma_y_sq);

    if alpha != 0.0 {
        alpha / (beta + 1e-8)
    } else if alpha == 0.0 && beta == 0.0 {
        1.0
    } else {
        0.0
    }
}
