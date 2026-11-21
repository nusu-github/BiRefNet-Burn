//! Weighted F-measure (`WeightedFMeasure`) evaluation metric for `BiRefNet`.

use std::{fmt, sync::Arc};

use birefnet_util::{euclidean_distance_transform, gaussian_filter_matlab};
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
    input::WeightedFMeasureInput,
};

/// Configuration for [`WeightedFMeasureMetric`].
#[derive(Config, Debug)]
pub struct WeightedFMeasureMetricConfig {
    /// Display name for the metric.
    #[config(default = "String::from(\"WF_measure\")")]
    pub name: String,
    /// Beta squared weight (`beta = 1.0` by default).
    #[config(default = 1.0)]
    pub beta: f64,
}

impl Default for WeightedFMeasureMetricConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl WeightedFMeasureMetricConfig {
    /// Initializes a [`WeightedFMeasureMetric`] from this configuration.
    #[must_use]
    pub fn init(&self) -> WeightedFMeasureMetric {
        WeightedFMeasureMetric {
            state: NumericMetricState::default(),
            name: Arc::new(self.name.clone()),
            beta: self.beta,
        }
    }
}

/// Weighted F-measure metric weighting pixels by spatial error dependency and boundary distance.
#[derive(Clone)]
pub struct WeightedFMeasureMetric {
    state: NumericMetricState,
    name: Arc<String>,
    beta: f64,
}

impl fmt::Debug for WeightedFMeasureMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WeightedFMeasureMetric")
            .field("name", &self.name)
            .field("beta", &self.beta)
            .field("value", &self.value())
            .finish_non_exhaustive()
    }
}

impl Default for WeightedFMeasureMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl WeightedFMeasureMetric {
    /// Creates a new [`WeightedFMeasureMetric`] with default `beta = 1.0`.
    #[must_use]
    pub fn new() -> Self {
        WeightedFMeasureMetricConfig::new().init()
    }

    /// Creates a new [`WeightedFMeasureMetric`] from `config`.
    #[must_use]
    pub fn with_config(config: &WeightedFMeasureMetricConfig) -> Self {
        config.init()
    }
}

impl Metric for WeightedFMeasureMetric {
    type Input = WeightedFMeasureInput;

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
            let mut total_wfm = 0.0;
            for b in 0..batch_size {
                let pred = extract_sample_3d(&item.predictions, b);
                let gt = extract_sample_3d(&item.targets, b);
                total_wfm += calculate_weighted_f_measure(pred, gt, self.beta);
            }
            let avg_wfm = total_wfm / usize_to_f64(batch_size);
            self.state.update(avg_wfm, batch_size);
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

impl Numeric for WeightedFMeasureMetric {
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

/// Computes Weighted F-measure for a single 3D `[C, H, W]` prediction-target pair.
#[must_use]
pub fn calculate_weighted_f_measure(predictions: Tensor<3>, targets: Tensor<3>, beta: f64) -> f64 {
    const EPSILON: f64 = 1e-8;

    let pred_2d = predictions.squeeze_dim::<2>(0);
    let gt_2d = targets.squeeze_dim::<2>(0);
    let (pred, gt) = prepare_data(pred_2d, gt_2d);

    let gt_bool = gt.clone().greater_elem(0.5);
    let gt_sum = gt.clone().sum().into_scalar::<f64>();
    if gt_sum == 0.0 {
        return 0.0;
    }

    let binary_bg = gt_bool.clone().bool_not().float().unsqueeze::<4>();
    let (dst_4d, (nearest_row_4d, nearest_col_4d)) = euclidean_distance_transform(binary_bg);
    let dst = dst_4d.squeeze_dims::<2>(&[0, 1]);
    let idx_y = nearest_row_4d.squeeze_dims::<2>(&[0, 1]);
    let idx_x = nearest_col_4d.squeeze_dims::<2>(&[0, 1]);

    let abs_error = (pred - gt).abs();
    let et = apply_error_dependency(abs_error.clone(), gt_bool.clone(), idx_y, idx_x);
    let ea = gaussian_filter(et, 7, 5.0);
    let min_e_ea = calculate_min_error(abs_error, ea, gt_bool.clone());
    let importance = calculate_importance_weights(gt_bool.clone(), &dst);
    let ew = min_e_ea * importance;

    let zero_mask = ew.zeros_like();
    let ew_fg = ew
        .clone()
        .mask_where(gt_bool.clone().bool_not(), zero_mask.clone())
        .sum()
        .into_scalar::<f64>();
    let tpw = gt_sum - ew_fg;
    let fpw = ew.mask_where(gt_bool, zero_mask).sum().into_scalar::<f64>();

    let r = (1.0 - ew_fg / gt_sum.max(EPSILON)).max(0.0);
    let p = tpw / (tpw + fpw + EPSILON);

    (1.0 + beta) * r * p / (beta.mul_add(p, r) + EPSILON)
}

fn coord_to_usize(coord: f32, limit: usize) -> usize {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "nearest foreground indices are integer coordinates in [0, limit - 1]"
    )]
    let idx = coord.max(0.0).round() as usize;
    idx.min(limit.saturating_sub(1))
}

fn apply_error_dependency(
    abs_error: Tensor<2>,
    gt: Tensor<2, Bool>,
    idx_y: Tensor<2>,
    idx_x: Tensor<2>,
) -> Tensor<2> {
    let [height, width] = abs_error.dims();
    if height == 0 || width == 0 {
        return abs_error;
    }

    let device = abs_error.device();
    let err_vals = abs_error
        .into_data()
        .try_to_vec::<f32>()
        .expect("error tensor must be convertible to f32");
    let gt_vals = gt
        .float()
        .into_data()
        .try_to_vec::<f32>()
        .expect("ground truth mask must be convertible to f32");
    let row_coords = idx_y
        .into_data()
        .try_to_vec::<f32>()
        .expect("row index tensor must be convertible to f32");
    let col_coords = idx_x
        .into_data()
        .try_to_vec::<f32>()
        .expect("col index tensor must be convertible to f32");

    let mut et = err_vals.clone();
    for (idx, (&is_fg, (&ny, &nx))) in gt_vals
        .iter()
        .zip(row_coords.iter().zip(col_coords.iter()))
        .enumerate()
    {
        if is_fg < 0.5 {
            let row = coord_to_usize(ny, height);
            let col = coord_to_usize(nx, width);
            et[idx] = err_vals[row * width + col];
        }
    }

    Tensor::<2>::from_data(TensorData::new(et, [height, width]), &device)
}

fn gaussian_filter(tensor: Tensor<2>, kernel_size: usize, sigma: f64) -> Tensor<2> {
    gaussian_filter_matlab(tensor.unsqueeze::<4>(), (kernel_size, kernel_size), sigma)
        .squeeze_dims::<2>(&[0, 1])
}

fn calculate_min_error(
    abs_error: Tensor<2>,
    smoothed: Tensor<2>,
    gt: Tensor<2, Bool>,
) -> Tensor<2> {
    let condition = gt.bool_and(smoothed.clone().lower(abs_error.clone()));
    abs_error.mask_where(condition, smoothed)
}

fn calculate_importance_weights(gt: Tensor<2, Bool>, dst: &Tensor<2>) -> Tensor<2> {
    let bg_mask = gt.bool_not();
    let decay = 0.5_f64.ln() / 5.0;
    let bg_weights = (dst.clone().mul_scalar(decay).exp()).neg().add_scalar(2.0);
    dst.ones_like().mask_where(bg_mask, bg_weights)
}
