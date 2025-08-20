//! Weighted F-measure metric for BiRefNet.
//!
//! WF-measure evaluates segmentation quality with pixel importance weighting
//! based on distance from ground truth boundaries.

use core::marker::PhantomData;

use birefnet_util::{euclidean_distance_transform_simple, gaussian_filter_matlab};
use burn::{
    config::Config,
    tensor::{backend::Backend, cast::ToElement, s, Bool, Tensor},
    train::metric::{
        state::{FormatOptions, NumericMetricState},
        Metric, MetricEntry, MetricMetadata, Numeric,
    },
};

/// Configuration for the Weighted F-measure metric.
#[derive(Config, Debug)]
pub struct WeightedFMeasureMetricConfig {
    /// Name of the metric (default: "WF-measure").
    #[config(default = "String::from(\"WF_measure\")")]
    name: String,
    /// Beta value for F-measure calculation (default: 1.0).
    #[config(default = 1.0)]
    beta: f64,
}

use crate::input::WeightedFMeasureInput;

/// Weighted F-measure metric.
pub struct WeightedFMeasureMetric<B: Backend> {
    state: NumericMetricState,
    name: String,
    beta: f64,
    _backend: PhantomData<B>,
}

impl<B: Backend> Default for WeightedFMeasureMetric<B> {
    fn default() -> Self {
        Self {
            state: NumericMetricState::default(),
            name: "WF_measure".to_owned(),
            beta: 1.0,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> WeightedFMeasureMetric<B> {
    /// Creates a new Weighted F-measure metric.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new Weighted F-measure metric with custom configuration.
    pub fn with_config(config: WeightedFMeasureMetricConfig) -> Self {
        Self {
            state: NumericMetricState::default(),
            name: config.name,
            beta: config.beta,
            _backend: PhantomData,
        }
    }
}

impl<B: Backend> Metric for WeightedFMeasureMetric<B> {
    type Input = WeightedFMeasureInput<B>;

    fn name(&self) -> String {
        self.name.to_string()
    }

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let [batch_size, ..] = input.predictions.dims();

        let mut total_wfm = 0.0;

        // Process each item in the batch
        for b in 0..batch_size {
            let pred: Tensor<B, 3> = input
                .predictions
                .clone()
                .slice(s![b..=b, .., .., ..])
                .squeeze(0);
            let gt: Tensor<B, 3> = input
                .targets
                .clone()
                .slice(s![b..=b, .., .., ..])
                .squeeze(0);

            let wfm = calculate_weighted_f_measure(pred, gt, self.beta);
            total_wfm += wfm;
        }

        let avg_wfm = total_wfm / batch_size as f64;
        self.state.update(
            avg_wfm,
            batch_size,
            FormatOptions::new(self.name.to_string()).precision(5),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl<B: Backend> Numeric for WeightedFMeasureMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

/// Calculates Weighted F-measure for a single prediction-target pair.
///
/// Implements the weighted F-measure calculation from Python BiRefNet:
/// 1. Apply _prepare_data to normalize inputs
/// 2. Check for all-background case
/// 3. Calculate distance transform for background pixels  
/// 4. Apply error dependency and Gaussian smoothing
/// 5. Calculate pixel importance weights
/// 6. Compute weighted precision, recall, and F-measure
///
/// # Arguments
/// * `predictions` - Predictions with shape `[channels, height, width]` or `[height, width]`.
/// * `targets` - Ground truth with shape `[channels, height, width]` or `[height, width]`.
/// * `beta` - Beta value for F-measure calculation.
///
/// # Returns
/// The Weighted F-measure value.
pub fn calculate_weighted_f_measure<B: Backend>(
    predictions: Tensor<B, 3>,
    targets: Tensor<B, 3>,
    beta: f64,
) -> f64 {
    // Prepare data following Python _prepare_data function
    let (pred, gt) = prepare_data(predictions, targets);

    // Check if all background (np.all(~gt))
    if gt
        .clone()
        .greater_elem(0.5)
        .int()
        .sum()
        .into_scalar()
        .to_f64()
        == 0.0
    {
        return 0.0;
    }

    // Calculate distance transform for background pixels
    let gt_bool = gt.clone().greater_elem(0.5);
    let dst = distance_transform(gt_bool.clone());

    // Calculate error
    let e = (pred - gt.clone()).abs();

    // Apply error dependency transformation
    let et = apply_error_dependency(e.clone(), gt_bool.clone(), dst.clone());

    // Apply Gaussian filter for smoothing
    let ea = gaussian_filter(et, 7, 5.0);

    // Calculate minimum error
    let min_e_ea = calculate_min_error(e, ea, gt_bool.clone());

    // Calculate pixel importance weights
    let b = calculate_importance_weights(gt_bool.clone(), dst);

    // Calculate weighted error
    let ew = min_e_ea * b;

    // Calculate weighted metrics
    let gt_sum = gt.sum().into_scalar().to_f64();
    let zero_mask = ew.zeros_like();
    let tpw = gt_sum
        - ew.clone()
            .mask_where(gt_bool.clone().bool_not(), zero_mask.clone())
            .sum()
            .into_scalar()
            .to_f64();
    let fpw = ew
        .clone()
        .mask_where(gt_bool.clone(), zero_mask.clone())
        .sum()
        .into_scalar()
        .to_f64();

    // Calculate weighted recall and precision
    const EPSILON: f64 = 1e-8;

    let r = 1.0
        - (ew
            .mask_where(gt_bool.bool_not(), zero_mask)
            .mean()
            .into_scalar())
        .to_f64();
    let p = tpw / (tpw + fpw + EPSILON);

    // Calculate Q score (weighted F-measure)
    (1.0 + beta) * r * p / (beta.mul_add(p, r) + EPSILON)
}

/// Prepares prediction and ground truth data following Python _prepare_data logic.
fn prepare_data<B: Backend>(pred: Tensor<B, 3>, gt: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>) {
    // Squeeze to 2D (remove channel dimension)
    let pred_2d = pred.squeeze(0);
    let gt_2d = gt.squeeze(0);

    // gt = gt > 128 (binary ground truth)
    let gt_binary = gt_2d.greater_elem(128).float();

    // pred = pred / 255 (normalize predictions to [0, 1])
    let pred_norm = pred_2d.div_scalar(255.0);

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

/// Proper Euclidean distance transform using birefnet-util implementation.
fn distance_transform<B: Backend>(gt: Tensor<B, 2, Bool>) -> Tensor<B, 2> {
    // Convert bool tensor to float and reshape to 4D for distance transform
    let binary_image = gt.bool_not().float().unsqueeze::<4>(); // Add batch and channel dims

    // Apply Euclidean distance transform
    let distance = euclidean_distance_transform_simple(binary_image);

    // Squeeze back to 2D
    distance.squeeze::<2>(0).squeeze::<2>(0)
}

/// Apply error dependency transformation.
fn apply_error_dependency<B: Backend>(
    e: Tensor<B, 2>,
    gt: Tensor<B, 2, Bool>,
    _dst: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let mut et = e.clone();

    // For background pixels, use error from nearest foreground pixel
    let bg_mask = gt.bool_not();

    // TODO: Implement proper error dependency mapping using distance transform indices
    // Current: Simplified implementation copying error values
    // Should implement: Use index map from distance transform to propagate
    // error values from nearest foreground pixels as per weighted F-measure algorithm
    et = et.mask_where(bg_mask, e);

    et
}

/// Apply MATLAB-style Gaussian filter for smoothing.
fn gaussian_filter<B: Backend>(
    tensor: Tensor<B, 2>,
    kernel_size: usize,
    sigma: f64,
) -> Tensor<B, 2> {
    // Convert to 4D for filtering
    let input_4d = tensor.unsqueeze::<4>();

    // Apply MATLAB-compatible Gaussian filtering
    let filtered_4d = gaussian_filter_matlab(input_4d, (kernel_size, kernel_size), sigma);

    // Convert back to 2D
    filtered_4d.squeeze::<2>(0).squeeze::<2>(0)
}

/// Calculate minimum error between original and smoothed versions.
fn calculate_min_error<B: Backend>(
    e: Tensor<B, 2>,
    ea: Tensor<B, 2>,
    gt: Tensor<B, 2, Bool>,
) -> Tensor<B, 2> {
    let condition = gt.bool_and(ea.clone().lower(e.clone()));
    e.mask_where(condition, ea)
}

/// Calculate pixel importance weights based on distance from boundaries.
fn calculate_importance_weights<B: Backend>(
    gt: Tensor<B, 2, Bool>,
    dst: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let bg_mask = gt.bool_not();
    let importance = dst
        .mul_scalar(-0.5 / 5.0)
        .exp()
        .mul_scalar(2.0 - 1.0)
        .add_scalar(1.0);

    importance
        .clone()
        .mask_where(bg_mask, importance.ones_like())
}
