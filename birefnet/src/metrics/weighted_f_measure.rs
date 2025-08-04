//! Weighted F-measure metric for BiRefNet.
//!
//! WF-measure evaluates segmentation quality with pixel importance weighting
//! based on distance from ground truth boundaries.

use burn::tensor::cast::ToElement;
use burn::{
    config::Config,
    tensor::{backend::Backend, Bool, ElementConversion, Int, Tensor},
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
    beta: f32,
}

/// Weighted F-measure metric input.
#[derive(Debug, Clone)]
pub struct WeightedFMeasureInput<B: Backend> {
    /// Predictions with shape `[batch_size, height, width]`.
    pub predictions: Tensor<B, 3>,
    /// Ground truth with shape `[batch_size, height, width]`.
    pub targets: Tensor<B, 3>,
}

impl<B: Backend> WeightedFMeasureInput<B> {
    /// Creates a new Weighted F-measure input.
    pub const fn new(predictions: Tensor<B, 3>, targets: Tensor<B, 3>) -> Self {
        Self {
            predictions,
            targets,
        }
    }
}

/// Weighted F-measure metric.
pub struct WeightedFMeasureMetric<B: Backend> {
    state: NumericMetricState,
    name: String,
    beta: f32,
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> Default for WeightedFMeasureMetric<B> {
    fn default() -> Self {
        Self {
            state: NumericMetricState::default(),
            name: "WF_measure".to_string(),
            beta: 1.0,
            _backend: std::marker::PhantomData,
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
            _backend: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Metric for WeightedFMeasureMetric<B> {
    type Input = WeightedFMeasureInput<B>;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let [batch_size, height, width] = input.predictions.dims();

        let mut total_wfm = 0.0;

        for b in 0..batch_size {
            let pred = input
                .predictions
                .clone()
                .slice([b..b + 1, 0..height, 0..width])
                .squeeze(0);
            let gt = input
                .targets
                .clone()
                .slice([b..b + 1, 0..height, 0..width])
                .squeeze(0);

            let wfm = calculate_weighted_f_measure(pred, gt, self.beta);
            total_wfm += wfm as f64;
        }

        let avg_wfm = total_wfm / batch_size as f64;
        self.state.update(
            avg_wfm,
            batch_size,
            FormatOptions::new(self.name.clone()).precision(5),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }
}

impl<B: Backend> Numeric for WeightedFMeasureMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

/// Calculates Weighted F-measure for a single prediction-target pair.
///
/// # Arguments
/// * `predictions` - Predictions with shape `[height, width]`.
/// * `targets` - Ground truth with shape `[height, width]`.
/// * `beta` - Beta value for F-measure calculation.
///
/// # Returns
/// The Weighted F-measure value.
pub fn calculate_weighted_f_measure<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
    beta: f32,
) -> f32 {
    // Prepare data
    let gt = targets.div_scalar(255.0).greater_equal_elem(0.5);

    // Check if all background
    if gt.clone().float().sum().into_scalar().elem::<f32>() == 0.0 {
        return 0.0;
    }

    // Normalize predictions if needed
    let min_val = predictions.clone().min();
    let max_val = predictions.clone().max();
    let range = max_val - min_val.clone();
    let epsilon = 1e-8;

    let pred = if range.clone().greater_elem(epsilon).into_scalar().to_bool() {
        let min_scalar = min_val.into_scalar().elem::<f32>();
        let range_scalar = range.into_scalar().elem::<f32>();
        predictions.sub_scalar(min_scalar).div_scalar(range_scalar)
    } else {
        predictions
    };

    // Calculate distance transform for background pixels
    let dst = distance_transform(&gt);

    // Calculate error
    let e = (pred - gt.clone().float()).abs();

    // Apply error dependency transformation
    let et = apply_error_dependency(&e, &gt, &dst);

    // Apply Gaussian filter for smoothing
    let ea = gaussian_filter(&et, 7, 5.0);

    // Calculate minimum error
    let min_e_ea = calculate_min_error(&e, &ea, &gt);

    // Calculate pixel importance weights
    let b = calculate_importance_weights(&gt, &dst);

    // Calculate weighted error
    let ew = min_e_ea * b;

    // Calculate weighted metrics
    let gt_sum = gt.clone().float().sum().into_scalar().elem::<f32>();
    let zero_mask = ew.zeros_like();
    let tpw = gt_sum
        - ew.clone()
            .mask_where(gt.clone().bool_not(), zero_mask.clone())
            .sum()
            .into_scalar()
            .elem::<f32>();
    let fpw = ew
        .clone()
        .mask_where(gt.clone(), zero_mask.clone())
        .sum()
        .into_scalar()
        .elem::<f32>();

    // Calculate weighted recall and precision
    let r = 1.0 - (ew.mask_where(gt.bool_not(), zero_mask).mean().into_scalar()).elem::<f32>();
    let p = tpw / (tpw + fpw + epsilon);

    // Calculate Q score (weighted F-measure)
    (1.0 + beta) * r * p / (beta.mul_add(p, r) + epsilon)
}

/// TODO: Implement proper Euclidean distance transform for accurate weighted F-measure.
/// Current implementation uses simplified approximation.
/// Should implement: Fast marching method or chamfer distance transform
/// for accurate boundary distance calculation as per weighted F-measure paper.
fn distance_transform<B: Backend>(gt: &Tensor<B, 2, Bool>) -> Tensor<B, 2> {
    let [height, width] = gt.dims();
    let device = gt.device();

    // Create coordinate grids
    let y_coords: Tensor<B, 2> = Tensor::<B, 1, Int>::arange(0..height as i64, &device)
        .float()
        .unsqueeze_dim(1);
    let x_coords: Tensor<B, 2> = Tensor::<B, 1, Int>::arange(0..width as i64, &device)
        .float()
        .unsqueeze_dim(0);

    let y_grid = y_coords.expand([height, width]);
    let x_grid = x_coords.expand([height, width]);

    // TODO: Implement proper boundary detection using morphological operations
    // Current: Using simple dilation-based approximation
    // Should implement: Sobel edge detection or proper contour finding
    let kernel = Tensor::<B, 2>::ones([3, 3], &device);
    let dilated = conv2d_simple(
        &gt.clone().float().unsqueeze_dims(&[0, 0]),
        &kernel.unsqueeze_dims(&[0, 0]),
        1,
    );
    let _boundary = dilated
        .squeeze_dims(&[0, 0])
        .greater_elem(0.0)
        .bool_and(gt.clone().bool_not());

    // TODO: Replace with proper distance transform algorithm
    // Current: Using center-of-mass approximation which is inaccurate
    // Should implement: Fast marching method, chamfer distance, or EDT
    // for pixel-wise accurate distance computation
    let _bg_mask = gt.clone().bool_not();
    let _max_dist = ((height * height + width * width) as f32).sqrt();

    // Simple approximation: use distance from center of mass of foreground
    let fg_y_scalar = (y_grid.clone() * gt.clone().float())
        .sum()
        .div_scalar(
            (gt.clone().float().sum() + 1e-8)
                .into_scalar()
                .elem::<f32>(),
        )
        .into_scalar()
        .elem::<f32>();
    let fg_x_scalar = (x_grid.clone() * gt.clone().float())
        .sum()
        .div_scalar(
            (gt.clone().float().sum() + 1e-8)
                .into_scalar()
                .elem::<f32>(),
        )
        .into_scalar()
        .elem::<f32>();

    let dist_y = y_grid - fg_y_scalar;
    let dist_x = x_grid - fg_x_scalar;
    let dist = (dist_y.powf_scalar(2.0) + dist_x.powf_scalar(2.0)).sqrt();

    dist.clone().mask_where(gt.clone(), dist.zeros_like())
}

/// Apply error dependency transformation.
fn apply_error_dependency<B: Backend>(
    e: &Tensor<B, 2>,
    gt: &Tensor<B, 2, Bool>,
    _dst: &Tensor<B, 2>,
) -> Tensor<B, 2> {
    let mut et = e.clone();

    // For background pixels, use error from nearest foreground pixel
    let bg_mask = gt.clone().bool_not();

    // TODO: Implement proper error dependency mapping using distance transform indices
    // Current: Simplified implementation copying error values
    // Should implement: Use index map from distance transform to propagate
    // error values from nearest foreground pixels as per weighted F-measure algorithm
    et = et.clone().mask_where(bg_mask, e.clone());

    et
}

/// Apply Gaussian filter for smoothing.
fn gaussian_filter<B: Backend>(
    tensor: &Tensor<B, 2>,
    kernel_size: usize,
    sigma: f32,
) -> Tensor<B, 2> {
    let device = tensor.device();

    // Create Gaussian kernel
    let half_size = (kernel_size as i32 - 1) / 2;
    let mut kernel_data = vec![0.0f32; kernel_size * kernel_size];

    let mut sum = 0.0;
    for i in 0..kernel_size {
        for j in 0..kernel_size {
            let x = (i as i32 - half_size) as f32;
            let y = (j as i32 - half_size) as f32;
            let value = (-x.mul_add(x, y * y) / (2.0 * sigma * sigma)).exp();
            kernel_data[i * kernel_size + j] = value;
            sum += value;
        }
    }

    // Normalize kernel
    for val in &mut kernel_data {
        *val /= sum;
    }

    let kernel = Tensor::<B, 2>::from_data(kernel_data.as_slice(), &device)
        .reshape([kernel_size, kernel_size]);

    // Apply convolution
    conv2d_simple(
        &tensor.clone().unsqueeze_dims(&[0, 0]),
        &kernel.unsqueeze_dims(&[0, 0]),
        half_size as usize,
    )
    .squeeze_dims(&[0, 0])
}

/// Simple 2D convolution (for demonstration - in practice, use proper conv2d).
fn conv2d_simple<B: Backend>(
    input: &Tensor<B, 4>,
    _kernel: &Tensor<B, 4>,
    _padding: usize,
) -> Tensor<B, 4> {
    // This is a placeholder for actual convolution
    // In practice, you would use burn's conv2d module
    input.clone()
}

/// Calculate minimum error between original and smoothed versions.
fn calculate_min_error<B: Backend>(
    e: &Tensor<B, 2>,
    ea: &Tensor<B, 2>,
    gt: &Tensor<B, 2, Bool>,
) -> Tensor<B, 2> {
    let condition = gt.clone().bool_and(ea.clone().lower(e.clone()));
    e.clone().mask_where(condition, ea.clone())
}

/// Calculate pixel importance weights based on distance from boundaries.
fn calculate_importance_weights<B: Backend>(
    gt: &Tensor<B, 2, Bool>,
    dst: &Tensor<B, 2>,
) -> Tensor<B, 2> {
    let bg_mask = gt.clone().bool_not();
    let importance = dst
        .clone()
        .mul_scalar(-0.5 / 5.0)
        .exp()
        .mul_scalar(2.0 - 1.0)
        .add_scalar(1.0);

    importance
        .clone()
        .mask_where(bg_mask, importance.ones_like())
}
