//! Mean Squared Error (MSE) metric for BiRefNet.

use burn::{
    config::Config,
    tensor::{backend::Backend, cast::ToElement, ElementConversion, Tensor},
    train::metric::{
        state::{FormatOptions, NumericMetricState},
        Metric, MetricEntry, MetricMetadata, Numeric,
    },
};

/// Configuration for the MSE metric.
#[derive(Config, Debug)]
pub struct MSEMetricConfig {
    /// Name of the metric (default: "MSE").
    #[config(default = "String::from(\"MSE\")")]
    name: String,
}

/// MSE metric input.
#[derive(Debug, Clone)]
pub struct MSEInput<B: Backend> {
    /// Predictions with shape `[batch_size, height, width]`.
    pub predictions: Tensor<B, 3>,
    /// Ground truth with shape `[batch_size, height, width]`.
    pub targets: Tensor<B, 3>,
}

impl<B: Backend> MSEInput<B> {
    /// Creates a new MSE input.
    pub const fn new(predictions: Tensor<B, 3>, targets: Tensor<B, 3>) -> Self {
        Self {
            predictions,
            targets,
        }
    }
}

/// MSE metric.
#[derive(Default)]
pub struct MSEMetric<B: Backend> {
    state: NumericMetricState,
    name: String,
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> MSEMetric<B> {
    /// Creates a new MSE metric.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new MSE metric with a custom name.
    pub fn with_name(name: String) -> Self {
        Self {
            state: NumericMetricState::default(),
            name,
            _backend: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Metric for MSEMetric<B> {
    type Input = MSEInput<B>;

    fn name(&self) -> String {
        self.name.to_string()
    }

    fn update(&mut self, input: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let [batch_size, _height, _width] = input.predictions.dims();

        // Prepare data (normalize and threshold)
        let predictions = input.predictions.clone();
        let targets = input.targets.clone().div_scalar(255.0);
        let targets = targets.greater_equal_elem(0.5).float();

        // Normalize predictions if needed
        let min_val = predictions.clone().min_dim(1).min_dim(1);
        let max_val = predictions.clone().max_dim(1).max_dim(1);
        let range = max_val - min_val.clone();
        let epsilon = 1e-8;

        // Avoid division by zero
        let mask = range.clone().greater_elem(epsilon);
        let normalized_preds = predictions
            .clone()
            .sub(min_val.unsqueeze::<3>())
            .div(range.add_scalar(epsilon).unsqueeze::<3>());

        // Use original predictions where range is too small
        let predictions = mask
            .clone()
            .unsqueeze::<3>()
            .float()
            .mul(normalized_preds)
            .add(mask.bool_not().unsqueeze::<3>().float().mul(predictions));

        // Calculate MSE
        let diff = predictions - targets;
        let squared_diff = diff.powf_scalar(2.0);
        let mse = squared_diff.mean();

        let mse_value = mse.into_scalar().elem::<f64>();
        self.state.update(
            mse_value,
            batch_size,
            FormatOptions::new(self.name.to_string()).precision(5),
        )
    }

    fn clear(&mut self) {
        self.state.reset()
    }
}

impl<B: Backend> Numeric for MSEMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

/// Calculates MSE for a single prediction-target pair.
///
/// # Arguments
/// * `predictions` - Predictions with shape `[height, width]`.
/// * `targets` - Ground truth with shape `[height, width]`.
///
/// # Returns
/// The MSE value.
pub fn calculate_mse<B: Backend>(predictions: Tensor<B, 2>, targets: Tensor<B, 2>) -> f32 {
    // Prepare data
    let targets = targets.div_scalar(255.0);
    let targets = targets.greater_equal_elem(0.5).float();

    // Normalize predictions if needed
    let min_val = predictions.clone().min();
    let max_val = predictions.clone().max();
    let range = max_val - min_val.clone();
    let epsilon = 1e-8;

    let predictions = if range.clone().greater_elem(epsilon).into_scalar().to_bool() {
        let min_scalar = min_val.into_scalar().elem::<f32>();
        let range_scalar = range.into_scalar().elem::<f32>();
        predictions.sub_scalar(min_scalar).div_scalar(range_scalar)
    } else {
        predictions
    };

    // Calculate MSE
    let diff = predictions - targets;
    let squared_diff = diff.powf_scalar(2.0);
    let mse = squared_diff.mean();

    mse.into_scalar().elem::<f32>()
}
