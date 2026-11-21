//! Mean Squared Error (`MSE`) metric for `BiRefNet`.

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
    input::MSEInput,
};

/// Configuration for [`MSEMetric`].
#[derive(Config, Debug, Default)]
pub struct MSEMetricConfig {
    /// Display name for the metric.
    #[config(default = "String::from(\"MSE\")")]
    pub name: String,
}

impl MSEMetricConfig {
    /// Initializes an [`MSEMetric`] from this configuration.
    #[must_use]
    pub fn init(&self) -> MSEMetric {
        MSEMetric {
            state: NumericMetricState::default(),
            name: Arc::new(self.name.clone()),
        }
    }
}

/// Mean Squared Error metric over normalized `[0, 1]` segmentation predictions.
#[derive(Clone)]
pub struct MSEMetric {
    state: NumericMetricState,
    name: Arc<String>,
}

impl fmt::Debug for MSEMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MSEMetric")
            .field("name", &self.name)
            .field("value", &self.value())
            .finish_non_exhaustive()
    }
}

impl Default for MSEMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl MSEMetric {
    /// Creates a new [`MSEMetric`] with default name `"MSE"`.
    #[must_use]
    pub fn new() -> Self {
        MSEMetricConfig::new().init()
    }

    /// Creates a new [`MSEMetric`] from `config`.
    #[must_use]
    pub fn with_config(config: &MSEMetricConfig) -> Self {
        config.init()
    }
}

impl Metric for MSEMetric {
    type Input = MSEInput;

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
            let mut total_mse = 0.0;
            for b in 0..batch_size {
                let pred = extract_sample_3d(&item.predictions, b);
                let gt = extract_sample_3d(&item.targets, b);
                total_mse += calculate_mse_tensor(pred, gt);
            }
            let avg_mse = total_mse / usize_to_f64(batch_size);
            self.state.update(avg_mse, batch_size);
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
            higher_is_better: false,
        }
        .into()
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl Numeric for MSEMetric {
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

fn calculate_mse_tensor<const D: usize>(predictions: Tensor<D>, targets: Tensor<D>) -> f64 {
    let (pred, gt) = prepare_data(predictions, targets);
    (pred - gt).powf_scalar(2.0).mean().into_scalar::<f64>()
}

/// Computes Mean Squared Error for a single 2D `[H, W]` prediction-target pair.
#[must_use]
pub fn calculate_mse(predictions: Tensor<2>, targets: Tensor<2>) -> f64 {
    calculate_mse_tensor(predictions, targets)
}
