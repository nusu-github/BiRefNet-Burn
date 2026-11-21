//! Mean Absolute Error (`MAE`) metric for `BiRefNet`.

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
    input::MAEInput,
};

/// Configuration for [`MAEMetric`].
#[derive(Config, Debug, Default)]
pub struct MAEMetricConfig {
    /// Display name for the metric.
    #[config(default = "String::from(\"MAE\")")]
    pub name: String,
}

impl MAEMetricConfig {
    /// Initializes an [`MAEMetric`] from this configuration.
    #[must_use]
    pub fn init(&self) -> MAEMetric {
        MAEMetric {
            state: NumericMetricState::default(),
            name: Arc::new(self.name.clone()),
        }
    }
}

/// Mean Absolute Error metric over normalized `[0, 1]` segmentation predictions.
#[derive(Clone)]
pub struct MAEMetric {
    state: NumericMetricState,
    name: Arc<String>,
}

impl fmt::Debug for MAEMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MAEMetric")
            .field("name", &self.name)
            .field("value", &self.value())
            .finish_non_exhaustive()
    }
}

impl Default for MAEMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl MAEMetric {
    /// Creates a new [`MAEMetric`] with default name `"MAE"`.
    #[must_use]
    pub fn new() -> Self {
        MAEMetricConfig::new().init()
    }

    /// Creates a new [`MAEMetric`] from `config`.
    #[must_use]
    pub fn with_config(config: &MAEMetricConfig) -> Self {
        config.init()
    }
}

impl Metric for MAEMetric {
    type Input = MAEInput;

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
            let mut total_mae = 0.0;
            for b in 0..batch_size {
                let pred = extract_sample_3d(&item.predictions, b);
                let gt = extract_sample_3d(&item.targets, b);
                total_mae += calculate_mae_tensor(pred, gt);
            }
            let avg_mae = total_mae / usize_to_f64(batch_size);
            self.state.update(avg_mae, batch_size);
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

impl Numeric for MAEMetric {
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

fn calculate_mae_tensor<const D: usize>(predictions: Tensor<D>, targets: Tensor<D>) -> f64 {
    let (pred, gt) = prepare_data(predictions, targets);
    (pred - gt).abs().mean().into_scalar::<f64>()
}

/// Computes Mean Absolute Error for a single 2D `[H, W]` prediction-target pair.
#[must_use]
pub fn calculate_mae(predictions: Tensor<2>, targets: Tensor<2>) -> f64 {
    calculate_mae_tensor(predictions, targets)
}
