//! Loss metric implementation for BiRefNet.
//!
//! This module implements a simple loss tracking metric used during
//! BiRefNet training and evaluation.

use std::sync::Arc;

use burn::{
    tensor::cast::ToElement,
    train::metric::{
        Metric, MetricMetadata, Numeric, NumericEntry,
        state::{FormatOptions, NumericMetricState},
    },
};

use super::input::BiRefNetLossInput;

// --- Loss Metric ---

#[derive(Default, Clone)]
pub struct LossMetric {
    state: NumericMetricState,
    name: Arc<String>,
}

impl LossMetric {
    pub fn new() -> Self {
        Self {
            state: NumericMetricState::default(),
            name: Arc::new("Loss".to_owned()),
        }
    }
}

impl Metric for LossMetric {
    type Input = BiRefNetLossInput;

    fn name(&self) -> Arc<String> {
        self.name.clone()
    }

    fn update(
        &mut self,
        item: &Self::Input,
        _metadata: &MetricMetadata,
    ) -> Result<burn::train::metric::SerializedEntry, burn::tensor::TensorReadError> {
        let loss = item.loss.clone().into_scalar::<f32>().to_f64();
        self.state.update(loss, item.batch_size);
        Ok(self
            .state
            .compute_update(FormatOptions::new(self.name()).precision(5)))
    }

    fn compute(
        &mut self,
    ) -> Result<burn::train::metric::SerializedEntry, burn::tensor::TensorReadError> {
        Ok(self
            .state
            .compute_final(FormatOptions::new(self.name()).precision(5)))
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl Numeric for LossMetric {
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
