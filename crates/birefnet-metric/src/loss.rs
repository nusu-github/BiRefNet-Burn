//! Loss metric implementation for BiRefNet.
//!
//! This module implements a simple loss tracking metric used during
//! BiRefNet training and evaluation.

use core::marker::PhantomData;
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
    _b: PhantomData,
}

impl LossMetric {
    pub fn new() -> Self {
        Self {
            state: NumericMetricState::default(),
            name: Arc::new("Loss".to_owned()),
            _b: PhantomData,
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
    ) -> burn::train::metric::SerializedEntry {
        let loss = item.loss.clone().into_scalar::<f32>().to_f64();
        self.state.update(
            loss,
            item.batch_size,
            FormatOptions::new(self.name()).precision(5),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl Numeric for LossMetric {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}
