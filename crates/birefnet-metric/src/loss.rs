//! Loss metric implementation for BiRefNet.
//!
//! This module implements a simple loss tracking metric used during
//! BiRefNet training and evaluation.

use core::marker::PhantomData;
use std::sync::Arc;

use burn::{
    tensor::{backend::Backend, cast::ToElement},
    train::metric::{
        Metric, MetricMetadata, Numeric, NumericEntry,
        state::{FormatOptions, NumericMetricState},
    },
};

use super::input::BiRefNetLossInput;

// --- Loss Metric ---

#[derive(Default, Clone)]
pub struct LossMetric<B: Backend> {
    state: NumericMetricState,
    name: Arc<String>,
    _b: PhantomData<B>,
}

impl<B: Backend> LossMetric<B> {
    pub fn new() -> Self {
        Self {
            state: NumericMetricState::default(),
            name: Arc::new("Loss".to_owned()),
            _b: PhantomData,
        }
    }
}

impl<B: Backend> Metric for LossMetric<B> {
    type Input = BiRefNetLossInput<B>;

    fn name(&self) -> Arc<String> {
        self.name.clone()
    }

    fn update(
        &mut self,
        item: &Self::Input,
        _metadata: &MetricMetadata,
    ) -> burn::train::metric::SerializedEntry {
        let loss = item.loss.clone().into_scalar().to_f64();
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

impl<B: Backend> Numeric for LossMetric<B> {
    fn value(&self) -> NumericEntry {
        self.state.current_value()
    }

    fn running_value(&self) -> NumericEntry {
        self.state.running_value()
    }
}
