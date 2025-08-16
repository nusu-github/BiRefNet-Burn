//! Loss metric implementation for BiRefNet.
//!
//! This module implements a simple loss tracking metric used during
//! BiRefNet training and evaluation.

use core::marker::PhantomData;

use burn::{
    tensor::{backend::Backend, cast::ToElement},
    train::metric::{
        state::{FormatOptions, NumericMetricState},
        Metric, MetricEntry, MetricMetadata, Numeric,
    },
};

use super::input::BiRefNetLossInput;

// --- Loss Metric ---

#[derive(Default)]
pub struct LossMetric<B: Backend> {
    state: NumericMetricState,
    _b: PhantomData<B>,
}

impl<B: Backend> LossMetric<B> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<B: Backend> Metric for LossMetric<B> {
    type Input = BiRefNetLossInput<B>;

    fn name(&self) -> String {
        "Loss".to_owned()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
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
    fn value(&self) -> f64 {
        self.state.value()
    }
}
