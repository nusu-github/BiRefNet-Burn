//! MAE (Mean Absolute Error) metric implementation for BiRefNet.
//!
//! This module implements the Mean Absolute Error metric used in BiRefNet
//! for evaluating segmentation performance.

use burn::{
    prelude::*,
    tensor::{backend::Backend, ElementConversion, Tensor},
    train::metric::{
        state::{FormatOptions, NumericMetricState},
        Metric, MetricEntry, MetricMetadata, Numeric,
    },
};
use std::marker::PhantomData;

use crate::metrics::input::MAEInput;

// --- MAE Metric ---

#[derive(Config, Debug)]
pub struct MAEMetricConfig {
    #[config(default = true)]
    pub apply_sigmoid: bool,
}

pub struct MAEMetric<B: Backend> {
    state: NumericMetricState,
    apply_sigmoid: bool,
    _b: PhantomData<B>,
}

impl MAEMetricConfig {
    pub fn init<B: Backend>(&self) -> MAEMetric<B> {
        MAEMetric {
            state: NumericMetricState::default(),
            apply_sigmoid: self.apply_sigmoid,
            _b: PhantomData,
        }
    }
}

impl<B: Backend> Default for MAEMetric<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> MAEMetric<B> {
    pub fn new() -> Self {
        MAEMetricConfig::new().init()
    }
}

impl<B: Backend> Metric for MAEMetric<B> {
    type Input = MAEInput<B>;

    fn name(&self) -> String {
        "MAE".to_string()
    }

    fn update(&mut self, item: &Self::Input, _metadata: &MetricMetadata) -> MetricEntry {
        let batch_size = item.predictions.dims()[0];
        let preds_processed = if self.apply_sigmoid {
            burn::tensor::activation::sigmoid(item.predictions.clone())
        } else {
            item.predictions.clone()
        };
        let abs_error = (preds_processed - item.targets.clone()).abs().mean();
        let error_value = abs_error.into_scalar().elem::<f64>();
        self.state.update(
            error_value,
            batch_size,
            FormatOptions::new(self.name()).precision(5),
        )
    }

    fn clear(&mut self) {
        self.state.reset();
    }
}

impl<B: Backend> Numeric for MAEMetric<B> {
    fn value(&self) -> f64 {
        self.state.value()
    }
}

/// Calculate MAE metric using a simple function interface.
pub fn calculate_mae<B: Backend>(
    predictions: Tensor<B, 4>,
    targets: Tensor<B, 4>,
    apply_sigmoid: bool,
) -> f32 {
    let preds_processed = if apply_sigmoid {
        burn::tensor::activation::sigmoid(predictions)
    } else {
        predictions
    };

    let mae = (preds_processed - targets).abs().mean();
    mae.into_scalar().elem::<f32>()
}
