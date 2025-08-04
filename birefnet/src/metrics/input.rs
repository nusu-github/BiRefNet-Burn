//! Input structures for BiRefNet metrics.
//!
//! This module contains the input structures used by various metrics
//! to pass prediction and target tensors along with other required data.

use burn::{prelude::*, tensor::backend::Backend};

// --- Input Structs for Metrics ---

pub struct FMeasureInput<B: Backend> {
    pub predictions: Tensor<B, 4>,
    pub targets: Tensor<B, 4>,
}

impl<B: Backend> FMeasureInput<B> {
    pub const fn new(predictions: Tensor<B, 4>, targets: Tensor<B, 4>) -> Self {
        Self {
            predictions,
            targets,
        }
    }
}

pub struct MAEInput<B: Backend> {
    pub predictions: Tensor<B, 4>,
    pub targets: Tensor<B, 4>,
}

impl<B: Backend> MAEInput<B> {
    pub const fn new(predictions: Tensor<B, 4>, targets: Tensor<B, 4>) -> Self {
        Self {
            predictions,
            targets,
        }
    }
}

pub struct IoUInput<B: Backend> {
    pub predictions: Tensor<B, 4>,
    pub targets: Tensor<B, 4>,
}

impl<B: Backend> IoUInput<B> {
    pub const fn new(predictions: Tensor<B, 4>, targets: Tensor<B, 4>) -> Self {
        Self {
            predictions,
            targets,
        }
    }
}

pub struct BiRefNetLossInput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub batch_size: usize,
}

impl<B: Backend> BiRefNetLossInput<B> {
    pub const fn new(loss: Tensor<B, 1>, batch_size: usize) -> Self {
        Self { loss, batch_size }
    }
}
