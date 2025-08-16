//! Input structures for BiRefNet metrics.
//!
//! This module contains the input structures used by various metrics
//! to pass prediction and target tensors along with other required data.

use burn::{prelude::*, tensor::backend::Backend};
use derive_new::new;

// --- Input Structs for Metrics ---

/// F-measure metric input.
#[derive(new, Debug, Clone)]
pub struct FMeasureInput<B: Backend> {
    /// Predictions with shape `[batch_size, channels, height, width]`.
    pub predictions: Tensor<B, 4>,
    /// Ground truth with shape `[batch_size, channels, height, width]`.
    pub targets: Tensor<B, 4>,
}

/// MAE metric input.
#[derive(new, Debug, Clone)]
pub struct MAEInput<B: Backend> {
    /// Predictions with shape `[batch_size, channels, height, width]`.
    pub predictions: Tensor<B, 4>,
    /// Ground truth with shape `[batch_size, channels, height, width]`.
    pub targets: Tensor<B, 4>,
}

/// MSE metric input.
#[derive(new, Debug, Clone)]
pub struct MSEInput<B: Backend> {
    /// Predictions with shape `[batch_size, channels, height, width]`.
    pub predictions: Tensor<B, 4>,
    /// Ground truth with shape `[batch_size, channels, height, width]`.
    pub targets: Tensor<B, 4>,
}

/// BIoU metric input.
#[derive(new, Debug, Clone)]
pub struct BIoUInput<B: Backend> {
    /// Predictions with shape `[batch_size, channels, height, width]`.
    pub predictions: Tensor<B, 4>,
    /// Ground truth with shape `[batch_size, channels, height, width]`.
    pub targets: Tensor<B, 4>,
}

/// Weighted F-measure metric input.
#[derive(new, Debug, Clone)]
pub struct WeightedFMeasureInput<B: Backend> {
    /// Predictions with shape `[batch_size, channels, height, width]`.
    pub predictions: Tensor<B, 4>,
    /// Ground truth with shape `[batch_size, channels, height, width]`.
    pub targets: Tensor<B, 4>,
}

/// S-measure metric input.
#[derive(new, Debug, Clone)]
pub struct SMeasureInput<B: Backend> {
    /// Predictions with shape `[batch_size, channels, height, width]`.
    pub predictions: Tensor<B, 4>,
    /// Ground truth with shape `[batch_size, channels, height, width]`.
    pub targets: Tensor<B, 4>,
}

/// E-measure metric input.
#[derive(new, Debug, Clone)]
pub struct EMeasureInput<B: Backend> {
    /// Predictions with shape `[batch_size, channels, height, width]`.
    pub predictions: Tensor<B, 4>,
    /// Ground truth with shape `[batch_size, channels, height, width]`.
    pub targets: Tensor<B, 4>,
}

/// Loss metric input for BiRefNet.
#[derive(new, Debug, Clone)]
pub struct BiRefNetLossInput<B: Backend> {
    /// Loss tensor with shape `[batch_size]` or scalar.
    pub loss: Tensor<B, 1>,
    /// Batch size for averaging.
    pub batch_size: usize,
}
