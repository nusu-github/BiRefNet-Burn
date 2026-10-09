//! Input structures for BiRefNet metrics.
//!
//! This module contains the input structures used by various metrics
//! to pass prediction and target tensors along with other required data.

use burn::prelude::*;

// --- Input Structs for Metrics ---

/// F-measure metric input.
#[derive(Debug, Clone)]
pub struct FMeasureInput {
    /// Predictions with shape `[batch_size, channels, height, width]`.
    pub predictions: Tensor<4>,
    /// Ground truth with shape `[batch_size, channels, height, width]`.
    pub targets: Tensor<4>,
}

/// MAE metric input.
#[derive(Debug, Clone)]
pub struct MAEInput {
    /// Predictions with shape `[batch_size, channels, height, width]`.
    pub predictions: Tensor<4>,
    /// Ground truth with shape `[batch_size, channels, height, width]`.
    pub targets: Tensor<4>,
}

/// MSE metric input.
#[derive(Debug, Clone)]
pub struct MSEInput {
    /// Predictions with shape `[batch_size, channels, height, width]`.
    pub predictions: Tensor<4>,
    /// Ground truth with shape `[batch_size, channels, height, width]`.
    pub targets: Tensor<4>,
}

/// BIoU metric input.
#[derive(Debug, Clone)]
pub struct BIoUInput {
    /// Predictions with shape `[batch_size, channels, height, width]`.
    pub predictions: Tensor<4>,
    /// Ground truth with shape `[batch_size, channels, height, width]`.
    pub targets: Tensor<4>,
}

/// Weighted F-measure metric input.
#[derive(Debug, Clone)]
pub struct WeightedFMeasureInput {
    /// Predictions with shape `[batch_size, channels, height, width]`.
    pub predictions: Tensor<4>,
    /// Ground truth with shape `[batch_size, channels, height, width]`.
    pub targets: Tensor<4>,
}

/// S-measure metric input.
#[derive(Debug, Clone)]
pub struct SMeasureInput {
    /// Predictions with shape `[batch_size, channels, height, width]`.
    pub predictions: Tensor<4>,
    /// Ground truth with shape `[batch_size, channels, height, width]`.
    pub targets: Tensor<4>,
}

/// E-measure metric input.
#[derive(Debug, Clone)]
pub struct EMeasureInput {
    /// Predictions with shape `[batch_size, channels, height, width]`.
    pub predictions: Tensor<4>,
    /// Ground truth with shape `[batch_size, channels, height, width]`.
    pub targets: Tensor<4>,
}

macro_rules! impl_new {
    ($($name:ident),* $(,)?) => {
        $(
            impl $name {
                /// Creates the input from predictions and ground truth.
                pub const fn new(predictions: Tensor<4>, targets: Tensor<4>) -> Self {
                    Self {
                        predictions,
                        targets,
                    }
                }
            }
        )*
    };
}

impl_new!(
    FMeasureInput,
    MAEInput,
    MSEInput,
    BIoUInput,
    WeightedFMeasureInput,
    SMeasureInput,
    EMeasureInput
);
