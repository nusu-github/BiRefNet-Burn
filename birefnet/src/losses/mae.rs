//! Mean Absolute Error (L1) loss.

use burn::{
    prelude::*,
    tensor::{backend::Backend, Tensor},
};

/// Configuration for MAE Loss.
#[derive(Config, Debug)]
pub struct MaeLossConfig {
    #[config(default = 1.0)]
    pub weight: f32,
}

/// Mean Absolute Error (L1) loss.
#[derive(Module, Debug)]
pub struct MaeLoss<B: Backend> {
    pub weight: f32,
    _phantom: std::marker::PhantomData<B>,
}

impl MaeLossConfig {
    /// Initialize a new MAE loss with the given configuration.
    pub const fn init<B: Backend>(&self) -> MaeLoss<B> {
        MaeLoss {
            weight: self.weight,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Default for MaeLoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> MaeLoss<B> {
    /// Create a new MAE loss with default configuration.
    pub fn new() -> Self {
        MaeLossConfig::new().init()
    }

    /// Calculate MAE loss.
    pub fn forward(&self, pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        (pred - target).abs().mean() * self.weight
    }
}
