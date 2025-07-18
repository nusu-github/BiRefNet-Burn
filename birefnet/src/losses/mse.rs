//! Mean Squared Error (L2) loss.

use burn::{
    prelude::*,
    tensor::{backend::Backend, Tensor},
};

/// Configuration for MSE Loss.
#[derive(Config, Debug)]
pub struct MseLossConfig {
    #[config(default = 1.0)]
    pub weight: f32,
}

/// Mean Squared Error (L2) loss.
#[derive(Module, Debug)]
pub struct MseLoss<B: Backend> {
    pub weight: f32,
    _phantom: std::marker::PhantomData<B>,
}

impl MseLossConfig {
    /// Initialize a new MSE loss with the given configuration.
    pub const fn init<B: Backend>(&self) -> MseLoss<B> {
        MseLoss {
            weight: self.weight,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Default for MseLoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> MseLoss<B> {
    /// Create a new MSE loss with default configuration.
    pub fn new() -> Self {
        MseLossConfig::new().init()
    }

    /// Calculate MSE loss.
    pub fn forward(&self, pred: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        (pred - target).powf_scalar(2.0).mean() * self.weight
    }
}
