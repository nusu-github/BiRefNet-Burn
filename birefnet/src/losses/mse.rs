//! Mean Squared Error (L2) loss.

use burn::nn::loss::{MseLoss as BurnMseLoss, Reduction};
use burn::{
    prelude::*,
    tensor::{backend::Backend, Tensor},
};

/// Configuration for MSE Loss.
#[derive(Config, Debug)]
pub struct MseLossConfig {}

/// Mean Squared Error (L2) loss.
#[derive(Module, Debug)]
pub struct MseLoss<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl MseLossConfig {
    /// Initialize a new MSE loss with the given configuration.
    pub const fn init<B: Backend>(&self) -> MseLoss<B> {
        MseLoss {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> MseLoss<B> {
    /// Calculate MSE loss.
    pub fn forward<const D: usize>(
        &self,
        pred: Tensor<B, D>,
        target: Tensor<B, D>,
    ) -> Tensor<B, 1> {
        BurnMseLoss::new().forward(pred, target, Reduction::Auto)
    }
}
