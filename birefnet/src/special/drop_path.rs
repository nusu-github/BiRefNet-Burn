//! # DropPath Regularization
//!
//! Implements the DropPath regularization technique, also known as stochastic depth.
//! During training, it randomly drops entire paths (sub-networks) and scales the
//! remaining ones, effectively preventing co-adaptation of parallel paths.

use burn::{prelude::*, tensor::Distribution};

/// Configuration for the `DropPath` module.
#[derive(Config, Debug)]
pub struct DropPathConfig {
    /// The probability of dropping a path.
    #[config(default = "0.0")]
    drop_prob: f64,
    /// Whether the module is in training mode.
    #[config(default = "false")]
    training: bool,
    /// Whether to scale the output by the keep probability.
    #[config(default = "true")]
    scale_by_keep: bool,
}

impl DropPathConfig {
    /// Initializes a new `DropPath` module.
    pub const fn init(&self) -> DropPath {
        DropPath {
            drop_prob: self.drop_prob,
            training: self.training,
            scale_by_keep: self.scale_by_keep,
        }
    }
}

/// DropPath module.
#[derive(Module, Clone, Debug)]
pub struct DropPath {
    drop_prob: f64,
    training: bool,
    scale_by_keep: bool,
}

impl DropPath {
    /// Applies DropPath to the input tensor.
    ///
    /// If not in training mode or `drop_prob` is 0, it returns the input tensor unchanged.
    /// Otherwise, it randomly zeros out entire examples in the batch with probability `drop_prob`.
    pub fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        if !self.training || self.drop_prob == 0.0 {
            return x;
        }
        let keep_prob = 1.0 - self.drop_prob;
        let other_dims = vec![1; D - 1];
        let shape: Vec<_> = core::iter::once(x.dims()[0]).chain(other_dims).collect();
        let random_tensor = Tensor::random(shape, Distribution::Bernoulli(keep_prob), &x.device());
        if self.scale_by_keep {
            x * random_tensor / keep_prob
        } else {
            x * random_tensor
        }
    }
}
