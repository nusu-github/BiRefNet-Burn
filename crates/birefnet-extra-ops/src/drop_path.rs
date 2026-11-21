//! Stochastic depth (`DropPath`) regularization module.

use burn::{
    module::{Flag, Param},
    prelude::*,
    tensor::Distribution,
};

/// Configuration for [`DropPath`].
#[derive(Config, Debug, PartialEq)]
pub struct DropPathConfig {
    /// Probability in `[0.0, 1.0]` of zeroing each sample's residual branch during training.
    #[config(default = 0.0)]
    pub drop_prob: f64,
    /// Whether surviving samples are scaled by `1 / (1 - drop_prob)` to preserve expectation.
    #[config(default = true)]
    pub scale_by_keep: bool,
}

impl Default for DropPathConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl DropPathConfig {
    /// Initializes a [`DropPath`] module from this configuration.
    ///
    /// # Panics
    ///
    /// Panics if `drop_prob` is outside `[0.0, 1.0]` or non-finite.
    #[must_use]
    pub fn init(&self) -> DropPath {
        DropPath::new(self.drop_prob, self.scale_by_keep)
    }
}

/// Stochastic depth module that drops entire batch samples during training.
///
/// Like `burn::nn::Dropout`, it is active only when its training flag is enabled
/// (cleared by [`Module::valid`]) and the input tensor lives on an autodiff device.
#[derive(Module, Debug)]
pub struct DropPath {
    drop_prob: f64,
    scale_by_keep: bool,
    training: Param<Flag>,
}

impl DropPath {
    /// Creates a [`DropPath`] module directly from `drop_prob` and `scale_by_keep`.
    ///
    /// # Panics
    ///
    /// Panics if `drop_prob` is outside `[0.0, 1.0]` or non-finite.
    #[must_use]
    pub fn new(drop_prob: f64, scale_by_keep: bool) -> Self {
        assert!(
            (0.0..=1.0).contains(&drop_prob),
            "DropPath: drop_prob must be in [0.0, 1.0], got {drop_prob}"
        );
        Self {
            drop_prob,
            scale_by_keep,
            training: Param::from_bool(true),
        }
    }

    /// Applies stochastic depth along axis 0 of `x`.
    ///
    /// Returns `x` unchanged in evaluation mode, on non-autodiff devices, or when `drop_prob == 0.0`.
    ///
    /// # Panics
    ///
    /// Triggers a compile-time panic if `D == 0`.
    #[must_use]
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        const { assert!(D >= 1, "DropPath requires tensor rank >= 1") };

        if !self.training.is_enabled() || !x.device().is_autodiff() || self.drop_prob == 0.0 {
            return x;
        }
        if self.drop_prob >= 1.0 {
            return x.zeros_like();
        }

        let keep_prob = 1.0 - self.drop_prob;
        let mut mask_shape = [1; D];
        mask_shape[0] = x.dims()[0];

        let keep_mask = Tensor::random(mask_shape, Distribution::Bernoulli(keep_prob), &x.device());
        if self.scale_by_keep {
            x * keep_mask / keep_prob
        } else {
            x * keep_mask
        }
    }
}
