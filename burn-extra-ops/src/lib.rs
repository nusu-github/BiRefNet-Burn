//! Additional operations for the Burn deep learning framework
//!
//! This crate provides operations that are commonly used in deep learning but are not
//! yet available in the core Burn framework.

use burn::prelude::*;

mod drop_path;
mod erfinv;
mod identity;
mod trunc_normal;

// Convenient re-exports
pub use drop_path::{DropPath, DropPathConfig};
pub use erfinv::{erfinv, Erfinv};
pub use identity::Identity;
pub use trunc_normal::{trunc_normal, trunc_normal_};

/// Additional operations for Burn tensors
pub trait TensorExtraOps<B: Backend, const D: usize> {
    /// Apply drop path (stochastic depth) to the tensor
    fn drop_path(self, drop_prob: f64, training: bool) -> Self;
}

impl<B: Backend, const D: usize> TensorExtraOps<B, D> for Tensor<B, D> {
    fn drop_path(self, drop_prob: f64, training: bool) -> Self {
        DropPathConfig::new()
            .with_drop_prob(drop_prob)
            .with_training(training)
            .init(&self.device())
            .forward(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::{ndarray::NdArray, Autodiff},
        tensor::Tensor,
    };

    type TestBackend = Autodiff<NdArray<f32>>;

    #[test]
    fn test_tensor_extra_ops() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 4>::random(
            [2, 3, 4, 5],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        // Test drop_path (should return the same tensor when not training)
        let result_no_drop = tensor.clone().drop_path(0.1, false);
        assert_eq!(result_no_drop.dims(), tensor.dims());
    }
}
