//! Additional operations for the Burn deep learning framework
//!
//! This crate provides operations that are commonly used in deep learning but are not
//! yet available in the core Burn framework.

use burn::prelude::*;

pub mod drop_path;
pub mod erfinv;
pub mod identity;
pub mod slice;
pub mod trunc_normal;

// Convenient re-exports
pub use drop_path::{DropPath, DropPathConfig};
pub use erfinv::{erfinv, Erfinv};
pub use identity::Identity;
pub use slice::{slice_tensor, Slice};
pub use trunc_normal::{trunc_normal, trunc_normal_};

/// Additional operations for Burn tensors
pub trait TensorExtraOps<B: Backend, const D: usize> {
    /// Apply drop path (stochastic depth) to the tensor
    fn drop_path(self, drop_prob: f64, training: bool) -> Self;

    /// Slice the tensor with advanced indexing
    fn slice_advanced(self, ranges: &[std::ops::Range<usize>]) -> Self;
}

impl<B: Backend, const D: usize> TensorExtraOps<B, D> for Tensor<B, D> {
    fn drop_path(self, drop_prob: f64, training: bool) -> Self {
        if !training || drop_prob == 0.0 {
            return self;
        }

        // TODO: Implement drop path
        self
    }

    fn slice_advanced(self, ranges: &[std::ops::Range<usize>]) -> Self {
        slice_tensor(self, ranges)
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
        let result = tensor.clone().drop_path(0.1, false);
        assert_eq!(result.dims(), tensor.dims());

        // Test slice_advanced (placeholder)
        let ranges = vec![0..1, 0..2, 0..3, 0..4];
        let result = tensor.slice_advanced(&ranges);
        assert_eq!(result.dims(), [2, 3, 4, 5]);
    }
}
