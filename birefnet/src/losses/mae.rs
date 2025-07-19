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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_mae_loss_identical_tensors() {
        let device = Default::default();
        let loss = MaeLoss::<TestBackend>::new();

        let pred = Tensor::<TestBackend, 4>::ones([1, 1, 2, 2], &device);
        let target = Tensor::<TestBackend, 4>::ones([1, 1, 2, 2], &device);

        let result = loss.forward(pred, target);
        let loss_value = result.into_scalar().elem::<f32>();

        assert!(
            (loss_value - 0.0).abs() < 1e-6,
            "Loss should be 0 for identical tensors"
        );
    }

    #[test]
    fn test_mae_loss_different_tensors() {
        let device = Default::default();
        let loss = MaeLoss::<TestBackend>::new();

        let pred = Tensor::<TestBackend, 4>::ones([1, 1, 2, 2], &device);
        let target = Tensor::<TestBackend, 4>::zeros([1, 1, 2, 2], &device);

        let result = loss.forward(pred, target);
        let loss_value = result.into_scalar().elem::<f32>();

        // Expected: |1 - 0| = 1, mean of all ones = 1
        assert!(
            (loss_value - 1.0).abs() < 1e-6,
            "Loss should be 1.0 for unit difference"
        );
    }

    #[test]
    fn test_mae_loss_with_weight() {
        let device = Default::default();
        let config = MaeLossConfig::new().with_weight(2.0);
        let loss = config.init::<TestBackend>();

        let pred = Tensor::<TestBackend, 4>::ones([1, 1, 2, 2], &device);
        let target = Tensor::<TestBackend, 4>::zeros([1, 1, 2, 2], &device);

        let result = loss.forward(pred, target);
        let loss_value = result.into_scalar().elem::<f32>();

        // Expected: |1 - 0| = 1, mean = 1, weight = 2.0, total = 2.0
        assert!(
            (loss_value - 2.0).abs() < 1e-6,
            "Loss should respect weight parameter"
        );
    }

    #[test]
    fn test_mae_loss_custom_values() {
        let device = Default::default();
        let loss = MaeLoss::<TestBackend>::new();

        // Create tensors with known values
        let pred = Tensor::<TestBackend, 4>::from_floats([[[[2.0, 4.0], [6.0, 8.0]]]], &device);
        let target = Tensor::<TestBackend, 4>::from_floats([[[[1.0, 2.0], [3.0, 4.0]]]], &device);

        let result = loss.forward(pred, target);
        let loss_value = result.into_scalar().elem::<f32>();

        // Expected: |2-1| + |4-2| + |6-3| + |8-4| = 1 + 2 + 3 + 4 = 10, mean = 10/4 = 2.5
        assert!(
            (loss_value - 2.5).abs() < 1e-6,
            "Loss calculation should be correct for custom values"
        );
    }
}
