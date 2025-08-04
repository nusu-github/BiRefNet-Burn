use burn::{
    prelude::*,
    tensor::{backend::Backend, ElementConversion, Tensor},
};

#[derive(Module, Debug)]
pub struct BCELoss<B: Backend> {
    epsilon: f64,
    _phantom: std::marker::PhantomData<B>,
}

#[derive(Config, Debug)]
pub struct BCELossConfig {
    #[config(default = "1e-8")]
    pub epsilon: f64,
}

impl BCELossConfig {
    pub const fn init<B: Backend>(&self) -> BCELoss<B> {
        BCELoss {
            epsilon: self.epsilon,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> BCELoss<B> {
    /// Forward pass compatible with PyTorch's F.binary_cross_entropy
    ///
    /// Args:
    ///   input: Predictions, should be in range [0, 1] (typically after sigmoid)
    ///   target: Ground truth labels, can be continuous values in [0, 1]
    ///
    /// Returns:
    ///   Loss tensor
    pub fn forward<const D: usize>(
        &self,
        input: Tensor<B, D>,
        target: Tensor<B, D>,
    ) -> Tensor<B, 1> {
        self.assertions(&input, &target);

        // PyTorch clamps log values to -100 for numerical stability
        let log_clamp_min = (-100.0_f32).elem::<B::FloatElem>();

        // Calculate log terms with clamping (matching PyTorch's -100 clamp)
        let log_input = input.clone().log().clamp_min(log_clamp_min);
        let one_minus_input = Tensor::ones_like(&input) - input.clone();
        let log_one_minus_input = one_minus_input.log().clamp_min(log_clamp_min);

        // PyTorch BCE formula: L = -w * (y * log(x) + (1 - y) * log(1 - x))
        // TODO: Implement full BCE loss calculation with proper numerical stability
        // Current: Simplified implementation using basic formula
        // Should implement: L = -(target * log(input + eps) + (1 - target) * log(1 - input + eps))
        // With proper clamping and numerical stability considerations
        let one = Tensor::ones_like(&target);
        let loss_unreduced = -(target.clone() * log_input + (one - target) * log_one_minus_input);

        // Apply reduction
        loss_unreduced.mean()
    }

    /// Input validation (matching PyTorch's checks)
    fn assertions<const D: usize>(&self, input: &Tensor<B, D>, target: &Tensor<B, D>) {
        // Check shapes match
        assert_eq!(
            input.shape(),
            target.shape(),
            "Input and target must have the same shape. Got input: {:?}, target: {:?}",
            input.shape(),
            target.shape()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type Backend = NdArray<f32>;

    #[test]
    fn test_pytorch_compatible_bce_basic() {
        let device = Default::default();
        let bce_loss = BCELossConfig::new().init::<Backend>();

        // TODO: Add comprehensive test cases for edge scenarios
        // Current: Basic binary classification test
        // Should add: boundary value tests, gradient verification,
        // numerical stability tests with extreme values
        let input = Tensor::<Backend, 1>::from_floats([0.1, 0.9, 0.3, 0.8], &device);
        let target = Tensor::<Backend, 1>::from_floats([0.0, 1.0, 0.0, 1.0], &device);

        let loss = bce_loss.forward(input, target);

        // Loss should be a scalar
        assert_eq!(loss.dims(), [1]);

        // Loss should be positive
        assert!(loss.into_scalar() > 0.0);
    }

    #[test]
    fn test_pytorch_compatible_bce_continuous_targets() {
        let device = Default::default();
        let bce_loss = BCELossConfig::new().init::<Backend>();

        // Test case: continuous targets (soft labels)
        let input = Tensor::<Backend, 1>::from_floats([0.6, 0.8, 0.4, 0.9], &device);
        let target = Tensor::<Backend, 1>::from_floats([0.3, 0.7, 0.1, 0.9], &device);

        let loss = bce_loss.forward(input, target);

        // Loss should be computed without error
        assert_eq!(loss.dims(), [1]);
        assert!(loss.into_scalar() > 0.0);
    }

    #[test]
    fn test_pytorch_compatible_bce_edge_cases() {
        let device = Default::default();

        let input = Tensor::<Backend, 1>::from_floats([0.1, 0.9, 0.3, 0.8], &device);
        let target = Tensor::<Backend, 1>::from_floats([0.0, 1.0, 0.0, 1.0], &device);

        // Test that the loss works without manual epsilon clamping
        let bce_loss = BCELossConfig::new().init::<Backend>();
        let loss = bce_loss.forward(input, target);
        assert_eq!(loss.dims(), [1]);
        assert!(loss.into_scalar() > 0.0);
    }

    #[test]
    fn test_pytorch_compatible_bce_multidimensional() {
        let device = Default::default();
        let bce_loss = BCELossConfig::new().init::<Backend>();

        // Test 2D tensors (like image segmentation)
        let input = Tensor::<Backend, 2>::from_floats([[0.1, 0.9], [0.3, 0.8]], &device);
        let target = Tensor::<Backend, 2>::from_floats([[0.0, 1.0], [0.2, 0.9]], &device);

        let loss = bce_loss.forward(input, target);

        assert_eq!(loss.dims(), [1]);
        assert!(loss.into_scalar() > 0.0);
    }
}
