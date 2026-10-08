//! Identity module implementation

use burn::prelude::*;

/// Identity module that returns input unchanged
#[derive(Module, Debug)]
pub struct Identity {}

impl Identity {
    /// Create new Identity module
    pub const fn new() -> Self {
        Self {}
    }

    /// Forward pass (identity function)
    pub const fn forward<const D: usize>(&self, input: Tensor<D>) -> Tensor<D> {
        input
    }
}

impl Default for Identity {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::{Distribution, Tensor};

    use super::*;

    #[test]
    fn identity_forward_preserves_input_unchanged() {
        let device = burn::tensor::Device::cpu();
        let identity = Identity::new();
        let input = Tensor::<3>::random([2, 3, 4], Distribution::Normal(0.0, 1.0), &device);
        let output = identity.forward(input.clone());

        // Output should be identical to input
        assert_eq!(output.dims(), input.dims());
    }
}
