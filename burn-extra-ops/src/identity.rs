//! Identity module implementation

use burn::prelude::*;

/// Identity module that returns input unchanged
#[derive(Module, Debug)]
pub struct Identity<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> Identity<B> {
    /// Create new Identity module
    pub const fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Forward pass (identity function)
    pub const fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        input
    }
}

impl<B: Backend> Default for Identity<B> {
    fn default() -> Self {
        Self::new()
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
    fn identity_forward_input_unchanged() {
        let device = Default::default();
        let identity = Identity::<TestBackend>::new();
        let input = Tensor::<TestBackend, 3>::random(
            [2, 3, 4],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = identity.forward(input.clone());

        // Output should be identical to input
        assert_eq!(output.dims(), input.dims());
    }
}
