//! # Identity Module
//!
//! A simple module that returns the input tensor unchanged.
//! It is a `const` and `Default` struct, making it easy to use as a placeholder.

use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};

/// An identity module that passes its input through unchanged.
#[derive(Module, Clone, Debug, Default)]
pub struct Identity {}

impl Identity {
    /// Creates a new `Identity` module.
    pub const fn new() -> Self {
        Self {}
    }

    /// The forward pass, which returns the input tensor `x`.
    pub const fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        x
    }
}
