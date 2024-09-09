use std::marker::PhantomData;

use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub struct Identity<B: Backend> {
    _marker: PhantomData<B>,
}

impl<B: Backend> Identity<B> {
    pub fn new() -> Self {
        Self {
            _marker: PhantomData,
        }
    }
    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        x
    }
}
