use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Clone, Debug, Default)]
pub struct Identity {}

impl Identity {
    pub fn new() -> Self {
        Self {}
    }
    pub fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        x
    }
}
