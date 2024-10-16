use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Clone, Debug, Default)]
pub struct Identity {}

impl Identity {
    pub const fn new() -> Self {
        Self {}
    }
    pub const fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        x
    }
}
