use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, Distribution, Float, Tensor},
};

#[derive(Config, Debug)]
pub struct DropPathConfig {
    #[config(default = "0.0")]
    drop_prob: f64,
    #[config(default = "false")]
    training: bool,
    #[config(default = "true")]
    scale_by_keep: bool,
}

impl DropPathConfig {
    pub fn init(&self) -> DropPath {
        DropPath {
            drop_prob: self.drop_prob,
            training: self.training,
            scale_by_keep: self.scale_by_keep,
        }
    }
}

#[derive(Module, Clone, Debug, Default)]
pub struct DropPath {
    drop_prob: f64,
    training: bool,
    scale_by_keep: bool,
}

impl DropPath {
    pub fn forward<B: Backend, const D: usize>(
        &self,
        x: Tensor<B, D, Float>,
    ) -> Tensor<B, D, Float> {
        if !self.training || self.drop_prob == 0.0 {
            return x;
        }
        let keep_prob = 1.0 - self.drop_prob;
        let other_dims = vec![1; D - 1];
        let shape: Vec<_> = std::iter::once(x.dims()[0]).chain(other_dims).collect();
        let random_tensor = Tensor::random(shape, Distribution::Bernoulli(keep_prob), &x.device());
        if self.scale_by_keep {
            x * random_tensor / keep_prob
        } else {
            x * random_tensor
        }
    }
}
