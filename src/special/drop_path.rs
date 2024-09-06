use burn::{prelude::*, tensor::Distribution};
use std::marker::PhantomData;

#[derive(Config)]
pub struct DropPathConfig {
    #[config(default = "0.0")]
    drop_prob: f64,
    #[config(default = "false")]
    training: bool,
    #[config(default = "true")]
    scale_by_keep: bool,
}

impl DropPathConfig {
    pub fn init<B: Backend>(&self, _device: &Device<B>) -> DropPath<B> {
        DropPath {
            drop_prob: self.drop_prob,
            training: self.training,
            scale_by_keep: self.scale_by_keep,
            _p: PhantomData,
        }
    }
}

#[derive(Module, Debug)]
pub struct DropPath<B: Backend> {
    drop_prob: f64,
    training: bool,
    scale_by_keep: bool,
    _p: PhantomData<B>,
}

impl<B: Backend> DropPath<B> {
    pub fn forward(&self, x: Tensor<B, 3, Float>) -> Tensor<B, 3> {
        if !self.training || self.drop_prob == 0.0 {
            return x;
        }
        let keep_prob = 1.0 - self.drop_prob;
        let other_dims = vec![1; 3 - 1];
        let shape: Vec<usize> = std::iter::once(x.dims()[0])
            .chain(other_dims.into_iter())
            .collect();
        let random_tensor: Tensor<B, 3, Float> = Tensor::random(
            shape,
            Distribution::Bernoulli(keep_prob),
            &Default::default(),
        );
        if self.scale_by_keep {
            x * random_tensor / keep_prob
        } else {
            x * random_tensor
        }
    }
}
