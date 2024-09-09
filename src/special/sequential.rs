use burn::{nn::*, prelude::*};

use super::Identity;

#[derive(Module, Debug)]
pub enum SequentialType<B: Backend> {
    Conv2d(conv::Conv2d<B>),
    BatchNorm2d(BatchNorm<B, 2>),
    ReLU(Relu),
    Identity(Identity<B>),
}

#[derive(Module, Debug)]
pub struct Sequential<B: Backend> {
    layers: Vec<SequentialType<B>>,
}

impl<B: Backend> Sequential<B> {
    pub fn new(layers: Vec<SequentialType<B>>) -> Self {
        Self { layers }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = x;
        for layer in &self.layers {
            match layer {
                SequentialType::Conv2d(conv) => x = conv.forward(x),
                SequentialType::BatchNorm2d(bn) => x = bn.forward(x),
                SequentialType::ReLU(relu) => x = relu.forward(x),
                SequentialType::Identity(identity) => x = identity.forward(x),
            }
        }
        x
    }
}
