use burn::{
    nn::{BatchNorm, BatchNormConfig, Gelu, LayerNorm, LayerNormConfig, Relu},
    optim::{Adam, AdamW},
    prelude::*,
    tensor::activation::silu,
};

use super::{
    aspp::{ASPPDeformable, ASPP},
    decoder_blocks::{BasicDecBlk, ResBlk},
};
use crate::special::Identity;

pub enum OptimizerEnum<B: Backend> {
    Adam(Adam<B>),
    AdamW(AdamW<B>),
}

#[derive(Module, Debug)]
pub enum DecAttEnum<B: Backend> {
    None(Identity),
    ASPP(ASPP<B>),
    ASPPDeformable(ASPPDeformable<B>),
}

#[derive(Module, Debug)]
pub enum DecBlkEnum<B: Backend> {
    BasicDecBlk(BasicDecBlk<B>),
    ResBlk(ResBlk<B>),
}

#[derive(Module, Debug)]
pub enum NormLayerEnum<B: Backend> {
    ChannelsFirst(ChannelsFirst),
    ChannelsLast(ChannelsLast),
    BatchNorm2d(BatchNorm<B, 2>),
    BasicDecBlk(LayerNorm<B>),
}

#[derive(Module, Debug, Clone)]
pub struct ChannelsFirst;

impl ChannelsFirst {
    pub fn new() -> Self {
        Self {}
    }
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        input.permute([0, 3, 1, 2])
    }
}

#[derive(Module, Debug, Clone)]
pub struct ChannelsLast;

impl ChannelsLast {
    pub fn new() -> Self {
        Self {}
    }
    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        input.permute([0, 2, 3, 1])
    }
}

pub fn build_norm_layer<B: Backend>(
    dim: usize,
    norm_layer: &str,
    in_format: bool,
    out_format: bool,
    eps: f64,
    device: &Device<B>,
) -> Vec<NormLayerEnum<B>> {
    match norm_layer {
        "BN" => {
            let mut layer = Vec::with_capacity(3);
            if in_format {
                layer.push(NormLayerEnum::ChannelsFirst(ChannelsFirst::new()));
            }
            layer.push(NormLayerEnum::BatchNorm2d(
                BatchNormConfig::new(dim).init(device),
            ));
            if out_format {
                layer.push(NormLayerEnum::ChannelsLast(ChannelsLast::new()));
            }
            layer
        }
        "LN" => {
            let mut layer = Vec::with_capacity(3);
            if in_format {
                layer.push(NormLayerEnum::ChannelsFirst(ChannelsFirst::new()));
            }
            layer.push(NormLayerEnum::BasicDecBlk(
                LayerNormConfig::new(dim).with_epsilon(eps).init(device),
            ));
            if out_format {
                layer.push(NormLayerEnum::ChannelsLast(ChannelsLast::new()));
            }
            layer
        }
        _ => unimplemented!(),
    }
}

pub enum ActLayerEnum {
    ReLU(Relu),
    SiLU(Silu),
    GELU(Gelu),
}

#[derive(Module, Debug, Clone)]
pub struct Silu;

impl Silu {
    pub fn new() -> Self {
        Self {}
    }
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        silu(input)
    }
}

pub fn build_act_layer(act_layer: &str) -> ActLayerEnum {
    match act_layer {
        "ReLU" => ActLayerEnum::ReLU(Relu::new()),
        "SiLU" => ActLayerEnum::SiLU(Silu::new()),
        "GELU" => ActLayerEnum::GELU(Gelu::new()),
        _ => unimplemented!(),
    }
}