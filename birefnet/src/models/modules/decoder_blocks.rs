use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, PaddingConfig2d, Relu,
    },
    prelude::*,
};

use super::{ASPPConfig, ASPPDeformable, ASPPDeformableConfig, ASPP};
use crate::config::SqueezeBlock;

#[derive(Module, Debug)]
enum BasicDecSqueezeBlockModuleEnum<B: Backend> {
    ASPP(ASPP<B>),
    ASPPDeformable(ASPPDeformable<B>),
}

#[derive(Config, Debug)]
pub struct BasicDecBlkConfig {
    #[config(default = "64")]
    in_channels: usize,
    #[config(default = "64")]
    out_channels: usize,
    #[config(default = "64")]
    inter_channels: usize,
    dec_att: SqueezeBlock,
}

impl BasicDecBlkConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> BasicDecBlk<B> {
        let conv_in = Conv2dConfig::new([self.in_channels, self.inter_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let relu_in = Relu::new();
        let dec_att = match self.dec_att {
            SqueezeBlock::ASPP(_) => Some(BasicDecSqueezeBlockModuleEnum::ASPP(
                ASPPConfig::new()
                    .with_in_channels(self.inter_channels)
                    .init(device),
            )),
            SqueezeBlock::ASPPDeformable(_) => {
                Some(BasicDecSqueezeBlockModuleEnum::ASPPDeformable(
                    ASPPDeformableConfig::new()
                        .with_in_channels(self.inter_channels)
                        .init(device),
                ))
            }
            _ => None,
        };
        let conv_out = Conv2dConfig::new([self.inter_channels, self.out_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn_in = BatchNormConfig::new(self.inter_channels).init(device);
        let bn_out = BatchNormConfig::new(self.out_channels).init(device);

        BasicDecBlk {
            conv_in,
            relu_in,
            dec_att,
            conv_out,
            bn_in: Some(bn_in),
            bn_out: Some(bn_out),
        }
    }
}

#[derive(Module, Debug)]
pub struct BasicDecBlk<B: Backend> {
    conv_in: Conv2d<B>,
    bn_in: Option<BatchNorm<B, 2>>,
    relu_in: Relu,
    dec_att: Option<BasicDecSqueezeBlockModuleEnum<B>>,
    conv_out: Conv2d<B>,
    bn_out: Option<BatchNorm<B, 2>>,
}

impl<B: Backend> BasicDecBlk<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv_in.forward(x);
        let x = if let Some(bn_in) = &self.bn_in {
            bn_in.forward(x)
        } else {
            x
        };
        let x = self.relu_in.forward(x);
        let x = if let Some(dec_att) = &self.dec_att {
            match dec_att {
                BasicDecSqueezeBlockModuleEnum::ASPP(dec_att) => dec_att.forward(x),
                BasicDecSqueezeBlockModuleEnum::ASPPDeformable(dec_att) => dec_att.forward(x),
            }
        } else {
            x
        };
        let x = self.conv_out.forward(x);

        if let Some(bn_out) = &self.bn_out {
            bn_out.forward(x)
        } else {
            x
        }
    }
}

#[derive(Config, Debug)]
pub struct ResBlkConfig {
    #[config(default = "64")]
    in_channels: usize,
    #[config(default = "None")]
    out_channels: Option<usize>,
    #[config(default = "64")]
    inter_channels: usize,
}

impl ResBlkConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> ResBlk<B> {
        todo!()
    }
}

#[derive(Module, Debug)]
pub struct ResBlk<B: Backend> {
    conv_resi: Conv2d<B>,
    conv_in: Conv2d<B>,
    bn_in: Option<BatchNorm<B, 2>>,
    relu_in: Relu,
    dec_att: Option<ASPPDeformable<B>>,
    conv_out: Conv2d<B>,
    bn_out: Option<BatchNorm<B, 2>>,
}

impl<B: Backend> ResBlk<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let _x = self.conv_resi.forward(x.clone());
        let x = self.conv_in.forward(x);
        let x = if let Some(bn_in) = &self.bn_in {
            bn_in.forward(x)
        } else {
            x
        };
        let x = self.relu_in.forward(x);
        let x = if let Some(dec_att) = &self.dec_att {
            dec_att.forward(x)
        } else {
            x
        };
        let x = self.conv_out.forward(x);
        let x = if let Some(bn_out) = &self.bn_out {
            bn_out.forward(x)
        } else {
            x
        };
        x + _x
    }
}
