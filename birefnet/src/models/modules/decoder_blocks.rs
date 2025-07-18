//! # Decoder Blocks
//!
//! This module defines the core building blocks for the decoder part of BiRefNet.
//! These blocks are responsible for processing and upsampling feature maps.

use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, PaddingConfig2d, Relu,
    },
    prelude::*,
};

use super::{ASPPConfig, ASPPDeformable, ASPPDeformableConfig, ASPP};
use crate::config::{DecAtt, DecChannelsInter, SqueezeBlock};
use crate::error::BiRefNetResult;
use burn_extra_ops::Identity;

/// An enum to wrap different normalization layers (BatchNorm or Identity).
#[derive(Module, Debug)]
enum NormLayer<B: Backend> {
    BatchNorm(BatchNorm<B, 2>),
    Identity(Identity<B>),
}

impl<B: Backend> NormLayer<B> {
    /// Create a new normalization layer based on batch_size.
    /// Uses BatchNorm if batch_size > 1, otherwise Identity (like PyTorch).
    fn new(channels: usize, batch_size: usize, device: &Device<B>) -> Self {
        if batch_size > 1 {
            Self::BatchNorm(BatchNormConfig::new(channels).init(device))
        } else {
            Self::Identity(Identity::<B>::new())
        }
    }

    /// Forward pass through the normalization layer.
    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Self::BatchNorm(bn) => bn.forward(x),
            Self::Identity(identity) => identity.forward(x),
        }
    }
}

/// An enum to wrap different types of squeeze blocks used within a `BasicDecBlk`.
#[derive(Module, Debug)]
enum BasicDecSqueezeBlockModuleEnum<B: Backend> {
    ASPP(ASPP<B>),
    ASPPDeformable(ASPPDeformable<B>),
}

/// Configuration for the `BasicDecBlk` module.
#[derive(Config, Debug)]
pub struct BasicDecBlkConfig {
    /// Number of input channels.
    #[config(default = "64")]
    in_channels: usize,
    /// Number of output channels.
    #[config(default = "64")]
    out_channels: usize,
    /// Number of intermediate channels.
    #[config(default = "64")]
    inter_channels: usize,
    /// The attention/squeeze block to use within the decoder block.
    dec_att: SqueezeBlock,
    /// Batch size for conditional BatchNorm usage.
    #[config(default = "4")]
    batch_size: usize,
}

impl BasicDecBlkConfig {
    /// Initializes a new `BasicDecBlk` module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> BiRefNetResult<BasicDecBlk<B>> {
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
                        .init(device)?,
                ))
            }
            _ => None,
        };
        let conv_out = Conv2dConfig::new([self.inter_channels, self.out_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let bn_in = NormLayer::new(self.inter_channels, self.batch_size, device);
        let bn_out = NormLayer::new(self.out_channels, self.batch_size, device);

        Ok(BasicDecBlk {
            conv_in,
            relu_in,
            dec_att,
            conv_out,
            bn_in,
            bn_out,
        })
    }
}

/// A basic decoder block.
#[derive(Module, Debug)]
pub struct BasicDecBlk<B: Backend> {
    conv_in: Conv2d<B>,
    bn_in: NormLayer<B>,
    relu_in: Relu,
    dec_att: Option<BasicDecSqueezeBlockModuleEnum<B>>,
    conv_out: Conv2d<B>,
    bn_out: NormLayer<B>,
}

impl<B: Backend> BasicDecBlk<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv_in.forward(x);
        let x = self.bn_in.forward(x);
        let x = self.relu_in.forward(x);
        let x = match &self.dec_att {
            Some(dec_att) => match dec_att {
                BasicDecSqueezeBlockModuleEnum::ASPP(dec_att) => dec_att.forward(x),
                BasicDecSqueezeBlockModuleEnum::ASPPDeformable(dec_att) => dec_att.forward(x),
            },
            None => x,
        };
        let x = self.conv_out.forward(x);

        self.bn_out.forward(x)
    }
}

/// An enum to wrap different types of squeeze blocks used within a `ResBlk`.
#[derive(Module, Debug)]
enum ResBlkSqueezeBlockModuleEnum<B: Backend> {
    ASPP(ASPP<B>),
    ASPPDeformable(ASPPDeformable<B>),
}

/// Configuration for the `ResBlk` module.
#[derive(Config, Debug)]
pub struct ResBlkConfig {
    /// Number of input channels.
    #[config(default = "64")]
    in_channels: usize,
    /// Number of output channels. Defaults to `in_channels`.
    #[config(default = "None")]
    out_channels: Option<usize>,
    /// Number of intermediate channels.
    #[config(default = "64")]
    inter_channels: usize,
    /// The attention/squeeze block to use within the decoder block.
    #[config(default = "DecAtt::ASPPDeformable")]
    dec_att: DecAtt,
    /// Strategy for intermediate channel sizes.
    #[config(default = "DecChannelsInter::Fixed")]
    dec_channels_inter: DecChannelsInter,
    /// Batch size for conditional BatchNorm usage.
    #[config(default = "4")]
    batch_size: usize,
}

impl ResBlkConfig {
    /// Initializes a new `ResBlk` module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> BiRefNetResult<ResBlk<B>> {
        let out_channels = self.out_channels.unwrap_or(self.in_channels);

        // Calculate inter_channels based on dec_channels_inter strategy
        let inter_channels = match self.dec_channels_inter {
            DecChannelsInter::Adap => self.in_channels / 4,
            DecChannelsInter::Fixed => 64,
        };

        let conv_in = Conv2dConfig::new([self.in_channels, inter_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let bn_in = NormLayer::new(inter_channels, self.batch_size, device);
        let relu_in = Relu::new();

        // Initialize attention module based on dec_att
        let dec_att = match self.dec_att {
            DecAtt::ASPP => Some(ResBlkSqueezeBlockModuleEnum::ASPP(
                ASPPConfig::new()
                    .with_in_channels(inter_channels)
                    .init(device),
            )),
            DecAtt::ASPPDeformable => Some(ResBlkSqueezeBlockModuleEnum::ASPPDeformable(
                ASPPDeformableConfig::new()
                    .with_in_channels(inter_channels)
                    .init(device)?,
            )),
            DecAtt::None => None,
        };

        let conv_out = Conv2dConfig::new([inter_channels, out_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let bn_out = NormLayer::new(out_channels, self.batch_size, device);

        let conv_resi = Conv2dConfig::new([self.in_channels, out_channels], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .init(device);

        Ok(ResBlk {
            conv_resi,
            conv_in,
            bn_in,
            relu_in,
            dec_att,
            conv_out,
            bn_out,
        })
    }
}

/// A residual decoder block.
#[derive(Module, Debug)]
pub struct ResBlk<B: Backend> {
    conv_resi: Conv2d<B>,
    conv_in: Conv2d<B>,
    bn_in: NormLayer<B>,
    relu_in: Relu,
    dec_att: Option<ResBlkSqueezeBlockModuleEnum<B>>,
    conv_out: Conv2d<B>,
    bn_out: NormLayer<B>,
}

impl<B: Backend> ResBlk<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let _x = self.conv_resi.forward(x.clone());
        let x = self.conv_in.forward(x);
        let x = self.bn_in.forward(x);
        let x = self.relu_in.forward(x);
        let x = match &self.dec_att {
            Some(dec_att) => match dec_att {
                ResBlkSqueezeBlockModuleEnum::ASPP(dec_att) => dec_att.forward(x),
                ResBlkSqueezeBlockModuleEnum::ASPPDeformable(dec_att) => dec_att.forward(x),
            },
            None => x,
        };
        let x = self.conv_out.forward(x);
        let x = self.bn_out.forward(x);
        x + _x
    }
}
