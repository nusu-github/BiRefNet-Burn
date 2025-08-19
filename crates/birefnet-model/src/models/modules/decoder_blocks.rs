//! Decoder blocks for BiRefNet architecture.
//!
//! This module provides the core building blocks for BiRefNet's decoder, including
//! BasicDecBlk and ResBlk modules that handle feature processing and upsampling.

use birefnet_extra_ops::Identity;
use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, PaddingConfig2d, Relu,
    },
    prelude::*,
};

use super::{ASPPConfig, ASPPDeformable, ASPPDeformableConfig, ASPP};
use crate::{
    config::{DecChannelsInter, DecoderAttention, InterpolationStrategy, SqueezeBlock},
    error::BiRefNetResult,
};

/// Conditional normalization layer that switches between BatchNorm and Identity.
///
/// Uses BatchNorm when batch_size > 1, otherwise Identity (matching PyTorch behavior).
#[derive(Module, Debug)]
pub enum NormLayer<B: Backend> {
    BatchNorm(BatchNorm<B, 2>),
    Identity(Identity<B>),
}

impl<B: Backend> NormLayer<B> {
    /// Creates a conditional normalization layer.
    ///
    /// # Arguments
    /// * `channels` - Number of channels for BatchNorm
    /// * `batch_size` - Batch size to determine normalization type
    /// * `device` - Device for layer initialization
    pub fn new(channels: usize, batch_size: usize, device: &Device<B>) -> Self {
        if batch_size > 1 {
            Self::BatchNorm(BatchNormConfig::new(channels).init(device))
        } else {
            Self::Identity(Identity::<B>::new())
        }
    }

    /// Forward pass through the conditional normalization layer.
    pub(crate) fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Self::BatchNorm(bn) => bn.forward(x),
            Self::Identity(identity) => identity.forward(x),
        }
    }
}

/// An enum to wrap different types of squeeze blocks used within a `BasicDecBlk`.
#[derive(Module, Debug)]
enum BasicDecSqueezeBlockModule<B: Backend> {
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
    /// Interpolation strategy for tensor resizing operations.
    interpolation: InterpolationStrategy,
}

impl BasicDecBlkConfig {
    /// Creates a new `BasicDecBlk` module following Burn conventions.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> BiRefNetResult<BasicDecBlk<B>> {
        let conv_in = Conv2dConfig::new([self.in_channels, self.inter_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);
        let relu_in = Relu::new();
        let dec_att = match self.dec_att {
            SqueezeBlock::ASPP(_) => Some(BasicDecSqueezeBlockModule::ASPP(
                ASPPConfig::new(self.interpolation.clone())
                    .with_in_channels(self.inter_channels)
                    .init(device),
            )),
            SqueezeBlock::ASPPDeformable(_) => Some(BasicDecSqueezeBlockModule::ASPPDeformable(
                ASPPDeformableConfig::new(self.interpolation.clone())
                    .with_in_channels(self.inter_channels)
                    .init(device)?,
            )),
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
            bn_in,
            relu_in,
            dec_att,
            conv_out,
            bn_out,
        })
    }
}

/// Basic decoder block matching PyTorch BasicDecBlk structure.
///
/// # Shapes
///   - input: `[batch_size, in_channels, height, width]`
///   - output: `[batch_size, out_channels, height, width]`
#[derive(Module, Debug)]
pub struct BasicDecBlk<B: Backend> {
    conv_in: Conv2d<B>,
    bn_in: NormLayer<B>,
    relu_in: Relu,
    dec_att: Option<BasicDecSqueezeBlockModule<B>>,
    conv_out: Conv2d<B>,
    bn_out: NormLayer<B>,
}

impl<B: Backend> BasicDecBlk<B> {
    /// Forward pass through the basic decoder block.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape `[batch_size, in_channels, height, width]`
    ///
    /// # Returns
    /// Output tensor of shape `[batch_size, out_channels, height, width]`
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // First convolution block
        let x = self.conv_in.forward(x);
        let x = self.bn_in.forward(x);
        let x = self.relu_in.forward(x);

        // Optional attention/squeeze block
        let x = match &self.dec_att {
            Some(dec_att) => match dec_att {
                BasicDecSqueezeBlockModule::ASPP(aspp) => aspp.forward(x),
                BasicDecSqueezeBlockModule::ASPPDeformable(aspp_def) => aspp_def.forward(x),
            },
            None => x,
        };

        // Output convolution block
        let x = self.conv_out.forward(x);
        self.bn_out.forward(x)
    }
}

/// An enum to wrap different types of squeeze blocks used within a `ResBlk`.
#[derive(Module, Debug)]
enum ResBlkSqueezeBlockModule<B: Backend> {
    Aspp(ASPP<B>),
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
    #[config(default = "DecoderAttention::ASPPDeformable")]
    dec_att: DecoderAttention,
    /// Strategy for intermediate channel sizes.
    #[config(default = "DecChannelsInter::Fixed")]
    dec_channels_inter: DecChannelsInter,
    /// Batch size for conditional BatchNorm usage.
    #[config(default = "4")]
    batch_size: usize,
    /// Interpolation strategy for tensor resizing operations.
    interpolation: InterpolationStrategy,
}

impl ResBlkConfig {
    /// Creates a new `ResBlk` module following Burn conventions.
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
            DecoderAttention::ASPP => Some(ResBlkSqueezeBlockModule::Aspp(
                ASPPConfig::new(self.interpolation.clone())
                    .with_in_channels(inter_channels)
                    .init(device),
            )),
            DecoderAttention::ASPPDeformable => Some(ResBlkSqueezeBlockModule::ASPPDeformable(
                ASPPDeformableConfig::new(self.interpolation.clone())
                    .with_in_channels(inter_channels)
                    .init(device)?,
            )),
            DecoderAttention::None => None,
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

/// Residual decoder block matching PyTorch ResBlk structure.
///
/// # Shapes
///   - input: `[batch_size, in_channels, height, width]`
///   - output: `[batch_size, out_channels, height, width]`
#[derive(Module, Debug)]
pub struct ResBlk<B: Backend> {
    conv_resi: Conv2d<B>,
    conv_in: Conv2d<B>,
    bn_in: NormLayer<B>,
    relu_in: Relu,
    dec_att: Option<ResBlkSqueezeBlockModule<B>>,
    conv_out: Conv2d<B>,
    bn_out: NormLayer<B>,
}

impl<B: Backend> ResBlk<B> {
    /// Forward pass through the residual decoder block.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape `[batch_size, in_channels, height, width]`
    ///
    /// # Returns
    /// Output tensor of shape `[batch_size, out_channels, height, width]`
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Residual connection (skip connection)
        let residual = self.conv_resi.forward(x.clone());

        // Main path
        let x = self.conv_in.forward(x);
        let x = self.bn_in.forward(x);
        let x = self.relu_in.forward(x);

        // Optional attention/squeeze block
        let x = match &self.dec_att {
            Some(dec_att) => match dec_att {
                ResBlkSqueezeBlockModule::Aspp(aspp) => aspp.forward(x),
                ResBlkSqueezeBlockModule::ASPPDeformable(aspp_def) => aspp_def.forward(x),
            },
            None => x,
        };

        // Final convolution and batch normalization
        let x = self.conv_out.forward(x);
        let x = self.bn_out.forward(x);

        // Add residual connection
        x + residual
    }
}
