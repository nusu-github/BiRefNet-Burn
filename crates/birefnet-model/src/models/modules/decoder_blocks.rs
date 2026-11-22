//! Core decoder and residual building blocks for `BiRefNet`.

use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Identity, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
};

use super::{ASPP, ASPPConfig, ASPPDeformable, ASPPDeformableConfig};
use crate::{
    config::{DecChannelsInter, DecoderAttention, InterpolationStrategy},
    error::ModelError,
};

/// Conditional normalization layer switching between `BatchNorm` (`batch_size > 1`)
/// and identity pass-through (`batch_size <= 1`).
#[derive(Module, Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "Burn Module requires inline storage without Box indirection"
)]
pub enum NormLayer {
    /// Batch normalization over channel dimension.
    BatchNorm(BatchNorm),
    /// Identity normalization when batch size is 1.
    Identity(Identity),
}

impl NormLayer {
    /// Creates a conditional normalization layer matching `batch_size`.
    #[must_use]
    pub fn new(channels: usize, batch_size: usize, device: &Device) -> Self {
        if batch_size > 1 {
            Self::BatchNorm(BatchNormConfig::new(channels).init(device))
        } else {
            Self::Identity(Identity::new())
        }
    }

    pub(crate) fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        match self {
            Self::BatchNorm(bn) => bn.forward(x),
            Self::Identity(identity) => identity.forward(x),
        }
    }
}

#[derive(Module, Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "Burn Module requires inline storage without Box indirection"
)]
#[expect(
    clippy::upper_case_acronyms,
    reason = "preserves ASPP variant name matching existing BiRefNet schema"
)]
enum BasicDecSqueezeBlockModule {
    ASPP(ASPP),
    ASPPDeformable(ASPPDeformable),
}

/// Configuration for [`BasicDecBlk`].
#[derive(Config, Debug)]
pub struct BasicDecBlkConfig {
    /// Number of input channels.
    #[config(default = 64)]
    in_channels: usize,
    /// Number of output channels.
    #[config(default = 64)]
    out_channels: usize,
    /// Fixed intermediate channel width used when `dec_channels_inter` is `Fixed`.
    #[config(default = 64)]
    inter_channels: usize,
    /// Intermediate channel sizing strategy.
    #[config(default = "DecChannelsInter::Fixed")]
    dec_channels_inter: DecChannelsInter,
    /// Attention block applied between the input and output convolutions.
    dec_att: DecoderAttention,
    /// Batch size controlling conditional `BatchNorm`.
    #[config(default = 4)]
    batch_size: usize,
    /// Interpolation strategy passed to nested ASPP branches.
    interpolation: InterpolationStrategy,
}

impl BasicDecBlkConfig {
    /// Initializes a [`BasicDecBlk`] module.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if any nested ASPP configuration is invalid.
    pub fn init(&self, device: &Device) -> Result<BasicDecBlk, ModelError> {
        let inter_channels = match self.dec_channels_inter {
            DecChannelsInter::Adap => (self.in_channels / 4).max(1),
            DecChannelsInter::Fixed => self.inter_channels,
        };

        let conv_in = Conv2dConfig::new([self.in_channels, inter_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);
        let relu_in = Relu::new();
        let dec_att = match self.dec_att {
            DecoderAttention::ASPP => Some(BasicDecSqueezeBlockModule::ASPP(
                ASPPConfig::new(self.interpolation)
                    .with_in_channels(inter_channels)
                    .with_batch_size(self.batch_size)
                    .init(device)?,
            )),
            DecoderAttention::ASPPDeformable => Some(BasicDecSqueezeBlockModule::ASPPDeformable(
                ASPPDeformableConfig::new(self.interpolation)
                    .with_in_channels(inter_channels)
                    .with_batch_size(self.batch_size)
                    .init(device),
            )),
            DecoderAttention::None => None,
        };
        let conv_out = Conv2dConfig::new([inter_channels, self.out_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);
        let bn_in = NormLayer::new(inter_channels, self.batch_size, device);
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

/// Basic decoder block matching PyTorch's `BasicDecBlk` structure.
///
/// # Shapes
/// - input: `[batch_size, in_channels, height, width]`
/// - output: `[batch_size, out_channels, height, width]`
#[derive(Module, Debug)]
pub struct BasicDecBlk {
    conv_in: Conv2d,
    bn_in: NormLayer,
    relu_in: Relu,
    dec_att: Option<BasicDecSqueezeBlockModule>,
    conv_out: Conv2d,
    bn_out: NormLayer,
}

impl BasicDecBlk {
    /// Applies convolution, optional squeeze attention, and output projection.
    #[must_use]
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.conv_in.forward(x);
        let x = self.bn_in.forward(x);
        let x = self.relu_in.forward(x);

        let x = match &self.dec_att {
            Some(BasicDecSqueezeBlockModule::ASPP(aspp)) => aspp.forward(x),
            Some(BasicDecSqueezeBlockModule::ASPPDeformable(aspp_def)) => aspp_def.forward(x),
            None => x,
        };

        let x = self.conv_out.forward(x);
        self.bn_out.forward(x)
    }
}

#[derive(Module, Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "Burn Module requires inline storage without Box indirection"
)]
enum ResBlkSqueezeBlockModule {
    Aspp(ASPP),
    ASPPDeformable(ASPPDeformable),
}

/// Configuration for [`ResBlk`].
#[derive(Config, Debug)]
pub struct ResBlkConfig {
    /// Number of input channels.
    #[config(default = 64)]
    in_channels: usize,
    /// Number of output channels; defaults to `in_channels`.
    #[config(default = "None")]
    out_channels: Option<usize>,
    /// Fixed intermediate channel width used when `dec_channels_inter` is `Fixed`.
    #[config(default = 64)]
    inter_channels: usize,
    /// Attention/squeeze block inside the residual decoder block.
    #[config(default = "DecoderAttention::ASPPDeformable")]
    dec_att: DecoderAttention,
    /// Intermediate channel sizing strategy.
    #[config(default = "DecChannelsInter::Fixed")]
    dec_channels_inter: DecChannelsInter,
    /// Batch size controlling conditional `BatchNorm`.
    #[config(default = 4)]
    batch_size: usize,
    /// Interpolation strategy passed to nested ASPP branches.
    interpolation: InterpolationStrategy,
}

impl ResBlkConfig {
    /// Initializes a [`ResBlk`] module.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if nested ASPP initialization fails.
    pub fn init(&self, device: &Device) -> Result<ResBlk, ModelError> {
        let out_channels = self.out_channels.unwrap_or(self.in_channels);
        let inter_channels = match self.dec_channels_inter {
            DecChannelsInter::Adap => (self.in_channels / 4).max(1),
            DecChannelsInter::Fixed => self.inter_channels,
        };

        let conv_in = Conv2dConfig::new([self.in_channels, inter_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);
        let bn_in = NormLayer::new(inter_channels, self.batch_size, device);
        let relu_in = Relu::new();

        let dec_att = match self.dec_att {
            DecoderAttention::ASPP => Some(ResBlkSqueezeBlockModule::Aspp(
                ASPPConfig::new(self.interpolation)
                    .with_in_channels(inter_channels)
                    .with_batch_size(self.batch_size)
                    .init(device)?,
            )),
            DecoderAttention::ASPPDeformable => Some(ResBlkSqueezeBlockModule::ASPPDeformable(
                ASPPDeformableConfig::new(self.interpolation)
                    .with_in_channels(inter_channels)
                    .with_batch_size(self.batch_size)
                    .init(device),
            )),
            DecoderAttention::None => None,
        };

        let conv_out = Conv2dConfig::new([inter_channels, out_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);
        let bn_out = NormLayer::new(out_channels, self.batch_size, device);

        let conv_resi = Conv2dConfig::new([self.in_channels, out_channels], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Valid)
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

/// Residual decoder block matching PyTorch's `ResBlk` structure.
///
/// # Shapes
/// - input: `[batch_size, in_channels, height, width]`
/// - output: `[batch_size, out_channels, height, width]`
#[derive(Module, Debug)]
pub struct ResBlk {
    conv_resi: Conv2d,
    conv_in: Conv2d,
    bn_in: NormLayer,
    relu_in: Relu,
    dec_att: Option<ResBlkSqueezeBlockModule>,
    conv_out: Conv2d,
    bn_out: NormLayer,
}

impl ResBlk {
    /// Applies the residual decoder block forward pass.
    #[must_use]
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let residual = self.conv_resi.forward(x.clone());

        let x = self.conv_in.forward(x);
        let x = self.bn_in.forward(x);
        let x = self.relu_in.forward(x);

        let x = match &self.dec_att {
            Some(ResBlkSqueezeBlockModule::Aspp(aspp)) => aspp.forward(x),
            Some(ResBlkSqueezeBlockModule::ASPPDeformable(aspp_def)) => aspp_def.forward(x),
            None => x,
        };

        let x = self.conv_out.forward(x);
        let x = self.bn_out.forward(x);

        x + residual
    }
}
