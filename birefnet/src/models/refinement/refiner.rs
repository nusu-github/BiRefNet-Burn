use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig, PaddingConfig2d, Relu,
    },
    prelude::*,
    tensor::{
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};

use crate::{
    config::SqueezeBlock,
    error::BiRefNetResult,
    models::modules::{
        utils::{build_act_layer, build_norm_layer, ActLayerEnum, NormLayerEnum},
        BasicDecBlk, BasicDecBlkConfig, BasicLatBlk, BasicLatBlkConfig,
    },
    DecAtt, ModelConfig,
};

use backbones::{create_backbone, Backbone, BackboneType, BackboneWrapper, PVTv2Variant};

/// Configuration for the RefinerDecoder
#[derive(Config, Debug)]
pub struct RefinerDecoderConfig {
    /// The detailed model configuration.
    config: ModelConfig,
    /// Channel sizes
    channels: [usize; 4],
    /// Whether to use multi-scale supervision
    #[config(default = false)]
    ms_supervision: bool,
}

/// Decoder used by Refiner modules
#[derive(Module, Debug)]
pub struct RefinerDecoder<B: Backend> {
    decoder_block4: BasicDecBlk<B>,
    decoder_block3: BasicDecBlk<B>,
    decoder_block2: BasicDecBlk<B>,
    decoder_block1: BasicDecBlk<B>,
    lateral_block4: BasicLatBlk<B>,
    lateral_block3: BasicLatBlk<B>,
    lateral_block2: BasicLatBlk<B>,
    conv_out1: Conv2d<B>,
    conv_ms_spvn_4: Option<Conv2d<B>>,
    conv_ms_spvn_3: Option<Conv2d<B>>,
    conv_ms_spvn_2: Option<Conv2d<B>>,
    ms_supervision: bool,
}

impl RefinerDecoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BiRefNetResult<RefinerDecoder<B>> {
        let squeeze_block = match self.config.decoder.dec_att {
            DecAtt::None => SqueezeBlock::None,
            DecAtt::ASPP => SqueezeBlock::ASPP(0),
            DecAtt::ASPPDeformable => SqueezeBlock::ASPPDeformable(0),
        };

        let decoder_block4 = BasicDecBlkConfig::new(squeeze_block.clone())
            .with_in_channels(self.channels[0])
            .with_out_channels(self.channels[1])
            .init(device)?;

        let decoder_block3 = BasicDecBlkConfig::new(squeeze_block.clone())
            .with_in_channels(self.channels[1])
            .with_out_channels(self.channels[2])
            .init(device)?;

        let decoder_block2 = BasicDecBlkConfig::new(squeeze_block.clone())
            .with_in_channels(self.channels[2])
            .with_out_channels(self.channels[3])
            .init(device)?;

        let decoder_block1 = BasicDecBlkConfig::new(squeeze_block)
            .with_in_channels(self.channels[3])
            .with_out_channels(self.channels[3] / 2)
            .init(device)?;

        let lateral_block4 = BasicLatBlkConfig::new()
            .with_in_channels(self.channels[1])
            .with_out_channels(self.channels[1])
            .init(device);

        let lateral_block3 = BasicLatBlkConfig::new()
            .with_in_channels(self.channels[2])
            .with_out_channels(self.channels[2])
            .init(device);

        let lateral_block2 = BasicLatBlkConfig::new()
            .with_in_channels(self.channels[3])
            .with_out_channels(self.channels[3])
            .init(device);

        let conv_out1 = Conv2dConfig::new([self.channels[3] / 2, 1], [1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .init(device);

        let (conv_ms_spvn_4, conv_ms_spvn_3, conv_ms_spvn_2) = if self.ms_supervision {
            (
                Some(
                    Conv2dConfig::new([self.channels[1], 1], [1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .init(device),
                ),
                Some(
                    Conv2dConfig::new([self.channels[2], 1], [1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .init(device),
                ),
                Some(
                    Conv2dConfig::new([self.channels[3], 1], [1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .init(device),
                ),
            )
        } else {
            (None, None, None)
        };

        Ok(RefinerDecoder {
            decoder_block4,
            decoder_block3,
            decoder_block2,
            decoder_block1,
            lateral_block4,
            lateral_block3,
            lateral_block2,
            conv_out1,
            conv_ms_spvn_4,
            conv_ms_spvn_3,
            conv_ms_spvn_2,
            ms_supervision: self.ms_supervision,
        })
    }
}

impl<B: Backend> RefinerDecoder<B> {
    pub fn forward(&self, features: [Tensor<B, 4>; 5]) -> Vec<Tensor<B, 4>> {
        let [x, x1, x2, x3, x4] = features;
        let mut outs = Vec::new();

        // Decoder block 4
        let p4 = self.decoder_block4.forward(x4);
        let [_, _, h3, w3] = x3.dims();
        let _p4 = interpolate(
            p4.clone(),
            [h3, w3],
            InterpolateOptions::new(InterpolateMode::Bilinear),
        );
        let _p3 = _p4 + self.lateral_block4.forward(x3);

        // Decoder block 3
        let p3 = self.decoder_block3.forward(_p3);
        let [_, _, h2, w2] = x2.dims();
        let _p3 = interpolate(
            p3.clone(),
            [h2, w2],
            InterpolateOptions::new(InterpolateMode::Bilinear),
        );
        let _p2 = _p3 + self.lateral_block3.forward(x2);

        // Decoder block 2
        let p2 = self.decoder_block2.forward(_p2);
        let [_, _, h1, w1] = x1.dims();
        let _p2 = interpolate(
            p2.clone(),
            [h1, w1],
            InterpolateOptions::new(InterpolateMode::Bilinear),
        );
        let _p1 = _p2 + self.lateral_block2.forward(x1);

        // Decoder block 1
        let _p1 = self.decoder_block1.forward(_p1);
        let [_, _, h, w] = x.dims();
        let _p1 = interpolate(
            _p1,
            [h, w],
            InterpolateOptions::new(InterpolateMode::Bilinear),
        );
        let p1_out = self.conv_out1.forward(_p1);

        // Multi-scale supervision outputs
        if self.ms_supervision {
            if let Some(conv) = &self.conv_ms_spvn_4 {
                outs.push(conv.forward(p4));
            }
            if let Some(conv) = &self.conv_ms_spvn_3 {
                outs.push(conv.forward(p3));
            }
            if let Some(conv) = &self.conv_ms_spvn_2 {
                outs.push(conv.forward(p2));
            }
        }

        outs.push(p1_out);
        outs
    }
}

/// Stem layer for processing input
#[derive(Module, Debug)]
pub struct StemLayer<B: Backend> {
    conv1: Conv2d<B>,
    norm1: Vec<NormLayerEnum<B>>,
    act: ActLayerEnum,
    conv2: Conv2d<B>,
    norm2: Vec<NormLayerEnum<B>>,
}

impl<B: Backend> StemLayer<B> {
    pub fn new(
        in_channels: usize,
        inter_channels: usize,
        out_channels: usize,
        norm_layer: &str,
        device: &B::Device,
    ) -> BiRefNetResult<Self> {
        let conv1 = Conv2dConfig::new([in_channels, inter_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let norm1 = build_norm_layer(inter_channels, norm_layer, true, true, 1e-5, device)?;
        let act = build_act_layer("GELU")?;

        let conv2 = Conv2dConfig::new([inter_channels, out_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let norm2 = build_norm_layer(out_channels, norm_layer, true, true, 1e-5, device)?;

        Ok(Self {
            conv1,
            norm1,
            act,
            conv2,
            norm2,
        })
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut x = self.conv1.forward(x);

        // Apply norm layers
        for norm in &self.norm1 {
            x = match norm {
                NormLayerEnum::ChannelsFirst(layer) => layer.forward(x),
                NormLayerEnum::ChannelsLast(layer) => layer.forward(x),
                NormLayerEnum::BatchNorm2d(layer) => layer.forward(x),
                NormLayerEnum::BasicDecBlk(layer) => {
                    // Need to reshape for LayerNorm
                    let [b, c, h, w] = x.dims();
                    let x_reshaped = x.reshape([b, c, h * w]).transpose();
                    let x_normed = layer.forward(x_reshaped);
                    x_normed.transpose().reshape([b, c, h, w])
                }
            };
        }

        // Apply activation
        x = match &self.act {
            ActLayerEnum::ReLU(act) => act.forward(x),
            ActLayerEnum::SiLU(act) => act.forward(x),
            ActLayerEnum::GELU(act) => act.forward(x),
        };

        x = self.conv2.forward(x);

        // Apply second norm layers
        for norm in &self.norm2 {
            x = match norm {
                NormLayerEnum::ChannelsFirst(layer) => layer.forward(x),
                NormLayerEnum::ChannelsLast(layer) => layer.forward(x),
                NormLayerEnum::BatchNorm2d(layer) => layer.forward(x),
                NormLayerEnum::BasicDecBlk(layer) => {
                    // Need to reshape for LayerNorm
                    let [b, c, h, w] = x.dims();
                    let x_reshaped = x.reshape([b, c, h * w]).transpose();
                    let x_normed = layer.forward(x_reshaped);
                    x_normed.transpose().reshape([b, c, h, w])
                }
            };
        }

        x
    }
}

/// RefinerPVTInChannels4 configuration
#[derive(Config, Debug)]
pub struct RefinerPVTInChannels4Config {
    /// The detailed model configuration.
    config: ModelConfig,
    #[config(default = 4)]
    in_channels: usize,
    #[config(default = "PVTv2Variant::B2")]
    backbone_variant: PVTv2Variant,
}

/// RefinerPVTInChannels4 module
#[derive(Module, Debug)]
pub struct RefinerPVTInChannels4<B: Backend> {
    backbone: BackboneWrapper<B>,
    squeeze_module: BasicDecBlk<B>,
    decoder: RefinerDecoder<B>,
}

impl RefinerPVTInChannels4Config {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BiRefNetResult<RefinerPVTInChannels4<B>> {
        // Create PVTv2 backbone with specified variant
        let backbone = create_backbone(BackboneType::PVTv2(self.backbone_variant.clone()), device);

        let channels = backbone.output_channels();

        let squeeze_block = match self.config.decoder.dec_att {
            DecAtt::None => SqueezeBlock::None,
            DecAtt::ASPP => SqueezeBlock::ASPP(0),
            DecAtt::ASPPDeformable => SqueezeBlock::ASPPDeformable(0),
        };

        let squeeze_module = BasicDecBlkConfig::new(squeeze_block)
            .with_in_channels(channels[3]) // Use highest resolution features
            .with_out_channels(channels[3])
            .init(device)?;

        let decoder = RefinerDecoderConfig::new(self.config.clone(), channels)
            .with_ms_supervision(false)
            .init(device)?;

        Ok(RefinerPVTInChannels4 {
            backbone,
            squeeze_module,
            decoder,
        })
    }
}

impl<B: Backend> RefinerPVTInChannels4<B> {
    /// Forward pass for the refiner
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, 4, height, width] (RGB + mask)
    pub fn forward(&self, x: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        // Handle channel mismatch: backbone expects 3 channels, but we have 4
        let x_rgb = if x.dims()[1] == 4 {
            // Take only RGB channels for backbone
            x.clone()
                .slice([0..x.dims()[0], 0..3, 0..x.dims()[2], 0..x.dims()[3]])
        } else {
            x.clone()
        };

        // Extract features from backbone
        let [x1, x2, x3, x4] = self.backbone.forward(x_rgb);

        // Apply squeeze module to deepest features
        let x4 = self.squeeze_module.forward(x4);

        // Forward through decoder with original input and extracted features
        let features = [x, x1, x2, x3, x4];
        self.decoder.forward(features)
    }
}

/// Refiner configuration
#[derive(Config, Debug)]
pub struct RefinerConfig {
    /// The detailed model configuration.
    config: ModelConfig,
    #[config(default = 4)]
    in_channels: usize,
    #[config(default = 48)]
    inter_channels: usize,
    #[config(default = 3)]
    out_channels: usize,
    /// PVTv2 variant to use as backbone
    #[config(default = "PVTv2Variant::B2")]
    pvt_variant: PVTv2Variant,
}

/// Refiner module
#[derive(Module, Debug)]
pub struct Refiner<B: Backend> {
    stem_layer: StemLayer<B>,
    backbone: BackboneWrapper<B>,
    squeeze_module: BasicDecBlk<B>,
    decoder: RefinerDecoder<B>,
}

impl RefinerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> BiRefNetResult<Refiner<B>> {
        // Default norm layer - use BatchNorm
        let norm_layer = "BN";

        let stem_layer = StemLayer::new(
            self.in_channels,
            self.inter_channels,
            self.out_channels,
            norm_layer,
            device,
        )?;

        // Create backbone with specified type
        let backbone = create_backbone(BackboneType::PVTv2(self.pvt_variant.clone()), device);

        let channels = backbone.output_channels();

        let squeeze_block = match self.config.decoder.dec_att {
            DecAtt::None => SqueezeBlock::None,
            DecAtt::ASPP => SqueezeBlock::ASPP(0),
            DecAtt::ASPPDeformable => SqueezeBlock::ASPPDeformable(0),
        };

        let squeeze_module = BasicDecBlkConfig::new(squeeze_block)
            .with_in_channels(channels[3]) // Use highest resolution features
            .with_out_channels(channels[3])
            .init(device)?;

        let decoder = RefinerDecoderConfig::new(self.config.clone(), channels)
            .with_ms_supervision(false)
            .init(device)?;

        Ok(Refiner {
            stem_layer,
            backbone,
            squeeze_module,
            decoder,
        })
    }
}

impl<B: Backend> Refiner<B> {
    /// Forward pass for the refiner
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, 4, height, width] (RGB + mask)
    pub fn forward(&self, x: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        // Apply stem layer to reduce channels and preprocess input
        let x_processed = self.stem_layer.forward(x);

        // Extract RGB channels for backbone (backbone expects 3-channel input)
        let x_rgb = x_processed.clone();

        // Extract features from backbone
        let [x1, x2, x3, x4] = self.backbone.forward(x_rgb);

        // Apply squeeze module to deepest features
        let x4 = self.squeeze_module.forward(x4);

        // Forward through decoder with processed input and extracted features
        let features = [x_processed, x1, x2, x3, x4];
        self.decoder.forward(features)
    }
}

/// RefUNet configuration
#[derive(Config, Debug)]
pub struct RefUNetConfig {
    #[config(default = 4)]
    in_channels: usize,
}

/// RefUNet module - U-Net based refinement
#[derive(Module, Debug)]
pub struct RefUNet<B: Backend> {
    // Encoder layers
    encoder_1: (Conv2d<B>, Conv2d<B>, BatchNorm<B, 2>, Relu),
    encoder_2: (MaxPool2d, Conv2d<B>, BatchNorm<B, 2>, Relu),
    encoder_3: (MaxPool2d, Conv2d<B>, BatchNorm<B, 2>, Relu),
    encoder_4: (MaxPool2d, Conv2d<B>, BatchNorm<B, 2>, Relu),
    pool4: MaxPool2d,

    // Decoder layers
    decoder_5: (Conv2d<B>, BatchNorm<B, 2>, Relu),
    decoder_4: (Conv2d<B>, BatchNorm<B, 2>, Relu),
    decoder_3: (Conv2d<B>, BatchNorm<B, 2>, Relu),
    decoder_2: (Conv2d<B>, BatchNorm<B, 2>, Relu),
    decoder_1: (Conv2d<B>, BatchNorm<B, 2>, Relu),

    conv_d0: Conv2d<B>,
}

impl RefUNetConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RefUNet<B> {
        // Encoder
        let encoder_1 = (
            Conv2dConfig::new([self.in_channels, 64], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            Conv2dConfig::new([64, 64], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            BatchNormConfig::new(64).init(device),
            Relu::new(),
        );

        let encoder_2 = (
            MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            Conv2dConfig::new([64, 64], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            BatchNormConfig::new(64).init(device),
            Relu::new(),
        );

        let encoder_3 = (
            MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            Conv2dConfig::new([64, 64], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            BatchNormConfig::new(64).init(device),
            Relu::new(),
        );

        let encoder_4 = (
            MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
            Conv2dConfig::new([64, 64], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            BatchNormConfig::new(64).init(device),
            Relu::new(),
        );

        let pool4 = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();

        // Decoder
        let decoder_5 = (
            Conv2dConfig::new([64, 64], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            BatchNormConfig::new(64).init(device),
            Relu::new(),
        );

        let decoder_4 = (
            Conv2dConfig::new([128, 64], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            BatchNormConfig::new(64).init(device),
            Relu::new(),
        );

        let decoder_3 = (
            Conv2dConfig::new([128, 64], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            BatchNormConfig::new(64).init(device),
            Relu::new(),
        );

        let decoder_2 = (
            Conv2dConfig::new([128, 64], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            BatchNormConfig::new(64).init(device),
            Relu::new(),
        );

        let decoder_1 = (
            Conv2dConfig::new([128, 64], [3, 3])
                .with_padding(PaddingConfig2d::Explicit(1, 1))
                .init(device),
            BatchNormConfig::new(64).init(device),
            Relu::new(),
        );

        let conv_d0 = Conv2dConfig::new([64, 1], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        RefUNet {
            encoder_1,
            encoder_2,
            encoder_3,
            encoder_4,
            pool4,
            decoder_5,
            decoder_4,
            decoder_3,
            decoder_2,
            decoder_1,
            conv_d0,
        }
    }
}

impl<B: Backend> RefUNet<B> {
    pub fn forward(&self, x: Vec<Tensor<B, 4>>) -> Vec<Tensor<B, 4>> {
        let mut outs = Vec::new();

        // Concatenate inputs if provided as a list
        let hx = if x.len() > 1 {
            Tensor::cat(x, 1)
        } else {
            x[0].clone()
        };

        // Encoder
        let hx1 = {
            let x = self.encoder_1.0.forward(hx);
            let x = self.encoder_1.1.forward(x);
            let x = self.encoder_1.2.forward(x);
            self.encoder_1.3.forward(x)
        };

        let hx2 = {
            let x = self.encoder_2.0.forward(hx1.clone());
            let x = self.encoder_2.1.forward(x);
            let x = self.encoder_2.2.forward(x);
            self.encoder_2.3.forward(x)
        };

        let hx3 = {
            let x = self.encoder_3.0.forward(hx2.clone());
            let x = self.encoder_3.1.forward(x);
            let x = self.encoder_3.2.forward(x);
            self.encoder_3.3.forward(x)
        };

        let hx4 = {
            let x = self.encoder_4.0.forward(hx3.clone());
            let x = self.encoder_4.1.forward(x);
            let x = self.encoder_4.2.forward(x);
            self.encoder_4.3.forward(x)
        };

        // Decoder
        let hx = {
            let x = self.pool4.forward(hx4.clone());
            let x = self.decoder_5.0.forward(x);
            let x = self.decoder_5.1.forward(x);
            self.decoder_5.2.forward(x)
        };

        let hx = {
            let [_, _, h, w] = hx4.dims();
            let hx_up = interpolate(
                hx,
                [h, w],
                InterpolateOptions::new(InterpolateMode::Bilinear),
            );
            Tensor::cat(vec![hx_up, hx4], 1)
        };

        let d4 = {
            let x = self.decoder_4.0.forward(hx);
            let x = self.decoder_4.1.forward(x);
            self.decoder_4.2.forward(x)
        };

        let hx = {
            let [_, _, h, w] = hx3.dims();
            let d4_up = interpolate(
                d4,
                [h, w],
                InterpolateOptions::new(InterpolateMode::Bilinear),
            );
            Tensor::cat(vec![d4_up, hx3], 1)
        };

        let d3 = {
            let x = self.decoder_3.0.forward(hx);
            let x = self.decoder_3.1.forward(x);
            self.decoder_3.2.forward(x)
        };

        let hx = {
            let [_, _, h, w] = hx2.dims();
            let d3_up = interpolate(
                d3,
                [h, w],
                InterpolateOptions::new(InterpolateMode::Bilinear),
            );
            Tensor::cat(vec![d3_up, hx2], 1)
        };

        let d2 = {
            let x = self.decoder_2.0.forward(hx);
            let x = self.decoder_2.1.forward(x);
            self.decoder_2.2.forward(x)
        };

        let hx = {
            let [_, _, h, w] = hx1.dims();
            let d2_up = interpolate(
                d2,
                [h, w],
                InterpolateOptions::new(InterpolateMode::Bilinear),
            );
            Tensor::cat(vec![d2_up, hx1], 1)
        };

        let d1 = {
            let x = self.decoder_1.0.forward(hx);
            let x = self.decoder_1.1.forward(x);
            self.decoder_1.2.forward(x)
        };

        let x = self.conv_d0.forward(d1);
        outs.push(x);

        outs
    }
}
