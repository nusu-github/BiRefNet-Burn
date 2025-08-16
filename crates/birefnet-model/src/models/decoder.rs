//! # Decoder for BiRefNet
//!
//! This module implements the decoder part of the BiRefNet model. The decoder is
//! responsible for taking the feature maps from the backbone encoder and progressively
//! upsampling them to generate the final high-resolution segmentation mask.
//!
//! ## Key Components
//!
//! - `DecoderConfig`: Configuration struct for the decoder, specifying parameters like
//!   the number of channels, block types, and attention mechanisms.
//! - `Decoder`: The main decoder struct, which builds the decoder architecture based on
//!   the provided configuration.
//! - `DecoderBlockModule`, `LateralBlockModule`: Enums to handle different types
//!   of decoder and lateral blocks, allowing for flexible architecture design.
//! - `GdtConvs`, `SimpleConvs`: Helper modules used within the decoder for specific
//!   convolutional operations.
//!
//! The decoder's design allows for multi-scale supervision, gradient-guided attention,
//! and various other features from the original BiRefNet paper.

use burn::{
    module::Ignored,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, PaddingConfig2d, Relu,
    },
    prelude::*,
    tensor::activation::sigmoid,
};

use super::modules::{
    BasicDecBlk, BasicDecBlkConfig, BasicLatBlk, BasicLatBlkConfig, ResBlk, ResBlkConfig,
    SimpleConvs, SimpleConvsConfig,
};
use crate::{
    config::{DecBlk, InterpolationStrategy, LateralBlock, ModelConfig, SqueezeBlock},
    error::BiRefNetResult,
    models::modules::utils::intelligent_interpolate,
};

/// An enum to wrap different types of decoder blocks.
#[derive(Module, Debug)]
pub enum DecoderBlockModule<B: Backend> {
    BasicDecBlk(BasicDecBlk<B>),
    ResBlk(ResBlk<B>),
}

/// An enum to wrap different types of lateral blocks.
#[derive(Module, Debug)]
pub enum LateralBlockModule<B: Backend> {
    BasicLatBlk(BasicLatBlk<B>),
}

/// Configuration for the `Decoder` module.
#[derive(Config, Debug)]
pub struct DecoderConfig {
    /// The main model configuration.
    config: ModelConfig,
    /// The channel sizes of the encoder features.
    channels: [usize; 4],
    /// Interpolation strategy for tensor resizing operations.
    interpolation_strategy: InterpolationStrategy,
}

/// A small convolutional module used for gradient guidance.
#[derive(Module, Debug)]
struct GdtConvs<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B, 2>,
    relu: Relu,
}

impl<B: Backend> GdtConvs<B> {
    fn init(
        conv2d_config: Conv2dConfig,
        batch_norm_config: BatchNormConfig,
        device: &Device<B>,
    ) -> Self {
        let conv = conv2d_config.init(device);
        let bn = batch_norm_config.init(device);
        let relu = Relu::new();
        Self { conv, bn, relu }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(x);
        let x = self.bn.forward(x);
        self.relu.forward(x)
    }
}

impl DecoderConfig {
    const IC: usize = 64;
    const IPT_CHA_OPT: usize = 1;
    const N_DEC_IPT: usize = 64;
    const _N: usize = 16;

    /// Creates IPT blocks for the decoder with reduced duplication.
    fn create_ipt_blocks<B: Backend>(
        &self,
        split: bool,
        device: &B::Device,
    ) -> [Option<SimpleConvs<B>>; 5] {
        if !self.config.decoder.dec_ipt {
            return [None, None, None, None, None];
        }

        let powers = [10, 8, 6, 4, 0];
        let channel_indices = [0, 0, 1, 2, 3];
        let mut blocks = [None, None, None, None, None];

        for (i, (&power, &channel_idx)) in powers.iter().zip(channel_indices.iter()).enumerate() {
            let out_channels = if Self::IPT_CHA_OPT == 0 {
                Self::N_DEC_IPT
            } else {
                self.channels[channel_idx] / 8
            };

            let in_channels = if split { 2_usize.pow(power) * 3 } else { 3 };

            blocks[i] = Some(
                SimpleConvsConfig::new(in_channels, out_channels)
                    .with_inter_channels(Self::IC)
                    .init(device),
            );
        }

        blocks
    }

    /// Creates multi-scale supervision and GDT convolution layers.
    fn create_supervision_layers<B: Backend>(
        &self,
        device: &B::Device,
    ) -> (
        [Option<Conv2d<B>>; 3],   // conv_ms_spvn
        [Option<GdtConvs<B>>; 3], // gdt_convs
        [Option<Conv2d<B>>; 3],   // gdt_convs_pred
        [Option<Conv2d<B>>; 3],   // gdt_convs_attn
    ) {
        let mut conv_ms_spvn = [None, None, None];
        let mut gdt_convs = [None, None, None];
        let mut gdt_convs_pred = [None, None, None];
        let mut gdt_convs_attn = [None, None, None];

        if !self.config.decoder.ms_supervision {
            return (conv_ms_spvn, gdt_convs, gdt_convs_pred, gdt_convs_attn);
        }

        for i in 0..3 {
            let channel = self.channels[i + 1];

            conv_ms_spvn[i] = Some(
                Conv2dConfig::new([channel, 1], [1, 1])
                    .with_padding(PaddingConfig2d::Valid)
                    .init(device),
            );

            if self.config.decoder.out_ref {
                gdt_convs[i] = Some(GdtConvs::init(
                    Conv2dConfig::new([channel, Self::_N], [3, 3])
                        .with_padding(PaddingConfig2d::Explicit(1, 1)),
                    BatchNormConfig::new(Self::_N),
                    device,
                ));

                gdt_convs_pred[i] = Some(
                    Conv2dConfig::new([Self::_N, 1], [1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .init(device),
                );

                gdt_convs_attn[i] = Some(
                    Conv2dConfig::new([Self::_N, 1], [1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .init(device),
                );
            }
        }

        (conv_ms_spvn, gdt_convs, gdt_convs_pred, gdt_convs_attn)
    }

    /// Creates decoder blocks with reduced duplication.
    fn create_decoder_blocks<B: Backend>(
        &self,
        device: &B::Device,
    ) -> BiRefNetResult<[DecoderBlockModule<B>; 4]> {
        let channel_configs = [
            (0, 0, self.channels[1]),     // decoder_block4
            (1, 0, self.channels[2]),     // decoder_block3
            (2, 1, self.channels[3]),     // decoder_block2
            (3, 2, self.channels[3] / 2), // decoder_block1
        ];

        let mut blocks = Vec::with_capacity(4);

        for &(x, y, out_channels) in &channel_configs {
            let in_channels = self.channels[x] + {
                if self.config.decoder.dec_ipt {
                    if Self::IPT_CHA_OPT == 0 {
                        Self::N_DEC_IPT
                    } else {
                        self.channels[y] / 8
                    }
                } else {
                    0
                }
            };

            blocks.push(self.create_decoder_block(
                in_channels,
                out_channels,
                self.config.interpolation.clone(),
                device,
            )?);
        }

        // Convert Vec to array safely
        blocks.try_into().map_err(|_| {
            crate::error::BiRefNetError::ModelInitializationFailed {
                reason: "Failed to create decoder blocks".to_string(),
            }
        })
    }

    /// Creates lateral blocks with reduced duplication.
    fn create_lateral_blocks<B: Backend>(&self, device: &B::Device) -> [LateralBlockModule<B>; 3] {
        [
            self.create_lateral_block(self.channels[1], self.channels[1], device),
            self.create_lateral_block(self.channels[2], self.channels[2], device),
            self.create_lateral_block(self.channels[3], self.channels[3], device),
        ]
    }

    /// Initializes a `Decoder` module.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to create the module on.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn init<B: Backend>(&self, device: &B::Device) -> BiRefNetResult<Decoder<B>> {
        let split = self.config.decoder.dec_ipt && self.config.decoder.dec_ipt_split;

        // Create IPT blocks using helper function
        let [ipt_blk5, ipt_blk4, ipt_blk3, ipt_blk2, ipt_blk1] =
            self.create_ipt_blocks(split, device);

        // Create decoder blocks using helper function
        let [decoder_block4, decoder_block3, decoder_block2, decoder_block1] =
            self.create_decoder_blocks(device)?;

        // Create lateral blocks using helper function
        let [lateral_block4, lateral_block3, lateral_block2] = self.create_lateral_blocks(device);

        // Create supervision layers using helper function
        let (
            [conv_ms_spvn_4, conv_ms_spvn_3, conv_ms_spvn_2],
            [gdt_convs_4, gdt_convs_3, gdt_convs_2],
            [gdt_convs_pred_4, gdt_convs_pred_3, gdt_convs_pred_2],
            [gdt_convs_attn_4, gdt_convs_attn_3, gdt_convs_attn_2],
        ) = self.create_supervision_layers(device);

        // Create final output convolution
        let in_channels = (self.channels[3] / 2) + {
            if self.config.decoder.dec_ipt {
                if Self::IPT_CHA_OPT == 0 {
                    Self::N_DEC_IPT
                } else {
                    self.channels[3] / 8
                }
            } else {
                0
            }
        };
        let conv_out1 = Conv2dConfig::new([in_channels, 1], [1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .init(device);

        let interpolation_strategy = Ignored(self.interpolation_strategy.clone());

        Ok(Decoder {
            split,
            ipt_blk5,
            ipt_blk4,
            ipt_blk3,
            ipt_blk2,
            ipt_blk1,
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
            gdt_convs_4,
            gdt_convs_3,
            gdt_convs_2,
            gdt_convs_pred_4,
            gdt_convs_pred_3,
            gdt_convs_pred_2,
            gdt_convs_attn_4,
            gdt_convs_attn_3,
            gdt_convs_attn_2,
            interpolation_strategy,
        })
    }

    fn create_decoder_block<B: Backend>(
        &self,
        in_channels: usize,
        out_channels: usize,
        interpolation_strategy: InterpolationStrategy,
        device: &Device<B>,
    ) -> BiRefNetResult<DecoderBlockModule<B>> {
        match self.config.decoder.dec_blk {
            DecBlk::BasicDecBlk => {
                Ok(DecoderBlockModule::BasicDecBlk(
                    BasicDecBlkConfig::new(SqueezeBlock::ASPPDeformable(0), interpolation_strategy)
                        .with_in_channels(in_channels)
                        .with_out_channels(out_channels)
                        .init(device)?,
                ))
            }
            DecBlk::ResBlk => {
                Ok(DecoderBlockModule::ResBlk(
                    ResBlkConfig::new(interpolation_strategy)
                        .with_in_channels(in_channels)
                        .with_out_channels(Some(out_channels))
                        .init(device)?,
                ))
            }
        }
    }

    fn create_lateral_block<B: Backend>(
        &self,
        in_channels: usize,
        out_channels: usize,
        device: &Device<B>,
    ) -> LateralBlockModule<B> {
        match self.config.decoder.lat_blk {
            LateralBlock::BasicLatBlk => {
                LateralBlockModule::BasicLatBlk(
                    BasicLatBlkConfig::new()
                        .with_in_channels(in_channels)
                        .with_out_channels(out_channels)
                        .init(device),
                )
            }
        }
    }
}

/// The decoder module of BiRefNet.
#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    split: bool,
    ipt_blk5: Option<SimpleConvs<B>>,
    ipt_blk4: Option<SimpleConvs<B>>,
    ipt_blk3: Option<SimpleConvs<B>>,
    ipt_blk2: Option<SimpleConvs<B>>,
    ipt_blk1: Option<SimpleConvs<B>>,
    decoder_block4: DecoderBlockModule<B>,
    decoder_block3: DecoderBlockModule<B>,
    decoder_block2: DecoderBlockModule<B>,
    decoder_block1: DecoderBlockModule<B>,
    lateral_block4: LateralBlockModule<B>,
    lateral_block3: LateralBlockModule<B>,
    lateral_block2: LateralBlockModule<B>,
    conv_out1: Conv2d<B>,
    conv_ms_spvn_4: Option<Conv2d<B>>,
    conv_ms_spvn_3: Option<Conv2d<B>>,
    conv_ms_spvn_2: Option<Conv2d<B>>,
    gdt_convs_4: Option<GdtConvs<B>>,
    gdt_convs_3: Option<GdtConvs<B>>,
    gdt_convs_2: Option<GdtConvs<B>>,
    gdt_convs_pred_4: Option<Conv2d<B>>,
    gdt_convs_pred_3: Option<Conv2d<B>>,
    gdt_convs_pred_2: Option<Conv2d<B>>,
    gdt_convs_attn_4: Option<Conv2d<B>>,
    gdt_convs_attn_3: Option<Conv2d<B>>,
    gdt_convs_attn_2: Option<Conv2d<B>>,
    /// Interpolation strategy for tensor resizing operations.
    interpolation_strategy: Ignored<InterpolationStrategy>,
}

impl<B: Backend> Decoder<B> {
    /// Processes IPT block if present, otherwise returns the input feature unchanged.
    fn process_ipt_block(
        &self,
        ipt_blk: &Option<SimpleConvs<B>>,
        x: &Tensor<B, 4>,
        feature: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        match ipt_blk {
            Some(blk) => {
                let [_, _, h, w] = feature.dims();
                let patches_batch = if self.split {
                    self.get_patches_batch(x.clone(), feature.clone())
                } else {
                    x.clone()
                };

                Tensor::cat(
                    vec![
                        feature,
                        blk.forward(intelligent_interpolate(
                            patches_batch,
                            [h, w],
                            &self.interpolation_strategy.0,
                        )),
                    ],
                    1,
                )
            }
            None => feature,
        }
    }

    /// Processes decoder block based on its type.
    fn process_decoder_block(
        &self,
        block: &DecoderBlockModule<B>,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        match block {
            DecoderBlockModule::BasicDecBlk(blk) => blk.forward(input),
            DecoderBlockModule::ResBlk(blk) => blk.forward(input),
        }
    }

    /// Applies GDT attention if both convolution and attention layers are present.
    fn apply_gdt_attention(
        &self,
        input: Tensor<B, 4>,
        gdt_conv: &Option<GdtConvs<B>>,
        gdt_attn: &Option<Conv2d<B>>,
    ) -> Tensor<B, 4> {
        match (gdt_conv, gdt_attn) {
            (Some(conv), Some(attn)) => {
                let gdt = conv.forward(input.clone());
                let attention = sigmoid(attn.forward(gdt));
                input * attention
            }
            _ => input,
        }
    }

    /// Processes lateral block connection.
    fn process_lateral_block(
        &self,
        block: &LateralBlockModule<B>,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        match block {
            LateralBlockModule::BasicLatBlk(blk) => blk.forward(input),
        }
    }

    /// Forward pass for the decoder.
    ///
    /// # Arguments
    ///
    /// * `features` - A 5-element array containing the input image and the four
    ///   feature maps from the encoder `[x, x1, x2, x3, x4]`.
    ///
    /// # Returns
    ///
    /// The final segmentation map.
    pub fn forward(&self, features: [Tensor<B, 4>; 5]) -> Tensor<B, 4> {
        let [x, x1, x2, x3, x4] = features;

        // Level 4: Process with IPT block 5
        let x4_processed = self.process_ipt_block(&self.ipt_blk5, &x, x4);

        // Decoder block 4 -> GDT attention
        let mut p4 = self.process_decoder_block(&self.decoder_block4, x4_processed);
        p4 = self.apply_gdt_attention(p4, &self.gdt_convs_4, &self.gdt_convs_attn_4);

        // Interpolate and add lateral connection
        let [_, _, h3, w3] = x3.dims();
        let p4_interp = intelligent_interpolate(p4, [h3, w3], &self.interpolation_strategy.0);
        let p3_lateral = p4_interp + self.process_lateral_block(&self.lateral_block4, x3);

        // Level 3: Process with IPT block 4
        let p3_with_ipt = self.process_ipt_block(&self.ipt_blk4, &x, p3_lateral);

        // Decoder block 3 -> GDT attention
        let mut p3 = self.process_decoder_block(&self.decoder_block3, p3_with_ipt);
        p3 = self.apply_gdt_attention(p3, &self.gdt_convs_3, &self.gdt_convs_attn_3);

        // Interpolate and add lateral connection
        let [_, _, h2, w2] = x2.dims();
        let p3_interp = intelligent_interpolate(p3, [h2, w2], &self.interpolation_strategy.0);
        let p2_lateral = p3_interp + self.process_lateral_block(&self.lateral_block3, x2);

        // Level 2: Process with IPT block 3
        let p2_with_ipt = self.process_ipt_block(&self.ipt_blk3, &x, p2_lateral);

        // Decoder block 2 -> GDT attention
        let mut p2 = self.process_decoder_block(&self.decoder_block2, p2_with_ipt);
        p2 = self.apply_gdt_attention(p2, &self.gdt_convs_2, &self.gdt_convs_attn_2);

        // Interpolate and add lateral connection
        let [_, _, h1, w1] = x1.dims();
        let p2_interp = intelligent_interpolate(p2, [h1, w1], &self.interpolation_strategy.0);
        let p1_lateral = p2_interp + self.process_lateral_block(&self.lateral_block2, x1);

        // Level 1: Process with IPT block 2
        let p1_with_ipt = self.process_ipt_block(&self.ipt_blk2, &x, p1_lateral);

        // Decoder block 1
        let mut p1 = self.process_decoder_block(&self.decoder_block1, p1_with_ipt);

        // Final interpolation to original resolution
        let [_, _, h_orig, w_orig] = x.dims();
        p1 = intelligent_interpolate(p1, [h_orig, w_orig], &self.interpolation_strategy.0);

        // Final IPT processing
        let p1_final = self.process_ipt_block(&self.ipt_blk1, &x, p1);

        // Output convolution
        self.conv_out1.forward(p1_final)
    }

    fn get_patches_batch(&self, x: Tensor<B, 4>, p: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, c, h, w] = p.dims();
        let [_, _, h_, w_] = x.dims();
        let patch_count = (h_ / h) * (w_ / w);
        let mut patches = Vec::with_capacity(patch_count);
        for i in (0..w_).step_by(w) {
            let column_x = x.clone().slice(s![0..b, 0..c, 0..h_, i..i + w]);
            for j in (0..h_).step_by(h) {
                let patch = column_x.clone().slice([0..b, 0..c, j..j + h, 0..w]);
                patches.push(patch);
            }
        }
        Tensor::cat(patches, 1)
    }
}
