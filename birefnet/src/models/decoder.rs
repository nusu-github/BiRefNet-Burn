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
//! - `DecoderBlockModuleEnum`, `LateralBlockModuleEnum`: Enums to handle different types
//!   of decoder and lateral blocks, allowing for flexible architecture design.
//! - `GdtConvs`, `SimpleConvs`: Helper modules used within the decoder for specific
//!   convolutional operations.
//!
//! The decoder's design allows for multi-scale supervision, gradient-guided attention,
//! and various other features from the original BiRefNet paper.

use super::modules::{
    BasicDecBlk, BasicDecBlkConfig, BasicLatBlk, BasicLatBlkConfig, ResBlk, ResBlkConfig,
    SimpleConvs, SimpleConvsConfig,
};
use crate::{
    config::{DecBlk, LatBlk, ModelConfig, SqueezeBlock},
    error::BiRefNetResult,
};
use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, PaddingConfig2d, Relu,
    },
    prelude::*,
    tensor::{
        activation::sigmoid,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};

/// An enum to wrap different types of decoder blocks.
#[derive(Module, Debug)]
pub enum DecoderBlockModuleEnum<B: Backend> {
    BasicDecBlk(BasicDecBlk<B>),
    ResBlk(ResBlk<B>),
}

/// An enum to wrap different types of lateral blocks.
#[derive(Module, Debug)]
pub enum LateralBlockModuleEnum<B: Backend> {
    BasicLatBlk(BasicLatBlk<B>),
}

/// Configuration for the `Decoder` module.
#[derive(Config, Debug)]
pub struct DecoderConfig {
    /// The main model configuration.
    config: ModelConfig,
    /// The channel sizes of the encoder features.
    channels: [usize; 4],
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
    const N_DEC_IPT: usize = 64;
    const IC: usize = 64;
    const IPT_CHA_OPT: usize = 1;
    const _N: usize = 16;

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
        let split = {
            if self.config.decoder.dec_ipt {
                self.config.decoder.dec_ipt_split
            } else {
                false
            }
        };

        let mut ipt_blk5 = None;
        let mut ipt_blk4 = None;
        let mut ipt_blk3 = None;
        let mut ipt_blk2 = None;
        let mut ipt_blk1 = None;

        if self.config.decoder.dec_ipt {
            let out_channels = |x: usize| {
                if Self::IPT_CHA_OPT == 0 {
                    Self::N_DEC_IPT
                } else {
                    self.channels[x] / 8
                }
            };
            ipt_blk5 = Some(
                SimpleConvsConfig::new(
                    if split { 2_usize.pow(10) * 3 } else { 3 },
                    out_channels(0),
                )
                .with_inter_channels(Self::IC)
                .init(device),
            );
            ipt_blk4 = Some(
                SimpleConvsConfig::new(if split { 2_usize.pow(8) * 3 } else { 3 }, out_channels(0))
                    .with_inter_channels(Self::IC)
                    .init(device),
            );
            ipt_blk3 = Some(
                SimpleConvsConfig::new(if split { 2_usize.pow(6) * 3 } else { 3 }, out_channels(1))
                    .with_inter_channels(Self::IC)
                    .init(device),
            );
            ipt_blk2 = Some(
                SimpleConvsConfig::new(if split { 2_usize.pow(4) * 3 } else { 3 }, out_channels(2))
                    .with_inter_channels(Self::IC)
                    .init(device),
            );
            ipt_blk1 = Some(
                SimpleConvsConfig::new(if split { 2_usize.pow(0) * 3 } else { 3 }, out_channels(3))
                    .with_inter_channels(Self::IC)
                    .init(device),
            );
        }

        let in_channels = |x: usize, y: usize| {
            self.channels[x] + {
                if self.config.decoder.dec_ipt {
                    if Self::IPT_CHA_OPT == 0 {
                        Self::N_DEC_IPT
                    } else {
                        self.channels[y] / 8
                    }
                } else {
                    0
                }
            }
        };
        let decoder_block4 =
            self.create_decoder_block(in_channels(0, 0), self.channels[1], device)?;
        let decoder_block3 =
            self.create_decoder_block(in_channels(1, 0), self.channels[2], device)?;
        let decoder_block2 =
            self.create_decoder_block(in_channels(2, 1), self.channels[3], device)?;
        let decoder_block1 =
            self.create_decoder_block(in_channels(3, 2), self.channels[3] / 2, device)?;

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

        let lateral_block4 = self.create_lateral_block(self.channels[1], self.channels[1], device);
        let lateral_block3 = self.create_lateral_block(self.channels[2], self.channels[2], device);
        let lateral_block2 = self.create_lateral_block(self.channels[3], self.channels[3], device);

        let mut conv_ms_spvn_4 = None;
        let mut conv_ms_spvn_3 = None;
        let mut conv_ms_spvn_2 = None;

        let mut gdt_convs_4 = None;
        let mut gdt_convs_3 = None;
        let mut gdt_convs_2 = None;

        let mut gdt_convs_pred_4 = None;
        let mut gdt_convs_pred_3 = None;
        let mut gdt_convs_pred_2 = None;

        let mut gdt_convs_attn_4 = None;
        let mut gdt_convs_attn_3 = None;
        let mut gdt_convs_attn_2 = None;

        if self.config.decoder.ms_supervision {
            conv_ms_spvn_4 = Some(
                Conv2dConfig::new([self.channels[1], 1], [1, 1])
                    .with_padding(PaddingConfig2d::Valid)
                    .init(device),
            );
            conv_ms_spvn_3 = Some(
                Conv2dConfig::new([self.channels[2], 1], [1, 1])
                    .with_padding(PaddingConfig2d::Valid)
                    .init(device),
            );
            conv_ms_spvn_2 = Some(
                Conv2dConfig::new([self.channels[3], 1], [1, 1])
                    .with_padding(PaddingConfig2d::Valid)
                    .init(device),
            );

            if self.config.decoder.out_ref {
                gdt_convs_4 = Some(GdtConvs::init(
                    Conv2dConfig::new([self.channels[1], Self::_N], [3, 3])
                        .with_padding(PaddingConfig2d::Explicit(1, 1)),
                    BatchNormConfig::new(Self::_N),
                    device,
                ));
                gdt_convs_3 = Some(GdtConvs::init(
                    Conv2dConfig::new([self.channels[2], Self::_N], [3, 3])
                        .with_padding(PaddingConfig2d::Explicit(1, 1)),
                    BatchNormConfig::new(Self::_N),
                    device,
                ));
                gdt_convs_2 = Some(GdtConvs::init(
                    Conv2dConfig::new([self.channels[3], Self::_N], [3, 3])
                        .with_padding(PaddingConfig2d::Explicit(1, 1)),
                    BatchNormConfig::new(Self::_N),
                    device,
                ));

                gdt_convs_pred_4 = Some(
                    Conv2dConfig::new([Self::_N, 1], [1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .init(device),
                );
                gdt_convs_pred_3 = Some(
                    Conv2dConfig::new([Self::_N, 1], [1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .init(device),
                );
                gdt_convs_pred_2 = Some(
                    Conv2dConfig::new([Self::_N, 1], [1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .init(device),
                );

                gdt_convs_attn_4 = Some(
                    Conv2dConfig::new([Self::_N, 1], [1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .init(device),
                );
                gdt_convs_attn_3 = Some(
                    Conv2dConfig::new([Self::_N, 1], [1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .init(device),
                );
                gdt_convs_attn_2 = Some(
                    Conv2dConfig::new([Self::_N, 1], [1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .init(device),
                );
            }
        };

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
        })
    }

    fn create_decoder_block<B: Backend>(
        &self,
        in_channels: usize,
        out_channels: usize,
        device: &Device<B>,
    ) -> BiRefNetResult<DecoderBlockModuleEnum<B>> {
        match self.config.decoder.dec_blk {
            DecBlk::BasicDecBlk => Ok(DecoderBlockModuleEnum::BasicDecBlk(
                BasicDecBlkConfig::new(SqueezeBlock::ASPPDeformable(0))
                    .with_in_channels(in_channels)
                    .with_out_channels(out_channels)
                    .init(device)?,
            )),
            DecBlk::ResBlk => Ok(DecoderBlockModuleEnum::ResBlk(
                ResBlkConfig::new()
                    .with_in_channels(in_channels)
                    .with_out_channels(Some(out_channels))
                    .init(device)?,
            )),
        }
    }

    fn create_lateral_block<B: Backend>(
        &self,
        in_channels: usize,
        out_channels: usize,
        device: &Device<B>,
    ) -> LateralBlockModuleEnum<B> {
        match self.config.decoder.lat_blk {
            LatBlk::BasicLatBlk => LateralBlockModuleEnum::BasicLatBlk(
                BasicLatBlkConfig::new()
                    .with_in_channels(in_channels)
                    .with_out_channels(out_channels)
                    .init(device),
            ),
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
    decoder_block4: DecoderBlockModuleEnum<B>,
    decoder_block3: DecoderBlockModuleEnum<B>,
    decoder_block2: DecoderBlockModuleEnum<B>,
    decoder_block1: DecoderBlockModuleEnum<B>,
    lateral_block4: LateralBlockModuleEnum<B>,
    lateral_block3: LateralBlockModuleEnum<B>,
    lateral_block2: LateralBlockModuleEnum<B>,
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
}

impl<B: Backend> Decoder<B> {
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

        let x4 = {
            match &self.ipt_blk5 {
                Some(ipt_blk5) => {
                    let [_, _, h, w] = x4.dims();
                    let patches_batch = {
                        if self.split {
                            self.get_patches_batch(x.clone(), x4.clone())
                        } else {
                            x.clone()
                        }
                    };
                    Tensor::cat(
                        Vec::from([
                            x4,
                            ipt_blk5.forward(interpolate(
                                patches_batch,
                                [h, w],
                                InterpolateOptions::new(InterpolateMode::Bilinear),
                            )),
                        ]),
                        1,
                    )
                }
                None => x4,
            }
        };

        // Decoder block 4
        let p4 = {
            match &self.decoder_block4 {
                DecoderBlockModuleEnum::BasicDecBlk(decoder_block4) => decoder_block4.forward(x4),
                DecoderBlockModuleEnum::ResBlk(decoder_block4) => decoder_block4.forward(x4),
            }
        };
        // let m4 = None;
        let p4 = {
            match (&self.gdt_convs_4, &self.gdt_convs_attn_4) {
                (Some(gdt_convs_4), Some(gdt_convs_attn_4)) => {
                    let p4_gdt = gdt_convs_4.forward(p4.clone());
                    let gdt_attn_4 = sigmoid(gdt_convs_attn_4.forward(p4_gdt));
                    p4 * gdt_attn_4
                }
                _ => p4,
            }
        };
        let [_, _, h, w] = x3.dims();
        let _p4 = interpolate(
            p4,
            [h, w],
            InterpolateOptions::new(InterpolateMode::Bilinear),
        );
        let _p3 = _p4 + {
            match &self.lateral_block4 {
                LateralBlockModuleEnum::BasicLatBlk(lateral_block4) => {
                    lateral_block4.forward(x3.clone())
                }
            }
        };

        let _p3 = {
            match &self.ipt_blk4 {
                Some(ipt_blk4) => {
                    let [_, _, h, w] = x3.dims();
                    let patches_batch = {
                        if self.split {
                            self.get_patches_batch(x.clone(), _p3.clone())
                        } else {
                            x.clone()
                        }
                    };
                    Tensor::cat(
                        Vec::from([
                            _p3,
                            ipt_blk4.forward(interpolate(
                                patches_batch,
                                [h, w],
                                InterpolateOptions::new(InterpolateMode::Bilinear),
                            )),
                        ]),
                        1,
                    )
                }
                None => _p3,
            }
        };
        let p3 = {
            match &self.decoder_block3 {
                DecoderBlockModuleEnum::BasicDecBlk(decoder_block3) => decoder_block3.forward(_p3),
                DecoderBlockModuleEnum::ResBlk(decoder_block3) => decoder_block3.forward(_p3),
            }
        };
        // let m3 = None;
        let p3 = {
            match (&self.gdt_convs_3, &self.gdt_convs_attn_3) {
                (Some(gdt_convs_3), Some(gdt_convs_attn_3)) => {
                    let p3_gdt = gdt_convs_3.forward(p3.clone());
                    let gdt_attn_3 = sigmoid(gdt_convs_attn_3.forward(p3_gdt));
                    p3 * gdt_attn_3
                }
                _ => p3,
            }
        };
        let [_, _, h, w] = x2.dims();
        let _p3 = interpolate(
            p3,
            [h, w],
            InterpolateOptions::new(InterpolateMode::Bilinear),
        );
        let _p2 = _p3 + {
            match &self.lateral_block3 {
                LateralBlockModuleEnum::BasicLatBlk(lateral_block3) => {
                    lateral_block3.forward(x2.clone())
                }
            }
        };

        let _p2 = {
            match &self.ipt_blk3 {
                Some(ipt_blk3) => {
                    let [_, _, h, w] = x2.dims();
                    let patches_batch = {
                        if self.split {
                            self.get_patches_batch(x.clone(), _p2.clone())
                        } else {
                            x.clone()
                        }
                    };
                    Tensor::cat(
                        Vec::from([
                            _p2,
                            ipt_blk3.forward(interpolate(
                                patches_batch,
                                [h, w],
                                InterpolateOptions::new(InterpolateMode::Bilinear),
                            )),
                        ]),
                        1,
                    )
                }
                None => _p2,
            }
        };
        let p2 = {
            match &self.decoder_block2 {
                DecoderBlockModuleEnum::BasicDecBlk(decoder_block2) => decoder_block2.forward(_p2),
                DecoderBlockModuleEnum::ResBlk(decoder_block2) => decoder_block2.forward(_p2),
            }
        };
        // let m2 = None;
        let p2 = {
            match (&self.gdt_convs_2, &self.gdt_convs_attn_2) {
                (Some(gdt_convs_2), Some(gdt_convs_attn_2)) => {
                    let p2_gdt = gdt_convs_2.forward(p2.clone());
                    let gdt_attn_2 = sigmoid(gdt_convs_attn_2.forward(p2_gdt));
                    p2 * gdt_attn_2
                }
                _ => p2,
            }
        };
        let [_, _, h, w] = x1.dims();
        let _p2 = interpolate(
            p2,
            [h, w],
            InterpolateOptions::new(InterpolateMode::Bilinear),
        );
        let _p1 = _p2 + {
            match &self.lateral_block2 {
                LateralBlockModuleEnum::BasicLatBlk(lateral_block2) => {
                    lateral_block2.forward(x1.clone())
                }
            }
        };

        let _p1 = {
            match &self.ipt_blk2 {
                Some(ipt_blk2) => {
                    let [_, _, h, w] = x1.dims();
                    let patches_batch = {
                        if self.split {
                            self.get_patches_batch(x.clone(), _p1.clone())
                        } else {
                            x.clone()
                        }
                    };
                    Tensor::cat(
                        Vec::from([
                            _p1,
                            ipt_blk2.forward(interpolate(
                                patches_batch,
                                [h, w],
                                InterpolateOptions::new(InterpolateMode::Bilinear),
                            )),
                        ]),
                        1,
                    )
                }
                None => _p1,
            }
        };
        let _p1 = {
            match &self.decoder_block1 {
                DecoderBlockModuleEnum::BasicDecBlk(decoder_block1) => decoder_block1.forward(_p1),
                DecoderBlockModuleEnum::ResBlk(decoder_block1) => decoder_block1.forward(_p1),
            }
        };
        let [_, _, h, w] = x.dims();
        let _p1 = interpolate(
            _p1,
            [h, w],
            InterpolateOptions::new(InterpolateMode::Bilinear),
        );

        // let m1 = None;
        let _p1 = {
            match &self.ipt_blk1 {
                Some(ipt_blk1) => {
                    let [_, _, h, w] = x.dims();
                    let patches_batch = {
                        if self.split {
                            self.get_patches_batch(x, _p1.clone())
                        } else {
                            x
                        }
                    };
                    Tensor::cat(
                        Vec::from([
                            _p1,
                            ipt_blk1.forward(interpolate(
                                patches_batch,
                                [h, w],
                                InterpolateOptions::new(InterpolateMode::Bilinear),
                            )),
                        ]),
                        1,
                    )
                }
                None => _p1,
            }
        };
        self.conv_out1.forward(_p1)
    }

    fn get_patches_batch(&self, x: Tensor<B, 4>, p: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, c, h, w] = p.dims();
        let [_, _, h_, w_] = x.dims();
        let patch_count = (h_ / h) * (w_ / w);
        let mut patches = Vec::with_capacity(patch_count);
        for i in (0..w_).step_by(w) {
            let column_x = x.clone().slice([0..b, 0..c, 0..h_, i..i + w]);
            for j in (0..h_).step_by(h) {
                let patch = column_x.clone().slice([0..b, 0..c, j..j + h, 0..w]);
                patches.push(patch);
            }
        }
        Tensor::cat(patches, 1)
    }
}
