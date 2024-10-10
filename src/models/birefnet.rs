use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNormConfig, PaddingConfig2d, Relu,
    },
    prelude::*,
    tensor::{
        activation::sigmoid,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};

use super::{
    build_backbone, ASPPConfig, ASPPDeformable, ASPPDeformableConfig, BackboneEnum, BasicDecBlk,
    BasicDecBlkConfig, BasicLatBlk, BasicLatBlkConfig, ResBlk, ResBlkConfig, ASPP,
};
use crate::config::{DecBlk, LatBlk, MulSclIpt};
use crate::{
    config::{ModelConfig, SqueezeBlock},
    special::{Identity, Sequential, SequentialType},
};

#[derive(Module, Debug)]
pub enum SqueezeBlockModule<B: Backend> {
    BasicDecBlk(BasicDecBlk<B>),
    ResBlk(ResBlk<B>),
    ASPP(ASPP<B>),
    ASPPDeformable(ASPPDeformable<B>),
}

#[derive(Config, Debug)]
pub struct BiRefNetConfig {
    config: ModelConfig,
    bb_pretrained: bool,
}

impl BiRefNetConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> BiRefNet<B> {
        let bb = build_backbone(&self.config, self.bb_pretrained, device);
        let channels = self.config.lateral_channels_in_collection();
        let cxt = self.config.cxt();

        let squeeze_module = if self.config.squeeze_block == SqueezeBlock::None {
            vec![]
        } else {
            let mut squeeze_module = Vec::with_capacity(self.config.squeeze_block.count());

            for _ in 0..self.config.squeeze_block.count() {
                match self.config.squeeze_block {
                    SqueezeBlock::BasicDecBlk(_) => {
                        let model = BasicDecBlkConfig::new(SqueezeBlock::ASPPDeformable(0))
                            .with_in_channels(channels[0] + cxt.iter().sum::<usize>())
                            .with_out_channels(channels[0])
                            .init(device);
                        squeeze_module.push(SqueezeBlockModule::BasicDecBlk(model));
                    }
                    SqueezeBlock::ResBlk(_) => {
                        let model = ResBlkConfig::new()
                            .with_in_channels(channels[0] + cxt.iter().sum::<usize>())
                            .with_out_channels(Some(channels[0]))
                            .init(device);
                        squeeze_module.push(SqueezeBlockModule::ResBlk(model));
                    }
                    SqueezeBlock::ASPP(_) => {
                        let model = ASPPConfig::new()
                            .with_in_channels(channels[0] + cxt.iter().sum::<usize>())
                            .with_out_channels(Some(channels[0]))
                            .init(device);
                        squeeze_module.push(SqueezeBlockModule::ASPP(model));
                    }
                    SqueezeBlock::ASPPDeformable(_) => {
                        let model = ASPPDeformableConfig::new()
                            .with_in_channels(channels[0] + cxt.iter().sum::<usize>())
                            .with_out_channels(Some(channels[0]))
                            .init(device);
                        squeeze_module.push(SqueezeBlockModule::ASPPDeformable(model));
                    }
                    SqueezeBlock::None => unreachable!(),
                };
            }

            squeeze_module
        };

        // TODO: refine

        let mul_scl_ipt = match self.config.mul_scl_ipt {
            MulSclIpt::None => MulSclIpt_::None(Identity::new()),
            MulSclIpt::Add => MulSclIpt_::Add(Identity::new()),
            MulSclIpt::Cat => MulSclIpt_::Cat(Identity::new()),
        };

        BiRefNet {
            mul_scl_ipt,
            cxt,
            bb,
            squeeze_module,
            decoder: DecoderConfig::new(self.config.clone(), channels).init(device),
        }
    }
}

#[derive(Module, Debug, Clone)]
enum MulSclIpt_ {
    None(Identity),
    Add(Identity),
    Cat(Identity),
}

#[derive(Module, Debug)]
pub struct BiRefNet<B: Backend> {
    mul_scl_ipt: MulSclIpt_,
    cxt: [usize; 3],
    bb: BackboneEnum<B>,
    squeeze_module: Vec<SqueezeBlockModule<B>>,
    decoder: Decoder<B>,
}

impl<B: Backend> BiRefNet<B> {
    pub fn forward_enc(&self, x: Tensor<B, 4>) -> [Tensor<B, 4>; 4] {
        let [x1, x2, x3, x4] = match &self.bb {
            BackboneEnum::SwinTransformer(bb) => bb.forward(x.clone()),
        };
        let [x1, x2, x3, x4] = match self.mul_scl_ipt {
            MulSclIpt_::None(_) => [x1, x2, x3, x4],
            MulSclIpt_::Add(_) => {
                let [_, _, h, w] = x.dims();
                let [x1_, x2_, x3_, x4_] = match &self.bb {
                    BackboneEnum::SwinTransformer(bb) => bb.forward(interpolate(
                        x.clone(),
                        [h / 2, w / 2],
                        InterpolateOptions::new(InterpolateMode::Bilinear),
                    )),
                };

                let [_, _, h, w] = x1.dims();
                let x1 = x1
                    + interpolate(
                        x1_,
                        [h / 2, w / 2],
                        InterpolateOptions::new(InterpolateMode::Bilinear),
                    );
                let [_, _, h, w] = x2.dims();
                let x2 = x2
                    + interpolate(
                        x2_,
                        [h / 2, w / 2],
                        InterpolateOptions::new(InterpolateMode::Bilinear),
                    );
                let [_, _, h, w] = x3.dims();
                let x3 = x3
                    + interpolate(
                        x3_,
                        [h / 2, w / 2],
                        InterpolateOptions::new(InterpolateMode::Bilinear),
                    );
                let [_, _, h, w] = x4.dims();
                let x4 = x4
                    + interpolate(
                        x4_,
                        [h / 2, w / 2],
                        InterpolateOptions::new(InterpolateMode::Bilinear),
                    );

                [x1, x2, x3, x4]
            }
            MulSclIpt_::Cat(_) => {
                let [_, _, h, w] = x.dims();
                let [x1_, x2_, x3_, x4_] = match &self.bb {
                    BackboneEnum::SwinTransformer(bb) => bb.forward(interpolate(
                        x.clone(),
                        [h / 2, w / 2],
                        InterpolateOptions::new(InterpolateMode::Bilinear),
                    )),
                };

                let [_, _, h, w] = x1.dims();
                let x1 = Tensor::cat(
                    vec![
                        x1,
                        interpolate(
                            x1_,
                            [h, w],
                            InterpolateOptions::new(InterpolateMode::Bilinear),
                        ),
                    ],
                    1,
                );
                let [_, _, h, w] = x2.dims();
                let x2 = Tensor::cat(
                    vec![
                        x2,
                        interpolate(
                            x2_,
                            [h, w],
                            InterpolateOptions::new(InterpolateMode::Bilinear),
                        ),
                    ],
                    1,
                );
                let [_, _, h, w] = x3.dims();
                let x3 = Tensor::cat(
                    vec![
                        x3,
                        interpolate(
                            x3_,
                            [h, w],
                            InterpolateOptions::new(InterpolateMode::Bilinear),
                        ),
                    ],
                    1,
                );
                let [_, _, h, w] = x4.dims();
                let x4 = Tensor::cat(
                    vec![
                        x4,
                        interpolate(
                            x4_,
                            [h, w],
                            InterpolateOptions::new(InterpolateMode::Bilinear),
                        ),
                    ],
                    1,
                );

                [x1, x2, x3, x4]
            }
        };

        if self.cxt[0] > 0 {
            let mut combined = vec![];
            let [_, _, h, w] = x4.dims();
            combined.push(interpolate(
                x1.clone(),
                [h, w],
                InterpolateOptions::new(InterpolateMode::Nearest),
            ));
            combined.push(interpolate(
                x2.clone(),
                [h, w],
                InterpolateOptions::new(InterpolateMode::Nearest),
            ));
            combined.push(interpolate(
                x3.clone(),
                [h, w],
                InterpolateOptions::new(InterpolateMode::Nearest),
            ));
            let len_cxt = self.cxt.len();
            let mut result = combined[combined.len().saturating_sub(len_cxt)..].to_vec();
            result.push(x4);

            let x4 = Tensor::cat(result, 1);
            [x1, x2, x3, x4]
        } else {
            [x1, x2, x3, x4]
        }
    }
    pub fn forward_ori(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // ########## Encoder ##########
        let [x1, x2, x3, x4] = self.forward_enc(x.clone());
        let mut x4 = x4;
        for squeeze_module in &self.squeeze_module {
            match squeeze_module {
                SqueezeBlockModule::BasicDecBlk(model) => {
                    x4 = model.forward(x4);
                }
                SqueezeBlockModule::ResBlk(model) => {
                    x4 = model.forward(x4);
                }
                SqueezeBlockModule::ASPP(model) => {
                    x4 = model.forward(x4);
                }
                SqueezeBlockModule::ASPPDeformable(model) => {
                    x4 = model.forward(x4);
                }
            }
        }
        // ########## Decoder ##########
        let features = [x, x1, x2, x3, x4];

        self.decoder.forward(features)
    }
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.forward_ori(x)
    }
}

#[derive(Module, Debug)]
pub enum DecoderBlockModuleEnum<B: Backend> {
    BasicDecBlk(BasicDecBlk<B>),
    ResBlk(ResBlk<B>),
}

#[derive(Module, Debug)]
pub enum LateralBlockModuleEnum<B: Backend> {
    BasicLatBlk(BasicLatBlk<B>),
}

#[derive(Config, Debug)]
pub struct DecoderConfig {
    config: ModelConfig,
    channels: [usize; 4],
}

impl DecoderConfig {
    const N_DEC_IPT: usize = 64;
    const IC: usize = 64;
    const IPT_CHA_OPT: usize = 1;
    const _N: usize = 16;

    pub fn init<B: Backend>(&self, device: &B::Device) -> Decoder<B> {
        let split = if self.config.dec_ipt {
            self.config.dec_ipt_split.clone()
        } else {
            false
        };

        let mut ipt_blk5 = None;
        let mut ipt_blk4 = None;
        let mut ipt_blk3 = None;
        let mut ipt_blk2 = None;
        let mut ipt_blk1 = None;

        if self.config.dec_ipt {
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
            self.channels[x]
                + if self.config.dec_ipt {
                    if Self::IPT_CHA_OPT == 0 {
                        Self::N_DEC_IPT
                    } else {
                        self.channels[y] / 8
                    }
                } else {
                    0
                }
        };
        let decoder_block4 = self.create_decoder_block(in_channels(0, 0), self.channels[1], device);
        let decoder_block3 = self.create_decoder_block(in_channels(1, 0), self.channels[2], device);
        let decoder_block2 = self.create_decoder_block(in_channels(2, 1), self.channels[3], device);
        let decoder_block1 =
            self.create_decoder_block(in_channels(3, 2), self.channels[3] / 2, device);

        let in_channels = (self.channels[3] / 2)
            + if self.config.dec_ipt {
                if Self::IPT_CHA_OPT == 0 {
                    Self::N_DEC_IPT
                } else {
                    self.channels[3] / 8
                }
            } else {
                0
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

        if self.config.ms_supervision {
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

            if self.config.out_ref {
                gdt_convs_4 = Some(Sequential::new(vec![
                    SequentialType::Conv2d(
                        Conv2dConfig::new([self.channels[1], Self::_N], [3, 3])
                            .with_stride([1, 1])
                            .with_padding(PaddingConfig2d::Explicit(1, 1))
                            .init(device),
                    ),
                    SequentialType::BatchNorm2d(BatchNormConfig::new(Self::_N).init(device)),
                    SequentialType::ReLU(Relu::new()),
                ]));
                gdt_convs_3 = Some(Sequential::new(vec![
                    SequentialType::Conv2d(
                        Conv2dConfig::new([self.channels[2], Self::_N], [3, 3])
                            .with_stride([1, 1])
                            .with_padding(PaddingConfig2d::Explicit(1, 1))
                            .init(device),
                    ),
                    SequentialType::BatchNorm2d(BatchNormConfig::new(Self::_N).init(device)),
                    SequentialType::ReLU(Relu::new()),
                ]));
                gdt_convs_2 = Some(Sequential::new(vec![
                    SequentialType::Conv2d(
                        Conv2dConfig::new([self.channels[3], Self::_N], [3, 3])
                            .with_stride([1, 1])
                            .with_padding(PaddingConfig2d::Explicit(1, 1))
                            .init(device),
                    ),
                    SequentialType::BatchNorm2d(BatchNormConfig::new(Self::_N).init(device)),
                    SequentialType::ReLU(Relu::new()),
                ]));

                gdt_convs_pred_4 = Some(
                    Conv2dConfig::new([Self::_N, 1], [1, 1])
                        .with_stride([1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .init(device),
                );
                gdt_convs_pred_3 = Some(
                    Conv2dConfig::new([Self::_N, 1], [1, 1])
                        .with_stride([1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .init(device),
                );
                gdt_convs_pred_2 = Some(
                    Conv2dConfig::new([Self::_N, 1], [1, 1])
                        .with_stride([1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .init(device),
                );

                gdt_convs_attn_4 = Some(
                    Conv2dConfig::new([Self::_N, 1], [1, 1])
                        .with_stride([1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .init(device),
                );
                gdt_convs_attn_3 = Some(
                    Conv2dConfig::new([Self::_N, 1], [1, 1])
                        .with_stride([1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .init(device),
                );
                gdt_convs_attn_2 = Some(
                    Conv2dConfig::new([Self::_N, 1], [1, 1])
                        .with_stride([1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .init(device),
                );
            }
        };

        Decoder {
            dec_ipt: self.config.dec_ipt,
            split,
            out_ref: self.config.out_ref,
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
        }
    }

    fn create_decoder_block<B: Backend>(
        &self,
        in_channels: usize,
        out_channels: usize,
        device: &Device<B>,
    ) -> DecoderBlockModuleEnum<B> {
        match self.config.dec_blk {
            DecBlk::BasicDecBlk => DecoderBlockModuleEnum::BasicDecBlk(
                BasicDecBlkConfig::new(SqueezeBlock::ASPPDeformable(0))
                    .with_in_channels(in_channels)
                    .with_out_channels(out_channels)
                    .init(device),
            ),
            DecBlk::ResBlk => DecoderBlockModuleEnum::ResBlk(
                ResBlkConfig::new()
                    .with_in_channels(in_channels)
                    .with_out_channels(Some(out_channels))
                    .init(device),
            ),
        }
    }

    fn create_lateral_block<B: Backend>(
        &self,
        in_channels: usize,
        out_channels: usize,
        device: &Device<B>,
    ) -> LateralBlockModuleEnum<B> {
        match self.config.lat_blk {
            LatBlk::BasicLatBlk => LateralBlockModuleEnum::BasicLatBlk(
                BasicLatBlkConfig::new()
                    .with_in_channels(in_channels)
                    .with_out_channels(out_channels)
                    .init(device),
            ),
        }
    }
}

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    dec_ipt: bool,
    split: bool,
    out_ref: bool,
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
    gdt_convs_4: Option<Sequential<B>>,
    gdt_convs_3: Option<Sequential<B>>,
    gdt_convs_2: Option<Sequential<B>>,
    gdt_convs_pred_4: Option<Conv2d<B>>,
    gdt_convs_pred_3: Option<Conv2d<B>>,
    gdt_convs_pred_2: Option<Conv2d<B>>,
    gdt_convs_attn_4: Option<Conv2d<B>>,
    gdt_convs_attn_3: Option<Conv2d<B>>,
    gdt_convs_attn_2: Option<Conv2d<B>>,
}

impl<B: Backend> Decoder<B> {
    pub fn forward(&self, features: [Tensor<B, 4>; 5]) -> Tensor<B, 4> {
        let [x, x1, x2, x3, x4] = features;

        let mut x4 = x4;
        if self.dec_ipt {
            let patches_batch = if self.split {
                self.get_patches_batch(x.clone(), x4.clone())
            } else {
                x.clone()
            };
            let [_, _, h, w] = x4.dims();
            x4 = Tensor::cat(
                Vec::from([
                    x4,
                    self.ipt_blk5.as_ref().unwrap().forward(interpolate(
                        patches_batch,
                        [h, w],
                        InterpolateOptions::new(InterpolateMode::Bilinear),
                    )),
                ]),
                1,
            );
        }

        // Decoder block 4
        let p4 = match &self.decoder_block4 {
            DecoderBlockModuleEnum::BasicDecBlk(decoder_block4) => decoder_block4.forward(x4),
            DecoderBlockModuleEnum::ResBlk(decoder_block4) => decoder_block4.forward(x4),
        };
        // let m4 = None;
        let mut p4 = p4;
        if self.out_ref {
            let p4_gdt = self.gdt_convs_4.as_ref().unwrap().forward(p4.clone());
            let gdt_attn_4 = sigmoid(self.gdt_convs_attn_4.as_ref().unwrap().forward(p4_gdt));
            p4 = p4 * gdt_attn_4;
        };
        let [_, _, h, w] = x3.dims();
        let _p4 = interpolate(
            p4,
            [h, w],
            InterpolateOptions::new(InterpolateMode::Bilinear),
        );
        let _p3 = _p4
            + match &self.lateral_block4 {
                LateralBlockModuleEnum::BasicLatBlk(lateral_block4) => {
                    lateral_block4.forward(x3.clone())
                }
            };

        let mut _p3 = _p3;
        if self.dec_ipt {
            let patches_batch = if self.split {
                self.get_patches_batch(x.clone(), _p3.clone())
            } else {
                x.clone()
            };
            let [_, _, h, w] = x3.dims();
            _p3 = Tensor::cat(
                Vec::from([
                    _p3,
                    self.ipt_blk4.as_ref().unwrap().forward(interpolate(
                        patches_batch,
                        [h, w],
                        InterpolateOptions::new(InterpolateMode::Bilinear),
                    )),
                ]),
                1,
            );
        };
        let p3 = match &self.decoder_block3 {
            DecoderBlockModuleEnum::BasicDecBlk(decoder_block3) => decoder_block3.forward(_p3),
            DecoderBlockModuleEnum::ResBlk(decoder_block3) => decoder_block3.forward(_p3),
        };
        // let m3 = None;
        let mut p3 = p3;
        if self.out_ref {
            let p3_gdt = self.gdt_convs_3.as_ref().unwrap().forward(p3.clone());
            let gdt_attn_3 = sigmoid(self.gdt_convs_attn_3.as_ref().unwrap().forward(p3_gdt));
            p3 = p3 * gdt_attn_3;
        };
        let [_, _, h, w] = x2.dims();
        let _p3 = interpolate(
            p3,
            [h, w],
            InterpolateOptions::new(InterpolateMode::Bilinear),
        );
        let _p2 = _p3
            + match &self.lateral_block3 {
                LateralBlockModuleEnum::BasicLatBlk(lateral_block3) => {
                    lateral_block3.forward(x2.clone())
                }
            };

        let mut _p2 = _p2;
        if self.dec_ipt {
            let patches_batch = if self.split {
                self.get_patches_batch(x.clone(), _p2.clone())
            } else {
                x.clone()
            };
            let [_, _, h, w] = x2.dims();
            _p2 = Tensor::cat(
                Vec::from([
                    _p2,
                    self.ipt_blk3.as_ref().unwrap().forward(interpolate(
                        patches_batch,
                        [h, w],
                        InterpolateOptions::new(InterpolateMode::Bilinear),
                    )),
                ]),
                1,
            );
        };
        let p2 = match &self.decoder_block2 {
            DecoderBlockModuleEnum::BasicDecBlk(decoder_block2) => decoder_block2.forward(_p2),
            DecoderBlockModuleEnum::ResBlk(decoder_block2) => decoder_block2.forward(_p2),
        };
        // let m2 = None;
        let mut p2 = p2;
        if self.out_ref {
            let p2_gdt = self.gdt_convs_2.as_ref().unwrap().forward(p2.clone());
            let gdt_attn_2 = sigmoid(self.gdt_convs_attn_2.as_ref().unwrap().forward(p2_gdt));
            p2 = p2 * gdt_attn_2;
        };
        let [_, _, h, w] = x1.dims();
        let _p2 = interpolate(
            p2,
            [h, w],
            InterpolateOptions::new(InterpolateMode::Bilinear),
        );
        let _p1 = _p2
            + match &self.lateral_block2 {
                LateralBlockModuleEnum::BasicLatBlk(lateral_block2) => {
                    lateral_block2.forward(x1.clone())
                }
            };

        let mut _p1 = _p1;
        if self.dec_ipt {
            let patches_batch = if self.split {
                self.get_patches_batch(x.clone(), _p1.clone())
            } else {
                x.clone()
            };
            let [_, _, h, w] = x1.dims();
            _p1 = Tensor::cat(
                Vec::from([
                    _p1,
                    self.ipt_blk2.as_ref().unwrap().forward(interpolate(
                        patches_batch,
                        [h, w],
                        InterpolateOptions::new(InterpolateMode::Bilinear),
                    )),
                ]),
                1,
            );
        };
        let _p1 = match &self.decoder_block1 {
            DecoderBlockModuleEnum::BasicDecBlk(decoder_block1) => decoder_block1.forward(_p1),
            DecoderBlockModuleEnum::ResBlk(decoder_block1) => decoder_block1.forward(_p1),
        };
        let [_, _, h, w] = x.dims();
        let _p1 = interpolate(
            _p1,
            [h, w],
            InterpolateOptions::new(InterpolateMode::Bilinear),
        );

        // let m1 = None;
        let mut _p1 = _p1;
        if self.dec_ipt {
            let patches_batch = if self.split {
                self.get_patches_batch(x.clone(), _p1.clone())
            } else {
                x.clone()
            };
            let [_, _, h, w] = x.dims();
            _p1 = Tensor::cat(
                Vec::from([
                    _p1,
                    self.ipt_blk1.as_ref().unwrap().forward(interpolate(
                        patches_batch,
                        [h, w],
                        InterpolateOptions::new(InterpolateMode::Bilinear),
                    )),
                ]),
                1,
            );
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

#[derive(Config, Debug)]
pub struct SimpleConvsConfig {
    in_channels: usize,
    out_channels: usize,
    #[config(default = "64")]
    inter_channels: usize,
}

impl SimpleConvsConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> SimpleConvs<B> {
        let conv1 = Conv2dConfig::new([self.in_channels, self.inter_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let conv_out = Conv2dConfig::new([self.inter_channels, self.out_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        SimpleConvs { conv1, conv_out }
    }
}

#[derive(Module, Debug)]
pub struct SimpleConvs<B: Backend> {
    conv1: Conv2d<B>,
    conv_out: Conv2d<B>,
}

impl<B: Backend> SimpleConvs<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(x);

        self.conv_out.forward(x)
    }
}
