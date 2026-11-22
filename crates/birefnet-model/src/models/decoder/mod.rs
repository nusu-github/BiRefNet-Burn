//! Hierarchical multi-scale decoder for `BiRefNet`.

use burn::{
    nn::{
        BatchNorm, BatchNormConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::activation::sigmoid,
};

use super::modules::{
    BasicDecBlk, BasicDecBlkConfig, BasicLatBlk, BasicLatBlkConfig, ResBlk, ResBlkConfig,
    SimpleConvs, SimpleConvsConfig,
};
use crate::{
    config::{
        DecBlk, DecoderAttention, InterpolationStrategy, LateralBlock, ModelConfig, SqueezeBlock,
    },
    error::ModelError,
    models::modules::utils::intelligent_interpolate,
};

/// Wrapper enum over supported decoder block implementations.
#[derive(Module, Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "Burn Module requires inline storage without Box indirection"
)]
pub enum DecoderBlockModule {
    /// Basic decoder block.
    BasicDecBlk(BasicDecBlk),
    /// Residual decoder block.
    ResBlk(ResBlk),
}

/// Wrapper enum over supported lateral connection block implementations.
#[derive(Module, Debug)]
pub enum LateralBlockModule {
    /// Basic `1x1` lateral projection block.
    BasicLatBlk(BasicLatBlk),
}

/// Configuration for [`Decoder`].
#[derive(Config, Debug)]
pub struct DecoderConfig {
    /// Main model configuration.
    config: ModelConfig,
    /// Deep-to-shallow channel counts `[c4, c3, c2, c1]` of encoder features.
    channels: [usize; 4],
    /// Interpolation strategy for tensor resizing operations.
    interpolation_strategy: InterpolationStrategy,
}

#[derive(Module, Debug)]
struct GdtConvs {
    conv: Conv2d,
    bn: BatchNorm,
    relu: Relu,
}

impl GdtConvs {
    fn init(
        conv2d_config: &Conv2dConfig,
        batch_norm_config: &BatchNormConfig,
        device: &Device,
    ) -> Self {
        let conv = conv2d_config.init(device);
        let bn = batch_norm_config.init(device);
        let relu = Relu::new();
        Self { conv, bn, relu }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.conv.forward(x);
        let x = self.bn.forward(x);
        self.relu.forward(x)
    }
}

/// Multi-scale supervision and GDT layers:
/// `(conv_ms_spvn, gdt_convs, gdt_convs_pred, gdt_convs_attn)`.
type SupervisionLayers = (
    [Option<Conv2d>; 3],
    [Option<GdtConvs>; 3],
    [Option<Conv2d>; 3],
    [Option<Conv2d>; 3],
);

impl DecoderConfig {
    const IC: usize = 64;
    const IPT_CHA_OPT: usize = 1;
    const N_DEC_IPT: usize = 64;
    const GDT_CHANNELS: usize = 16;

    fn create_ipt_blocks(&self, split: bool, device: &Device) -> [Option<SimpleConvs>; 5] {
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

    fn create_supervision_layers(&self, device: &Device) -> SupervisionLayers {
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
                    &Conv2dConfig::new([channel, Self::GDT_CHANNELS], [3, 3])
                        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1)),
                    &BatchNormConfig::new(Self::GDT_CHANNELS),
                    device,
                ));

                gdt_convs_pred[i] = Some(
                    Conv2dConfig::new([Self::GDT_CHANNELS, 1], [1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .init(device),
                );

                gdt_convs_attn[i] = Some(
                    Conv2dConfig::new([Self::GDT_CHANNELS, 1], [1, 1])
                        .with_padding(PaddingConfig2d::Valid)
                        .init(device),
                );
            }
        }

        (conv_ms_spvn, gdt_convs, gdt_convs_pred, gdt_convs_attn)
    }

    fn ipt_extra_channels(&self, channel_idx: usize) -> usize {
        if !self.config.decoder.dec_ipt {
            0
        } else if Self::IPT_CHA_OPT == 0 {
            Self::N_DEC_IPT
        } else {
            self.channels[channel_idx] / 8
        }
    }

    fn create_decoder_blocks(
        &self,
        device: &Device,
    ) -> Result<[DecoderBlockModule; 4], ModelError> {
        let channel_configs = [
            (0, 0, self.channels[1]),     // decoder_block4
            (1, 0, self.channels[2]),     // decoder_block3
            (2, 1, self.channels[3]),     // decoder_block2
            (3, 2, self.channels[3] / 2), // decoder_block1
        ];

        let mut blocks = Vec::with_capacity(4);
        for &(x, y, out_channels) in &channel_configs {
            let in_channels = self.channels[x] + self.ipt_extra_channels(y);
            blocks.push(self.create_decoder_block(in_channels, out_channels, device)?);
        }

        blocks
            .try_into()
            .map_err(|_err| ModelError::invalid_config("failed to assemble 4 decoder blocks"))
    }

    fn create_lateral_blocks(&self, device: &Device) -> [LateralBlockModule; 3] {
        [
            self.create_lateral_block(self.channels[1], self.channels[1], device),
            self.create_lateral_block(self.channels[2], self.channels[2], device),
            self.create_lateral_block(self.channels[3], self.channels[3], device),
        ]
    }

    /// Initializes a [`Decoder`] module.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if any decoder block configuration is invalid.
    pub fn init(&self, device: &Device) -> Result<Decoder, ModelError> {
        let split = self.config.decoder.dec_ipt && self.config.decoder.dec_ipt_split;

        let [ipt_blk5, ipt_blk4, ipt_blk3, ipt_blk2, ipt_blk1] =
            self.create_ipt_blocks(split, device);

        let [
            decoder_block4,
            decoder_block3,
            decoder_block2,
            decoder_block1,
        ] = self.create_decoder_blocks(device)?;

        let [lateral_block4, lateral_block3, lateral_block2] = self.create_lateral_blocks(device);

        let (
            [conv_ms_spvn_4, conv_ms_spvn_3, conv_ms_spvn_2],
            [gdt_convs_4, gdt_convs_3, gdt_convs_2],
            [gdt_convs_pred_4, gdt_convs_pred_3, gdt_convs_pred_2],
            [gdt_convs_attn_4, gdt_convs_attn_3, gdt_convs_attn_2],
        ) = self.create_supervision_layers(device);

        let in_channels = (self.channels[3] / 2) + self.ipt_extra_channels(3);
        let conv_out1 = Conv2dConfig::new([in_channels, 1], [1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .init(device);

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
            interpolation_strategy: self.interpolation_strategy,
        })
    }

    fn create_decoder_block(
        &self,
        in_channels: usize,
        out_channels: usize,
        device: &Device,
    ) -> Result<DecoderBlockModule, ModelError> {
        let dec_att = self.config.decoder.dec_att;
        let dec_channels_inter = self.config.decoder.dec_channels_inter;
        let batch_size = self.config.task.batch_size;

        match self.config.decoder.dec_blk {
            DecBlk::BasicDecBlk => {
                let squeeze_block = match dec_att {
                    DecoderAttention::None => SqueezeBlock::None,
                    DecoderAttention::ASPP => SqueezeBlock::ASPP(0),
                    DecoderAttention::ASPPDeformable => SqueezeBlock::ASPPDeformable(0),
                };
                Ok(DecoderBlockModule::BasicDecBlk(
                    BasicDecBlkConfig::new(squeeze_block, self.interpolation_strategy)
                        .with_in_channels(in_channels)
                        .with_out_channels(out_channels)
                        .with_dec_channels_inter(dec_channels_inter)
                        .with_batch_size(batch_size)
                        .init(device)?,
                ))
            }
            DecBlk::ResBlk => Ok(DecoderBlockModule::ResBlk(
                ResBlkConfig::new(self.interpolation_strategy)
                    .with_in_channels(in_channels)
                    .with_out_channels(Some(out_channels))
                    .with_dec_att(dec_att)
                    .with_dec_channels_inter(dec_channels_inter)
                    .with_batch_size(batch_size)
                    .init(device)?,
            )),
        }
    }

    fn create_lateral_block(
        &self,
        in_channels: usize,
        out_channels: usize,
        device: &Device,
    ) -> LateralBlockModule {
        match self.config.decoder.lat_blk {
            LateralBlock::BasicLatBlk => LateralBlockModule::BasicLatBlk(
                BasicLatBlkConfig::new()
                    .with_in_channels(in_channels)
                    .with_out_channels(out_channels)
                    .init(device),
            ),
        }
    }
}

/// Multi-stage hierarchical decoder module for `BiRefNet`.
#[derive(Module, Debug)]
#[expect(
    clippy::struct_field_names,
    reason = "field names match BiRefNet PyTorch state_dict keys for checkpoint compatibility"
)]
pub struct Decoder {
    split: bool,
    ipt_blk5: Option<SimpleConvs>,
    ipt_blk4: Option<SimpleConvs>,
    ipt_blk3: Option<SimpleConvs>,
    ipt_blk2: Option<SimpleConvs>,
    ipt_blk1: Option<SimpleConvs>,
    decoder_block4: DecoderBlockModule,
    decoder_block3: DecoderBlockModule,
    decoder_block2: DecoderBlockModule,
    decoder_block1: DecoderBlockModule,
    lateral_block4: LateralBlockModule,
    lateral_block3: LateralBlockModule,
    lateral_block2: LateralBlockModule,
    conv_out1: Conv2d,
    conv_ms_spvn_4: Option<Conv2d>,
    conv_ms_spvn_3: Option<Conv2d>,
    conv_ms_spvn_2: Option<Conv2d>,
    gdt_convs_4: Option<GdtConvs>,
    gdt_convs_3: Option<GdtConvs>,
    gdt_convs_2: Option<GdtConvs>,
    gdt_convs_pred_4: Option<Conv2d>,
    gdt_convs_pred_3: Option<Conv2d>,
    gdt_convs_pred_2: Option<Conv2d>,
    gdt_convs_attn_4: Option<Conv2d>,
    gdt_convs_attn_3: Option<Conv2d>,
    gdt_convs_attn_2: Option<Conv2d>,
    #[module(skip)]
    interpolation_strategy: InterpolationStrategy,
}

impl Decoder {
    fn process_ipt_block(
        &self,
        ipt_blk: Option<&SimpleConvs>,
        x: &Tensor<4>,
        feature: Tensor<4>,
    ) -> Tensor<4> {
        let Some(blk) = ipt_blk else {
            return feature;
        };

        let [_, _, h, w] = feature.dims();
        let patches_batch = if self.split {
            Self::get_patches_batch(x, &feature)
        } else {
            x.clone()
        };

        Tensor::cat(
            vec![
                feature,
                blk.forward(intelligent_interpolate(
                    patches_batch,
                    [h, w],
                    self.interpolation_strategy,
                )),
            ],
            1,
        )
    }

    fn process_decoder_block(block: &DecoderBlockModule, input: Tensor<4>) -> Tensor<4> {
        match block {
            DecoderBlockModule::BasicDecBlk(blk) => blk.forward(input),
            DecoderBlockModule::ResBlk(blk) => blk.forward(input),
        }
    }

    fn apply_gdt_attention(
        input: Tensor<4>,
        gdt_conv: Option<&GdtConvs>,
        gdt_attn: Option<&Conv2d>,
    ) -> Tensor<4> {
        match (gdt_conv, gdt_attn) {
            (Some(conv), Some(attn)) => {
                let gdt = conv.forward(input.clone());
                let attention = sigmoid(attn.forward(gdt));
                input * attention
            }
            _ => input,
        }
    }

    fn process_lateral_block(block: &LateralBlockModule, input: Tensor<4>) -> Tensor<4> {
        match block {
            LateralBlockModule::BasicLatBlk(blk) => blk.forward(input),
        }
    }

    fn forward_core(&self, features: [Tensor<4>; 5], collect_ms: bool) -> Vec<Tensor<4>> {
        let [x, x1, x2, x3, x4] = features;
        let mut outs = Vec::with_capacity(4);

        // Stage 4
        let x4_processed = self.process_ipt_block(self.ipt_blk5.as_ref(), &x, x4);
        let p4 = Self::process_decoder_block(&self.decoder_block4, x4_processed);
        let p4 = Self::apply_gdt_attention(
            p4,
            self.gdt_convs_4.as_ref(),
            self.gdt_convs_attn_4.as_ref(),
        );
        if collect_ms && let Some(conv) = &self.conv_ms_spvn_4 {
            outs.push(conv.forward(p4.clone()));
        }

        let [_, _, h3, w3] = x3.dims();
        let p4_interp = intelligent_interpolate(p4, [h3, w3], self.interpolation_strategy);
        let p3_lateral = p4_interp + Self::process_lateral_block(&self.lateral_block4, x3);

        // Stage 3
        let p3_with_ipt = self.process_ipt_block(self.ipt_blk4.as_ref(), &x, p3_lateral);
        let p3 = Self::process_decoder_block(&self.decoder_block3, p3_with_ipt);
        let p3 = Self::apply_gdt_attention(
            p3,
            self.gdt_convs_3.as_ref(),
            self.gdt_convs_attn_3.as_ref(),
        );
        if collect_ms && let Some(conv) = &self.conv_ms_spvn_3 {
            outs.push(conv.forward(p3.clone()));
        }

        let [_, _, h2, w2] = x2.dims();
        let p3_interp = intelligent_interpolate(p3, [h2, w2], self.interpolation_strategy);
        let p2_lateral = p3_interp + Self::process_lateral_block(&self.lateral_block3, x2);

        // Stage 2
        let p2_with_ipt = self.process_ipt_block(self.ipt_blk3.as_ref(), &x, p2_lateral);
        let p2 = Self::process_decoder_block(&self.decoder_block2, p2_with_ipt);
        let p2 = Self::apply_gdt_attention(
            p2,
            self.gdt_convs_2.as_ref(),
            self.gdt_convs_attn_2.as_ref(),
        );
        if collect_ms && let Some(conv) = &self.conv_ms_spvn_2 {
            outs.push(conv.forward(p2.clone()));
        }

        let [_, _, h1, w1] = x1.dims();
        let p2_interp = intelligent_interpolate(p2, [h1, w1], self.interpolation_strategy);
        let p1_lateral = p2_interp + Self::process_lateral_block(&self.lateral_block2, x1);

        // Stage 1
        let p1_with_ipt = self.process_ipt_block(self.ipt_blk2.as_ref(), &x, p1_lateral);
        let p1 = Self::process_decoder_block(&self.decoder_block1, p1_with_ipt);

        let [_, _, h_orig, w_orig] = x.dims();
        let p1 = intelligent_interpolate(p1, [h_orig, w_orig], self.interpolation_strategy);
        let p1_final = self.process_ipt_block(self.ipt_blk1.as_ref(), &x, p1);
        outs.push(self.conv_out1.forward(p1_final));

        outs
    }

    /// Runs the decoder and returns only the final segmentation logit map `[B, 1, H, W]`.
    #[must_use]
    pub fn forward(&self, features: [Tensor<4>; 5]) -> Tensor<4> {
        self.forward_core(features, false)
            .pop()
            .expect("decoder always emits final prediction")
    }

    /// Runs the decoder and returns multi-scale supervision logits followed by the final prediction.
    #[must_use]
    pub fn forward_multi_scale(&self, features: [Tensor<4>; 5]) -> Vec<Tensor<4>> {
        self.forward_core(features, true)
    }

    fn get_patches_batch(image: &Tensor<4>, feature: &Tensor<4>) -> Tensor<4> {
        let [batch_size, _, patch_h, patch_w] = feature.dims();
        let [_, img_c, img_h, img_w] = image.dims();
        let cols = img_w / patch_w;
        let rows = img_h / patch_h;
        let mut patches = Vec::with_capacity(cols * rows);
        for col in 0..cols {
            let col_start = col * patch_w;
            let column = image.clone().slice(s![
                0..batch_size,
                0..img_c,
                0..img_h,
                col_start..col_start + patch_w
            ]);
            for row in 0..rows {
                let row_start = row * patch_h;
                let patch = column.clone().slice([
                    0..batch_size,
                    0..img_c,
                    row_start..row_start + patch_h,
                    0..patch_w,
                ]);
                patches.push(patch);
            }
        }
        Tensor::cat(patches, 1)
    }
}
