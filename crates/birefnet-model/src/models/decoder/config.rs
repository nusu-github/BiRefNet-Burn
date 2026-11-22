//! Construction of [`Decoder`] from a [`ModelConfig`].

use burn::{
    nn::{
        BatchNormConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
};

use super::{Decoder, DecoderBlockModule, GdtConvs, LateralBlockModule};
use crate::{
    config::{DecBlk, InterpolationStrategy, LateralBlock, ModelConfig},
    error::ModelError,
    models::modules::{
        BasicDecBlkConfig, BasicLatBlkConfig, ResBlkConfig, SimpleConvs, SimpleConvsConfig,
    },
};

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

/// Multi-scale supervision and GDT layers:
/// `(conv_ms_spvn, gdt_convs, gdt_convs_pred, gdt_convs_attn)`.
type SupervisionLayers = (
    [Option<Conv2d>; 3],
    [Option<GdtConvs>; 3],
    [Option<Conv2d>; 3],
    [Option<Conv2d>; 3],
);

/// Hidden width of the image-pyramid `SimpleConvs` blocks (upstream `ic = 64`).
const IPT_HIDDEN: usize = 64;
/// Upstream `ipt_cha_opt = 1`: image-pyramid branches emit `channels / 8` features.
const IPT_CHANNEL_DIVISOR: usize = 8;
/// Channel width of the gradient-guided (GDT) attention branches.
const GDT_CHANNELS: usize = 16;
/// `log2` of the patch count fed to `ipt_blk5..ipt_blk1` when `dec_ipt_split` is enabled:
/// `32x32`, `16x16`, `8x8`, `4x4`, and `1x1` grids of RGB tiles.
const IPT_SPLIT_POWERS: [u32; 5] = [10, 8, 6, 4, 0];
/// Index into `channels` that sizes each image-pyramid branch (`ipt_blk5..ipt_blk1`).
const IPT_CHANNEL_INDEX: [usize; 5] = [0, 0, 1, 2, 3];

impl DecoderConfig {
    fn create_ipt_blocks(&self, split: bool, device: &Device) -> [Option<SimpleConvs>; 5] {
        if !self.config.decoder.dec_ipt {
            return [None, None, None, None, None];
        }
        [0, 1, 2, 3, 4].map(|stage| {
            let in_channels = if split {
                2_usize.pow(IPT_SPLIT_POWERS[stage]) * 3
            } else {
                3
            };
            let out_channels = self.channels[IPT_CHANNEL_INDEX[stage]] / IPT_CHANNEL_DIVISOR;
            Some(
                SimpleConvsConfig::new(in_channels, out_channels)
                    .with_inter_channels(IPT_HIDDEN)
                    .init(device),
            )
        })
    }

    fn create_supervision_layers(&self, device: &Device) -> SupervisionLayers {
        if !self.config.decoder.ms_supervision {
            return (
                [None, None, None],
                [None, None, None],
                [None, None, None],
                [None, None, None],
            );
        }

        let side_channels = [self.channels[1], self.channels[2], self.channels[3]];
        let one_by_one = |in_channels: usize| {
            Some(
                Conv2dConfig::new([in_channels, 1], [1, 1])
                    .with_padding(PaddingConfig2d::Valid)
                    .init(device),
            )
        };
        let out_ref = self.config.decoder.out_ref;

        let conv_ms_spvn = side_channels.map(one_by_one);
        let gdt_convs = side_channels.map(|channels| {
            out_ref.then(|| {
                GdtConvs::init(
                    &Conv2dConfig::new([channels, GDT_CHANNELS], [3, 3])
                        .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1)),
                    &BatchNormConfig::new(GDT_CHANNELS),
                    device,
                )
            })
        });
        let gdt_convs_pred = side_channels.map(|_| {
            if out_ref {
                one_by_one(GDT_CHANNELS)
            } else {
                None
            }
        });
        let gdt_convs_attn = side_channels.map(|_| {
            if out_ref {
                one_by_one(GDT_CHANNELS)
            } else {
                None
            }
        });

        (conv_ms_spvn, gdt_convs, gdt_convs_pred, gdt_convs_attn)
    }

    fn ipt_extra_channels(&self, channel_idx: usize) -> usize {
        if self.config.decoder.dec_ipt {
            self.channels[channel_idx] / IPT_CHANNEL_DIVISOR
        } else {
            0
        }
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

        // (input feature index, image-pyramid channel index, output width) per stage.
        let decoder_block4 =
            self.create_decoder_block(0, IPT_CHANNEL_INDEX[0], self.channels[1], device)?;
        let decoder_block3 =
            self.create_decoder_block(1, IPT_CHANNEL_INDEX[1], self.channels[2], device)?;
        let decoder_block2 =
            self.create_decoder_block(2, IPT_CHANNEL_INDEX[2], self.channels[3], device)?;
        let decoder_block1 =
            self.create_decoder_block(3, IPT_CHANNEL_INDEX[3], self.channels[3] / 2, device)?;

        let lateral_block4 = self.create_lateral_block(self.channels[1], device);
        let lateral_block3 = self.create_lateral_block(self.channels[2], device);
        let lateral_block2 = self.create_lateral_block(self.channels[3], device);

        let (
            [conv_ms_spvn_4, conv_ms_spvn_3, conv_ms_spvn_2],
            [gdt_convs_4, gdt_convs_3, gdt_convs_2],
            [gdt_convs_pred_4, gdt_convs_pred_3, gdt_convs_pred_2],
            [gdt_convs_attn_4, gdt_convs_attn_3, gdt_convs_attn_2],
        ) = self.create_supervision_layers(device);

        let in_channels = (self.channels[3] / 2) + self.ipt_extra_channels(IPT_CHANNEL_INDEX[4]);
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
        feature_idx: usize,
        ipt_idx: usize,
        out_channels: usize,
        device: &Device,
    ) -> Result<DecoderBlockModule, ModelError> {
        let in_channels = self.channels[feature_idx] + self.ipt_extra_channels(ipt_idx);
        let decoder = &self.config.decoder;
        let batch_size = self.config.task.batch_size;

        Ok(match decoder.dec_blk {
            DecBlk::BasicDecBlk => DecoderBlockModule::BasicDecBlk(
                BasicDecBlkConfig::new(decoder.dec_att, self.interpolation_strategy)
                    .with_in_channels(in_channels)
                    .with_out_channels(out_channels)
                    .with_dec_channels_inter(decoder.dec_channels_inter)
                    .with_batch_size(batch_size)
                    .init(device)?,
            ),
            DecBlk::ResBlk => DecoderBlockModule::ResBlk(
                ResBlkConfig::new(self.interpolation_strategy)
                    .with_in_channels(in_channels)
                    .with_out_channels(Some(out_channels))
                    .with_dec_att(decoder.dec_att)
                    .with_dec_channels_inter(decoder.dec_channels_inter)
                    .with_batch_size(batch_size)
                    .init(device)?,
            ),
        })
    }

    fn create_lateral_block(&self, channels: usize, device: &Device) -> LateralBlockModule {
        match self.config.decoder.lat_blk {
            LateralBlock::BasicLatBlk => LateralBlockModule::BasicLatBlk(
                BasicLatBlkConfig::new()
                    .with_in_channels(channels)
                    .with_out_channels(channels)
                    .init(device),
            ),
        }
    }
}
