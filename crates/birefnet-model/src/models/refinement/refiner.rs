//! Backbone-based refinement modules (`Refiner` and `RefinerPVTInChannels4`) for `BiRefNet`.

use birefnet_backbones::{Backbone as _, BackboneModel, PvtV2Config, PvtV2Variant};
use burn::{
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::activation,
};

use crate::{
    ModelConfig,
    config::InterpolationStrategy,
    error::ModelError,
    models::modules::{
        BasicDecBlk, BasicDecBlkConfig, BasicLatBlk, BasicLatBlkConfig,
        utils::{ActLayerEnum, NormKind, NormLayerEnum, build_norm_layer, intelligent_interpolate},
    },
};

/// Configuration for [`RefinerDecoder`].
#[derive(Config, Debug)]
pub struct RefinerDecoderConfig {
    /// Model configuration providing attention, intermediate width, and batch size settings.
    config: ModelConfig,
    /// Deep-to-shallow channel widths `[c4, c3, c2, c1]`.
    channels: [usize; 4],
    /// Whether to emit multi-scale supervision outputs before the final map.
    #[config(default = false)]
    ms_supervision: bool,
    /// Interpolation strategy for tensor resizing operations.
    interpolation_strategy: InterpolationStrategy,
}

/// Decoder used by [`Refiner`] and [`RefinerPVTInChannels4`].
#[derive(Module, Debug)]
pub struct RefinerDecoder {
    decoder_block4: BasicDecBlk,
    decoder_block3: BasicDecBlk,
    decoder_block2: BasicDecBlk,
    decoder_block1: BasicDecBlk,
    lateral_block4: BasicLatBlk,
    lateral_block3: BasicLatBlk,
    lateral_block2: BasicLatBlk,
    conv_out1: Conv2d,
    conv_ms_spvn_4: Option<Conv2d>,
    conv_ms_spvn_3: Option<Conv2d>,
    conv_ms_spvn_2: Option<Conv2d>,
    ms_supervision: bool,
    #[module(skip)]
    interpolation_strategy: InterpolationStrategy,
}

impl RefinerDecoderConfig {
    fn make_decoder_block(
        &self,
        in_channels: usize,
        out_channels: usize,
        device: &Device,
    ) -> Result<BasicDecBlk, ModelError> {
        BasicDecBlkConfig::new(self.config.decoder.dec_att, self.config.interpolation)
            .with_in_channels(in_channels)
            .with_out_channels(out_channels)
            .with_dec_channels_inter(self.config.decoder.dec_channels_inter)
            .with_batch_size(self.config.task.batch_size)
            .init(device)
    }

    /// Initializes a [`RefinerDecoder`] module.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if any decoder block fails to initialize.
    pub fn init(&self, device: &Device) -> Result<RefinerDecoder, ModelError> {
        let decoder_block4 = self.make_decoder_block(self.channels[0], self.channels[1], device)?;
        let decoder_block3 = self.make_decoder_block(self.channels[1], self.channels[2], device)?;
        let decoder_block2 = self.make_decoder_block(self.channels[2], self.channels[3], device)?;
        let decoder_block1 =
            self.make_decoder_block(self.channels[3], self.channels[3] / 2, device)?;

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
            interpolation_strategy: self.interpolation_strategy,
        })
    }
}

impl RefinerDecoder {
    /// Runs the refinement decoder over `[input, x1, x2, x3, x4]`.
    #[must_use]
    pub fn forward(&self, features: [Tensor<4>; 5]) -> Vec<Tensor<4>> {
        let [x, x1, x2, x3, x4] = features;
        let mut outs = Vec::with_capacity(4);
        let resize_like = |tensor: Tensor<4>, like: &Tensor<4>| {
            let [_, _, h, w] = like.dims();
            intelligent_interpolate(tensor, [h, w], self.interpolation_strategy)
        };
        // Side outputs exist only with `ms_supervision`, so features are cloned only then.
        let mut side_output = |conv: Option<&Conv2d>, p: &Tensor<4>| {
            if let Some(conv) = conv {
                outs.push(conv.forward(p.clone()));
            }
        };

        let p4 = self.decoder_block4.forward(x4);
        side_output(self.conv_ms_spvn_4.as_ref(), &p4);
        let d3_in = resize_like(p4, &x3) + self.lateral_block4.forward(x3);

        let p3 = self.decoder_block3.forward(d3_in);
        side_output(self.conv_ms_spvn_3.as_ref(), &p3);
        let d2_in = resize_like(p3, &x2) + self.lateral_block3.forward(x2);

        let p2 = self.decoder_block2.forward(d2_in);
        side_output(self.conv_ms_spvn_2.as_ref(), &p2);
        let d1_in = resize_like(p2, &x1) + self.lateral_block2.forward(x1);

        let d1 = self.decoder_block1.forward(d1_in);
        outs.push(self.conv_out1.forward(resize_like(d1, &x)));
        outs
    }
}

/// Convolutional stem layer converting 4-channel `(RGB + mask)` inputs for refinement backbones.
#[derive(Module, Debug)]
pub struct StemLayer {
    conv1: Conv2d,
    norm1: Vec<NormLayerEnum>,
    #[module(skip)]
    act: ActLayerEnum,
    conv2: Conv2d,
    norm2: Vec<NormLayerEnum>,
}

impl StemLayer {
    /// Creates a new NCHW [`StemLayer`].
    #[must_use]
    pub fn new(
        in_channels: usize,
        inter_channels: usize,
        out_channels: usize,
        norm_kind: NormKind,
        device: &Device,
    ) -> Self {
        let conv1 = Conv2dConfig::new([in_channels, inter_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);

        // Convolutions produce NCHW tensors, so keep NCHW layout through stem normalization.
        let norm1 = build_norm_layer(inter_channels, norm_kind, false, false, 1e-5, device);
        let act = ActLayerEnum::Gelu;

        let conv2 = Conv2dConfig::new([inter_channels, out_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);

        let norm2 = build_norm_layer(out_channels, norm_kind, false, false, 1e-5, device);

        Self {
            conv1,
            norm1,
            act,
            conv2,
            norm2,
        }
    }

    /// Runs the two-stage stem block over `[B, in_channels, H, W]`.
    #[must_use]
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.conv1.forward(x);
        let x = self.norm1.iter().fold(x, |x, norm| norm.forward(x));
        let x = match self.act {
            ActLayerEnum::Gelu => activation::gelu(x),
        };
        let x = self.conv2.forward(x);
        self.norm2.iter().fold(x, |x, norm| norm.forward(x))
    }
}

/// Configuration for [`RefinerPVTInChannels4`].
#[derive(Config, Debug)]
pub struct RefinerPVTInChannels4Config {
    /// Model configuration.
    config: ModelConfig,
    /// Number of input channels (default: `4`).
    #[config(default = 4)]
    in_channels: usize,
    /// PVTv2 variant used as the refinement backbone.
    #[config(default = "PvtV2Variant::B2")]
    backbone_variant: PvtV2Variant,
    /// Interpolation strategy for tensor resizing operations.
    interpolation_strategy: InterpolationStrategy,
}

/// Refinement module taking 4-channel `(RGB + mask)` inputs using a PVTv2 backbone.
#[derive(Module, Debug)]
pub struct RefinerPVTInChannels4 {
    backbone: BackboneModel,
    squeeze_module: BasicDecBlk,
    decoder: RefinerDecoder,
}

impl RefinerPVTInChannels4Config {
    /// Initializes a [`RefinerPVTInChannels4`] module.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if backbone or decoder initialization fails.
    pub fn init(&self, device: &Device) -> Result<RefinerPVTInChannels4, ModelError> {
        let backbone = BackboneModel::PvtV2(
            PvtV2Config::from_variant(self.backbone_variant, self.in_channels).init(device)?,
        );
        let raw_channels = backbone.output_channels();
        let dec_channels = [
            raw_channels[3],
            raw_channels[2],
            raw_channels[1],
            raw_channels[0],
        ];

        let squeeze_module =
            BasicDecBlkConfig::new(self.config.decoder.dec_att, self.config.interpolation)
                .with_in_channels(raw_channels[3])
                .with_out_channels(raw_channels[3])
                .with_dec_channels_inter(self.config.decoder.dec_channels_inter)
                .with_batch_size(self.config.task.batch_size)
                .init(device)?;

        let decoder = RefinerDecoderConfig::new(
            self.config.clone(),
            dec_channels,
            self.interpolation_strategy,
        )
        .with_ms_supervision(false)
        .init(device)?;

        Ok(RefinerPVTInChannels4 {
            backbone,
            squeeze_module,
            decoder,
        })
    }
}

impl RefinerPVTInChannels4 {
    /// Runs refinement over `[B, in_channels, H, W]` `(RGB + mask)` inputs.
    ///
    /// Every input channel, including the coarse mask, is fed to the PVTv2 encoder.
    #[must_use]
    pub fn forward(&self, x: Tensor<4>) -> Vec<Tensor<4>> {
        let [x1, x2, x3, x4] = self.backbone.forward(x.clone());
        let x4 = self.squeeze_module.forward(x4);

        let features = [x, x1, x2, x3, x4];
        self.decoder.forward(features)
    }
}

/// Configuration for [`Refiner`].
#[derive(Config, Debug)]
pub struct RefinerConfig {
    /// Model configuration.
    config: ModelConfig,
    /// Number of input channels (default: `4` for RGB + mask).
    #[config(default = 4)]
    in_channels: usize,
    /// Intermediate stem channel width.
    #[config(default = 48)]
    inter_channels: usize,
    /// Stem output channel width; the PVTv2 backbone is built with this many input channels.
    #[config(default = 3)]
    out_channels: usize,
    /// PVTv2 variant used as the refinement backbone.
    #[config(default = "PvtV2Variant::B2")]
    pvt_variant: PvtV2Variant,
    /// Interpolation strategy for tensor resizing operations.
    interpolation_strategy: InterpolationStrategy,
}

/// Stem + PVTv2 refinement module for `BiRefNet`.
#[derive(Module, Debug)]
pub struct Refiner {
    stem_layer: StemLayer,
    backbone: BackboneModel,
    squeeze_module: BasicDecBlk,
    decoder: RefinerDecoder,
}

impl RefinerConfig {
    /// Initializes a [`Refiner`] module.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if backbone or decoder initialization fails.
    pub fn init(&self, device: &Device) -> Result<Refiner, ModelError> {
        let stem_layer = StemLayer::new(
            self.in_channels,
            self.inter_channels,
            self.out_channels,
            NormKind::BatchNorm,
            device,
        );

        let backbone = BackboneModel::PvtV2(
            PvtV2Config::from_variant(self.pvt_variant, self.out_channels).init(device)?,
        );
        let raw_channels = backbone.output_channels();
        let dec_channels = [
            raw_channels[3],
            raw_channels[2],
            raw_channels[1],
            raw_channels[0],
        ];

        let squeeze_module =
            BasicDecBlkConfig::new(self.config.decoder.dec_att, self.config.interpolation)
                .with_in_channels(raw_channels[3])
                .with_out_channels(raw_channels[3])
                .with_dec_channels_inter(self.config.decoder.dec_channels_inter)
                .with_batch_size(self.config.task.batch_size)
                .init(device)?;

        let decoder = RefinerDecoderConfig::new(
            self.config.clone(),
            dec_channels,
            self.interpolation_strategy,
        )
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

impl Refiner {
    /// Runs stem projection, backbone encoding, squeeze block, and refinement decoder.
    #[must_use]
    pub fn forward(&self, x: Tensor<4>) -> Vec<Tensor<4>> {
        let x_processed = self.stem_layer.forward(x);
        let [x1, x2, x3, x4] = self.backbone.forward(x_processed.clone());
        let x4 = self.squeeze_module.forward(x4);

        let features = [x_processed, x1, x2, x3, x4];
        self.decoder.forward(features)
    }
}
