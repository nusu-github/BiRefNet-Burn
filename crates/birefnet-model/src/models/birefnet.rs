//! Main `BiRefNet` model uniting backbone encoder, squeeze stage, decoder, and optional refinement.

use birefnet_backbones::{
    BackboneModel, BackboneType, PvtV2Variant, ResNetVariant, SwinVariant, VggVariant,
};
#[cfg(feature = "train")]
use birefnet_loss::{BiRefNetLoss, BiRefNetLossConfig};
use burn::prelude::*;

use super::{
    decoder::{Decoder, DecoderConfig},
    modules::{
        ASPP, ASPPConfig, ASPPDeformable, ASPPDeformableConfig, BasicDecBlk, BasicDecBlkConfig,
        ResBlk, ResBlkConfig, utils::intelligent_interpolate,
    },
    refinement::{
        RefUNet, RefUNetConfig, Refiner, RefinerConfig, RefinerPVTInChannels4,
        RefinerPVTInChannels4Config,
    },
};
use crate::{
    config::{Backbone, InterpolationStrategy, ModelConfig, MultiScaleInput, Refine, SqueezeBlock},
    error::ModelError,
};

/// Squeeze block variant applied between the encoder output and decoder input.
#[derive(Module, Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "ASPP variants own fixed multi-branch parameters and stay non-boxed inside Burn Module derive"
)]
#[expect(
    clippy::upper_case_acronyms,
    reason = "preserves ASPP variant name matching existing BiRefNet schema"
)]
pub enum SqueezeBlockModule {
    /// Basic decoder squeeze block.
    BasicDecBlk(BasicDecBlk),
    /// Residual decoder squeeze block.
    ResBlk(ResBlk),
    /// Atrous Spatial Pyramid Pooling squeeze block.
    ASPP(ASPP),
    /// Deformable ASPP squeeze block.
    ASPPDeformable(ASPPDeformable),
}

impl SqueezeBlockModule {
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        match self {
            Self::BasicDecBlk(model) => model.forward(x),
            Self::ResBlk(model) => model.forward(x),
            Self::ASPP(model) => model.forward(x),
            Self::ASPPDeformable(model) => model.forward(x),
        }
    }
}

/// Optional refinement module applied to coarse decoder predictions.
#[derive(Module, Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "Backbone-backed refiner variants stay inline for Burn Module parameter serialization"
)]
pub enum RefineModule {
    /// Lightweight U-Net refiner.
    RefUNet(RefUNet),
    /// Stem + PVTv2 refiner.
    Refiner(Refiner),
    /// Direct 4-channel PVTv2 refiner.
    RefinerPVTInChannels4(RefinerPVTInChannels4),
}

/// Configuration for [`BiRefNet`].
#[derive(Config, Debug)]
pub struct BiRefNetConfig {
    /// Detailed architecture configuration.
    config: ModelConfig,
    /// Optional training loss configuration when built with the `train` feature.
    #[cfg(feature = "train")]
    loss_config: Option<BiRefNetLossConfig>,
}

impl BiRefNetConfig {
    /// Returns the architecture configuration.
    #[must_use]
    pub const fn model_config(&self) -> &ModelConfig {
        &self.config
    }

    /// Initializes a [`BiRefNet`] model on `device`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if [`ModelConfig::validate`] fails or any submodule
    /// cannot be initialized.
    pub fn init(&self, device: &Device) -> Result<BiRefNet, ModelError> {
        self.config.validate()?;

        let bb = backbone_type(self.config.backbone.backbone).init(device)?;
        let channels = self.config.lateral_channels_in_collection();
        let cxt = self.config.cxt()?;
        let cxt_sum = cxt.iter().sum::<usize>();

        // The first squeeze block also consumes the concatenated context features.
        let squeeze_module = (0..self.config.decoder.squeeze_block.count())
            .map(|idx| {
                let in_channels = if idx == 0 {
                    channels[0] + cxt_sum
                } else {
                    channels[0]
                };
                self.squeeze_block(in_channels, channels[0], device)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let interpolation = self.config.interpolation;
        let refine = match self.config.refine.refine {
            Refine::RefUNet => Some(RefineModule::RefUNet(
                RefUNetConfig::new(interpolation).init(device),
            )),
            Refine::Refiner => Some(RefineModule::Refiner(
                RefinerConfig::new(self.config.clone(), interpolation).init(device)?,
            )),
            Refine::RefinerPVTInChannels4 => Some(RefineModule::RefinerPVTInChannels4(
                RefinerPVTInChannels4Config::new(self.config.clone(), interpolation)
                    .init(device)?,
            )),
            // `validate` rejects `Itself`.
            Refine::Itself | Refine::None => None,
        };

        let decoder =
            DecoderConfig::new(self.config.clone(), channels, interpolation).init(device)?;

        #[cfg(feature = "train")]
        let loss = self.loss_config.as_ref().map(BiRefNetLossConfig::init);

        Ok(BiRefNet {
            mul_scl_ipt: self.config.decoder.mul_scl_ipt,
            cxt,
            bb,
            squeeze_module,
            decoder,
            refine,
            #[cfg(feature = "train")]
            loss,
            interpolation_strategy: interpolation,
        })
    }

    fn squeeze_block(
        &self,
        in_channels: usize,
        out_channels: usize,
        device: &Device,
    ) -> Result<SqueezeBlockModule, ModelError> {
        let decoder = &self.config.decoder;
        let batch_size = self.config.task.batch_size;
        let interpolation = self.config.interpolation;
        Ok(match decoder.squeeze_block {
            SqueezeBlock::BasicDecBlk(_) => SqueezeBlockModule::BasicDecBlk(
                BasicDecBlkConfig::new(decoder.dec_att, interpolation)
                    .with_in_channels(in_channels)
                    .with_out_channels(out_channels)
                    .with_dec_channels_inter(decoder.dec_channels_inter)
                    .with_batch_size(batch_size)
                    .init(device)?,
            ),
            SqueezeBlock::ResBlk(_) => SqueezeBlockModule::ResBlk(
                ResBlkConfig::new(interpolation)
                    .with_in_channels(in_channels)
                    .with_out_channels(Some(out_channels))
                    .with_dec_att(decoder.dec_att)
                    .with_dec_channels_inter(decoder.dec_channels_inter)
                    .with_batch_size(batch_size)
                    .init(device)?,
            ),
            SqueezeBlock::ASPP(_) => SqueezeBlockModule::ASPP(
                ASPPConfig::new(interpolation)
                    .with_in_channels(in_channels)
                    .with_out_channels(Some(out_channels))
                    .with_batch_size(batch_size)
                    .init(device)?,
            ),
            SqueezeBlock::ASPPDeformable(_) => SqueezeBlockModule::ASPPDeformable(
                ASPPDeformableConfig::new(interpolation)
                    .with_in_channels(in_channels)
                    .with_out_channels(Some(out_channels))
                    .with_batch_size(batch_size)
                    .init(device),
            ),
            SqueezeBlock::None => {
                return Err(ModelError::invalid_config(
                    "SqueezeBlock::None has no blocks to build",
                ));
            }
        })
    }
}

const fn backbone_type(backbone: Backbone) -> BackboneType {
    match backbone {
        Backbone::Vgg16 => BackboneType::Vgg(VggVariant::Vgg16),
        Backbone::Vgg16Bn => BackboneType::Vgg(VggVariant::Vgg16Bn),
        Backbone::ResNet50 => BackboneType::ResNet(ResNetVariant::ResNet50),
        Backbone::SwinV1T => BackboneType::SwinTransformer(SwinVariant::SwinT),
        Backbone::SwinV1S => BackboneType::SwinTransformer(SwinVariant::SwinS),
        Backbone::SwinV1B => BackboneType::SwinTransformer(SwinVariant::SwinB),
        Backbone::SwinV1L => BackboneType::SwinTransformer(SwinVariant::SwinL),
        Backbone::PvtV2B0 => BackboneType::PvtV2(PvtV2Variant::B0),
        Backbone::PvtV2B1 => BackboneType::PvtV2(PvtV2Variant::B1),
        Backbone::PvtV2B2 => BackboneType::PvtV2(PvtV2Variant::B2),
        Backbone::PvtV2B5 => BackboneType::PvtV2(PvtV2Variant::B5),
    }
}

/// Bilateral Reference Network (`BiRefNet`) model for high-resolution dichotomous segmentation.
#[derive(Module, Debug)]
pub struct BiRefNet {
    #[module(skip)]
    mul_scl_ipt: MultiScaleInput,
    cxt: [usize; 3],
    bb: BackboneModel,
    squeeze_module: Vec<SqueezeBlockModule>,
    decoder: Decoder,
    refine: Option<RefineModule>,
    #[cfg(feature = "train")]
    pub(crate) loss: Option<BiRefNetLoss>,
    #[module(skip)]
    interpolation_strategy: InterpolationStrategy,
}

impl BiRefNet {
    fn resize(&self, tensor: Tensor<4>, size: [usize; 2]) -> Tensor<4> {
        intelligent_interpolate(tensor, size, self.interpolation_strategy)
    }

    /// Extracts multi-stage encoder feature maps `[x1, x2, x3, x4]` from `x`.
    ///
    /// # Shapes
    /// - `x`: `[batch_size, 3, height, width]`
    /// - returns: `[x1, x2, x3, x4]` ordered shallow-to-deep.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `x` does not have 3 channels or the backbone rejects its shape.
    pub fn forward_enc(&self, x: Tensor<4>) -> Result<[Tensor<4>; 4], ModelError> {
        let [_, channels, height, width] = x.dims();
        if channels != 3 {
            return Err(ModelError::invalid_input(format!(
                "BiRefNet expects 3 RGB input channels, got {channels}"
            )));
        }

        // Half-resolution copy for multi-scale input fusion, taken before `x` is consumed.
        let half_x = (self.mul_scl_ipt != MultiScaleInput::None)
            .then(|| self.resize(x.clone(), [(height / 2).max(1), (width / 2).max(1)]));
        let features = self.bb.try_forward(x)?;

        let [x1, x2, x3, x4] = match half_x {
            None => features,
            Some(half_x) => {
                let fuse = |full: Tensor<4>, half: Tensor<4>| {
                    let [_, _, h, w] = full.dims();
                    let half_up = self.resize(half, [h, w]);
                    if self.mul_scl_ipt == MultiScaleInput::Add {
                        full + half_up
                    } else {
                        Tensor::cat(vec![full, half_up], 1)
                    }
                };
                let [f1, f2, f3, f4] = features;
                let [h1, h2, h3, h4] = self.bb.try_forward(half_x)?;
                [fuse(f1, h1), fuse(f2, h2), fuse(f3, h3), fuse(f4, h4)]
            }
        };

        // Upstream concatenates the deepest `cxt_num` shallower stages into x4.
        let active_cxt = self.cxt.iter().filter(|&&count| count > 0).count();
        let x4 = if active_cxt == 0 {
            x4
        } else {
            let [_, _, h4, w4] = x4.dims();
            let shallow = [&x1, &x2, &x3];
            let mut concat_inputs = Vec::with_capacity(active_cxt + 1);
            concat_inputs.extend(
                shallow[shallow.len() - active_cxt..]
                    .iter()
                    .map(|feat| self.resize((*feat).clone(), [h4, w4])),
            );
            concat_inputs.push(x4);
            Tensor::cat(concat_inputs, 1)
        };

        Ok([x1, x2, x3, x4])
    }

    pub(crate) fn forward_backbone_decoder(
        &self,
        x: &Tensor<4>,
        collect_ms: bool,
    ) -> Result<Vec<Tensor<4>>, ModelError> {
        let [x1, x2, x3, x4] = self.forward_enc(x.clone())?;
        let x4 = self
            .squeeze_module
            .iter()
            .fold(x4, |x4, block| block.forward(x4));

        let features = [x.clone(), x1, x2, x3, x4];
        Ok(if collect_ms {
            self.decoder.forward_multi_scale(features)
        } else {
            vec![self.decoder.forward(features)]
        })
    }

    /// Returns `true` if a refinement module runs after the decoder.
    #[must_use]
    pub const fn has_refiner(&self) -> bool {
        self.refine.is_some()
    }

    /// Runs inference forward pass returning segmentation logits `[batch_size, 1, height, width]`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if inputs have invalid shape or any refinement module fails.
    pub fn forward(&self, x: Tensor<4>) -> Result<Tensor<4>, ModelError> {
        let decoder_output = self
            .forward_backbone_decoder(&x, false)?
            .pop()
            .ok_or_else(|| ModelError::invalid_input("decoder emitted no predictions"))?;

        let Some(refine) = &self.refine else {
            return Ok(decoder_output);
        };

        let refined = match refine {
            RefineModule::RefUNet(refine_module) => {
                refine_module.forward(&[x, decoder_output.clone()])?
            }
            RefineModule::Refiner(refine_module) => {
                refine_module.forward(Tensor::cat(vec![x, decoder_output.clone()], 1))
            }
            RefineModule::RefinerPVTInChannels4(refine_module) => {
                refine_module.forward(Tensor::cat(vec![x, decoder_output.clone()], 1))
            }
        };

        Ok(refined.into_iter().last().unwrap_or(decoder_output))
    }
}
