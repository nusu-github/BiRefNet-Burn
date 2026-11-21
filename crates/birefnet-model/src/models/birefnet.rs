//! Main `BiRefNet` model uniting backbone encoder, squeeze stage, decoder, and optional refinement.

use birefnet_backbones::{
    Backbone as _, BackboneModel, BackboneType, PvtV2Variant, ResNetVariant, SwinVariant,
    VggVariant,
};
#[cfg(feature = "train")]
use birefnet_loss::{BiRefNetLoss, BiRefNetLossConfig};
use burn::prelude::*;
#[cfg(feature = "train")]
use burn::train::{InferenceStep, TrainOutput, TrainStep};

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
#[cfg(feature = "train")]
use crate::training::{BiRefNetBatch, BiRefNetOutput};
use crate::{
    config::{
        Backbone, DecoderAttention, InterpolationStrategy, ModelConfig, MultiScaleInput, Refine,
        SqueezeBlock,
    },
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

fn nested_squeeze_for_attention(attention: DecoderAttention) -> SqueezeBlock {
    match attention {
        DecoderAttention::None => SqueezeBlock::None,
        DecoderAttention::ASPP => SqueezeBlock::ASPP(0),
        DecoderAttention::ASPPDeformable => SqueezeBlock::ASPPDeformable(0),
    }
}

impl BiRefNetConfig {
    /// Initializes a [`BiRefNet`] model on `device`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if [`ModelConfig::validate`] fails or any submodule
    /// cannot be initialized.
    pub fn init(&self, device: &Device) -> Result<BiRefNet, ModelError> {
        self.config.validate()?;

        let bb = self.backbone_type().init(device)?;
        let channels = self.config.lateral_channels_in_collection();
        let cxt = self.config.cxt()?;
        let cxt_sum = cxt.iter().sum::<usize>();
        let dec_att = self.config.decoder.dec_att;
        let dec_channels_inter = self.config.decoder.dec_channels_inter;
        let batch_size = self.config.task.batch_size;
        let count = self.config.decoder.squeeze_block.count();

        let mut squeeze_module = Vec::with_capacity(count);
        for idx in 0..count {
            let in_channels = if idx == 0 {
                channels[0] + cxt_sum
            } else {
                channels[0]
            };
            let block = match self.config.decoder.squeeze_block {
                SqueezeBlock::BasicDecBlk(_) => SqueezeBlockModule::BasicDecBlk(
                    BasicDecBlkConfig::new(
                        nested_squeeze_for_attention(dec_att),
                        self.config.interpolation,
                    )
                    .with_in_channels(in_channels)
                    .with_out_channels(channels[0])
                    .with_dec_channels_inter(dec_channels_inter)
                    .with_batch_size(batch_size)
                    .init(device)?,
                ),
                SqueezeBlock::ResBlk(_) => SqueezeBlockModule::ResBlk(
                    ResBlkConfig::new(self.config.interpolation)
                        .with_in_channels(in_channels)
                        .with_out_channels(Some(channels[0]))
                        .with_dec_att(dec_att)
                        .with_dec_channels_inter(dec_channels_inter)
                        .with_batch_size(batch_size)
                        .init(device)?,
                ),
                SqueezeBlock::ASPP(_) => SqueezeBlockModule::ASPP(
                    ASPPConfig::new(self.config.interpolation)
                        .with_in_channels(in_channels)
                        .with_out_channels(Some(channels[0]))
                        .with_batch_size(batch_size)
                        .init(device)?,
                ),
                SqueezeBlock::ASPPDeformable(_) => SqueezeBlockModule::ASPPDeformable(
                    ASPPDeformableConfig::new(self.config.interpolation)
                        .with_in_channels(in_channels)
                        .with_out_channels(Some(channels[0]))
                        .with_batch_size(batch_size)
                        .init(device),
                ),
                SqueezeBlock::None => unreachable!("count > 0 excludes SqueezeBlock::None"),
            };
            squeeze_module.push(block);
        }

        let refine = match self.config.refine.refine {
            Refine::RefUNet => Some(RefineModule::RefUNet(
                RefUNetConfig::new(self.config.interpolation).init(device),
            )),
            Refine::Refiner => Some(RefineModule::Refiner(
                RefinerConfig::new(self.config.clone(), self.config.interpolation).init(device)?,
            )),
            Refine::RefinerPVTInChannels4 => Some(RefineModule::RefinerPVTInChannels4(
                RefinerPVTInChannels4Config::new(self.config.clone(), self.config.interpolation)
                    .init(device)?,
            )),
            Refine::Itself | Refine::None => None,
        };

        let decoder = DecoderConfig::new(self.config.clone(), channels, self.config.interpolation)
            .init(device)?;

        #[cfg(feature = "train")]
        let loss = self.loss_config.as_ref().map(|config| config.init(device));

        Ok(BiRefNet {
            mul_scl_ipt: self.config.decoder.mul_scl_ipt,
            cxt,
            bb,
            squeeze_module,
            decoder,
            refine,
            #[cfg(feature = "train")]
            loss,
            interpolation_strategy: self.config.interpolation,
        })
    }

    const fn backbone_type(&self) -> BackboneType {
        match self.config.backbone.backbone {
            Backbone::Vgg16 => BackboneType::Vgg(VggVariant::Vgg16),
            Backbone::Vgg16bn => BackboneType::Vgg(VggVariant::Vgg16Bn),
            Backbone::Resnet50 => BackboneType::ResNet(ResNetVariant::ResNet50),
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
    loss: Option<BiRefNetLoss>,
    #[module(skip)]
    interpolation_strategy: InterpolationStrategy,
}

impl BiRefNet {
    /// Extracts multi-stage encoder feature maps `[x1, x2, x3, x4]` from `x`.
    ///
    /// # Shapes
    /// - `x`: `[batch_size, 3, height, width]`
    /// - returns: `[x1, x2, x3, x4]` ordered shallow-to-deep.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `x` has invalid channel count (`!= 3`).
    pub fn forward_enc(&self, x: Tensor<4>) -> Result<[Tensor<4>; 4], ModelError> {
        let [_, channels, height, width] = x.dims();
        if channels != 3 {
            return Err(ModelError::invalid_input(format!(
                "BiRefNet expects 3 RGB input channels, got {channels}"
            )));
        }

        let [mut x1, mut x2, mut x3, mut x4] = self.bb.forward(x.clone());

        if !matches!(self.mul_scl_ipt, MultiScaleInput::None) {
            let half_h = (height / 2).max(1);
            let half_w = (width / 2).max(1);
            let half_x = intelligent_interpolate(x, [half_h, half_w], self.interpolation_strategy);
            let [x1_half, x2_half, x3_half, x4_half] = self.bb.forward(half_x);

            let align = |base: &Tensor<4>, low_res: Tensor<4>| {
                let [_, _, target_h, target_w] = base.dims();
                intelligent_interpolate(low_res, [target_h, target_w], self.interpolation_strategy)
            };

            let x1_up = align(&x1, x1_half);
            let x2_up = align(&x2, x2_half);
            let x3_up = align(&x3, x3_half);
            let x4_up = align(&x4, x4_half);

            match self.mul_scl_ipt {
                MultiScaleInput::Add => {
                    x1 = x1 + x1_up;
                    x2 = x2 + x2_up;
                    x3 = x3 + x3_up;
                    x4 = x4 + x4_up;
                }
                MultiScaleInput::Cat => {
                    x1 = Tensor::cat(vec![x1, x1_up], 1);
                    x2 = Tensor::cat(vec![x2, x2_up], 1);
                    x3 = Tensor::cat(vec![x3, x3_up], 1);
                    x4 = Tensor::cat(vec![x4, x4_up], 1);
                }
                MultiScaleInput::None => unreachable!(),
            }
        }

        let active_cxt = self.cxt.iter().filter(|&&count| count > 0).count();
        if active_cxt > 0 {
            let [_, _, h4, w4] = x4.dims();
            let candidates = [&x1, &x2, &x3];
            let start = candidates.len().saturating_sub(active_cxt);
            let mut concat_inputs = Vec::with_capacity(active_cxt + 1);
            for feat in &candidates[start..] {
                concat_inputs.push(intelligent_interpolate(
                    (*feat).clone(),
                    [h4, w4],
                    self.interpolation_strategy,
                ));
            }
            concat_inputs.push(x4);
            x4 = Tensor::cat(concat_inputs, 1);
        }

        Ok([x1, x2, x3, x4])
    }

    fn forward_backbone_decoder(
        &self,
        x: &Tensor<4>,
        collect_ms: bool,
    ) -> Result<Vec<Tensor<4>>, ModelError> {
        let [x1, x2, x3, mut x4] = self.forward_enc(x.clone())?;
        for squeeze_module in &self.squeeze_module {
            x4 = match squeeze_module {
                SqueezeBlockModule::BasicDecBlk(model) => model.forward(x4),
                SqueezeBlockModule::ResBlk(model) => model.forward(x4),
                SqueezeBlockModule::ASPP(model) => model.forward(x4),
                SqueezeBlockModule::ASPPDeformable(model) => model.forward(x4),
            };
        }

        let features = [x.clone(), x1, x2, x3, x4];
        if collect_ms {
            Ok(self.decoder.forward_multi_scale(features))
        } else {
            Ok(vec![self.decoder.forward(features)])
        }
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

        let refined_output = match refine {
            RefineModule::RefUNet(refine_module) => refine_module
                .forward(&[x, decoder_output.clone()])?
                .into_iter()
                .last()
                .unwrap_or(decoder_output),
            RefineModule::Refiner(refine_module) => {
                let combined = Tensor::cat(vec![x, decoder_output.clone()], 1);
                refine_module
                    .forward(combined)
                    .into_iter()
                    .last()
                    .unwrap_or(decoder_output)
            }
            RefineModule::RefinerPVTInChannels4(refine_module) => {
                let combined = Tensor::cat(vec![x, decoder_output.clone()], 1);
                refine_module
                    .forward(combined)
                    .into_iter()
                    .last()
                    .unwrap_or(decoder_output)
            }
        };

        Ok(refined_output)
    }

    /// Runs a supervised training/validation step returning predictions, targets, and loss.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `loss_config` is unset on [`BiRefNetConfig`] or forward/loss fails.
    #[cfg(feature = "train")]
    pub fn forward_step(&self, batch: BiRefNetBatch) -> Result<BiRefNetOutput, ModelError> {
        let loss_fn = self.loss.as_ref().ok_or_else(|| {
            ModelError::invalid_config("loss_config is not set on BiRefNetConfig")
        })?;

        let (prediction, loss) = if self.refine.is_none() {
            let scaled_preds = self.forward_backbone_decoder(&batch.images, true)?;
            let loss = loss_fn.forward(&scaled_preds, &batch.masks)?;
            let prediction = scaled_preds
                .into_iter()
                .last()
                .ok_or_else(|| ModelError::invalid_input("decoder emitted no predictions"))?;
            (prediction, loss)
        } else {
            let prediction = self.forward(batch.images)?;
            let loss = loss_fn.forward(core::slice::from_ref(&prediction), &batch.masks)?;
            (prediction, loss)
        };

        Ok(BiRefNetOutput::new(loss, prediction, batch.masks))
    }
}

#[cfg(feature = "train")]
impl TrainStep for BiRefNet {
    type Input = BiRefNetBatch;
    type Output = BiRefNetOutput;

    fn step(&self, item: BiRefNetBatch) -> TrainOutput<BiRefNetOutput> {
        let output = self
            .forward_step(item)
            .expect("BiRefNet train forward_step failed");
        TrainOutput::new(self, output.loss.backward(), output)
    }
}

#[cfg(feature = "train")]
impl InferenceStep for BiRefNet {
    type Input = BiRefNetBatch;
    type Output = BiRefNetOutput;

    fn step(&self, item: BiRefNetBatch) -> BiRefNetOutput {
        self.forward_step(item)
            .expect("BiRefNet validation forward_step failed")
    }
}
