//! # BiRefNet Model Implementation
//!
//! This module defines the main `BiRefNet` model, which combines a backbone encoder
//! with a sophisticated decoder to produce segmentation masks.
//!
//! ## Core Components
//!
//! - `BiRefNetConfig`: A configuration struct to initialize the `BiRefNet` model.
//! - `BiRefNet`: The main model struct, which orchestrates the forward pass through
//!   the encoder and decoder.
//!
//! The implementation closely follows the original PyTorch version, including support for
//! multi-scale inputs, context aggregation, and various decoder block configurations.

use super::{
    decoder::{Decoder, DecoderConfig},
    modules::{
        ASPPConfig, ASPPDeformable, ASPPDeformableConfig, BasicDecBlk, BasicDecBlkConfig, ResBlk,
        ResBlkConfig, ASPP,
    },
    refinement::refiner::{
        RefUNet, RefUNetConfig, Refiner, RefinerConfig, RefinerPVTInChannels4,
        RefinerPVTInChannels4Config,
    },
};
use crate::{
    config::{ModelConfig, MulSclIpt, SqueezeBlock},
    error::{BiRefNetError, BiRefNetResult},
};
use backbones::{
    create_backbone, Backbone, BackboneType, BackboneWrapper, PvtV2Variant, ResNetVariant,
    SwinVariant, VGGVariant,
};
use burn::{
    prelude::*,
    tensor::{
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};
use burn_extra_ops::Identity;

#[cfg(feature = "train")]
use crate::{
    dataset::BiRefNetBatch,
    // Using complete BiRefNet training loss system (Level 3)
    // This matches the original PyTorch implementation: PixLoss + ClsLoss + GDT Loss
    losses::{BiRefNetLoss, BiRefNetLossConfig},
    training::BiRefNetOutput,
};

#[cfg(feature = "train")]
use burn::{
    tensor::backend::AutodiffBackend,
    train::{TrainOutput, TrainStep, ValidStep},
};

/// An enum to wrap different types of squeeze blocks used in the decoder.
#[derive(Module, Debug)]
pub enum SqueezeBlockModule<B: Backend> {
    BasicDecBlk(BasicDecBlk<B>),
    ResBlk(ResBlk<B>),
    ASPP(ASPP<B>),
    ASPPDeformable(ASPPDeformable<B>),
}

/// An enum to wrap different types of refinement modules.
#[derive(Module, Debug)]
pub enum RefineModule<B: Backend> {
    RefUNet(RefUNet<B>),
    Refiner(Refiner<B>),
    RefinerPVTInChannels4(RefinerPVTInChannels4<B>),
    None(Identity<B>),
}

/// Configuration for the `BiRefNet` model.
#[derive(Config, Debug)]
pub struct BiRefNetConfig {
    /// The detailed model configuration.
    config: ModelConfig,
    /// The complete BiRefNet training loss system configuration (Level 3).
    ///
    /// This integrates PixLoss + ClsLoss + GDT Loss as in the original PyTorch implementation.
    #[cfg(feature = "train")]
    loss: BiRefNetLossConfig,
}

impl BiRefNetConfig {
    /// Initializes a `BiRefNet` model with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to create the model on.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid or model initialization fails.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> BiRefNetResult<BiRefNet<B>> {
        let backbone_type = self.convert_backbone_config()?;
        let bb = create_backbone(backbone_type, device);
        let channels = self.config.lateral_channels_in_collection();
        let cxt = self.config.cxt()?;

        let squeeze_module = if self.config.decoder.squeeze_block == SqueezeBlock::None {
            vec![]
        } else {
            let mut squeeze_module = Vec::with_capacity(self.config.decoder.squeeze_block.count());

            for _ in 0..self.config.decoder.squeeze_block.count() {
                match self.config.decoder.squeeze_block {
                    SqueezeBlock::BasicDecBlk(_) => {
                        let cxt_sum = cxt.iter().sum::<usize>();
                        let model = BasicDecBlkConfig::new(SqueezeBlock::ASPPDeformable(0))
                            .with_in_channels(channels[0] + cxt_sum)
                            .with_out_channels(channels[0])
                            .init(device)?;
                        squeeze_module.push(SqueezeBlockModule::BasicDecBlk(model));
                    }
                    SqueezeBlock::ResBlk(_) => {
                        let cxt_sum = cxt.iter().sum::<usize>();
                        let model = ResBlkConfig::new()
                            .with_in_channels(channels[0] + cxt_sum)
                            .with_out_channels(Some(channels[0]))
                            .init(device)?;
                        squeeze_module.push(SqueezeBlockModule::ResBlk(model));
                    }
                    SqueezeBlock::ASPP(_) => {
                        let cxt_sum = cxt.iter().sum::<usize>();
                        let model = ASPPConfig::new()
                            .with_in_channels(channels[0] + cxt_sum)
                            .with_out_channels(Some(channels[0]))
                            .init(device);
                        squeeze_module.push(SqueezeBlockModule::ASPP(model));
                    }
                    SqueezeBlock::ASPPDeformable(_) => {
                        let cxt_sum = cxt.iter().sum::<usize>();
                        let model = ASPPDeformableConfig::new()
                            .with_in_channels(channels[0] + cxt_sum)
                            .with_out_channels(Some(channels[0]))
                            .init(device)?;
                        squeeze_module.push(SqueezeBlockModule::ASPPDeformable(model));
                    }
                    SqueezeBlock::None => {
                        return Err(BiRefNetError::InvalidConfiguration {
                            reason: "SqueezeBlock::None should not be processed in this loop"
                                .to_string(),
                        });
                    }
                };
            }

            squeeze_module
        };

        let mul_scl_ipt = match self.config.decoder.mul_scl_ipt {
            MulSclIpt::None => MulSclIptRecord::None(Identity::<B>::new()),
            MulSclIpt::Add => MulSclIptRecord::Add(Identity::<B>::new()),
            MulSclIpt::Cat => MulSclIptRecord::Cat(Identity::<B>::new()),
        };

        // Initialize refinement module based on configuration
        let refine = match self.config.refine.refine {
            crate::config::Refine::RefUNet => {
                let refine_config = RefUNetConfig::new();
                RefineModule::RefUNet(refine_config.init(device))
            }
            crate::config::Refine::Refiner => {
                let refine_config = RefinerConfig::new(self.config.clone());
                RefineModule::Refiner(refine_config.init(device)?)
            }
            crate::config::Refine::RefinerPVTInChannels4 => {
                let refine_config = RefinerPVTInChannels4Config::new(self.config.clone());
                RefineModule::RefinerPVTInChannels4(refine_config.init(device)?)
            }
            crate::config::Refine::Itself => RefineModule::None(Identity::<B>::new()),
            crate::config::Refine::None => RefineModule::None(Identity::<B>::new()),
        };

        Ok(BiRefNet {
            mul_scl_ipt,
            cxt,
            bb,
            squeeze_module,
            decoder: DecoderConfig::new(self.config.clone(), channels).init(device)?,
            refine,
            #[cfg(feature = "train")]
            loss: self.loss.init(device),
        })
    }

    /// Converts the old Backbone enum in config to the new BackboneType enum.
    const fn convert_backbone_config(&self) -> BiRefNetResult<BackboneType> {
        use crate::config::Backbone as OldBackbone;

        match self.config.backbone.backbone {
            OldBackbone::Vgg16 => Ok(BackboneType::VGG(VGGVariant::VGG16)),
            OldBackbone::Vgg16bn => Ok(BackboneType::VGG(VGGVariant::VGG16BN)),
            OldBackbone::Resnet50 => Ok(BackboneType::ResNet(ResNetVariant::ResNet50)),
            OldBackbone::SwinV1T => Ok(BackboneType::SwinTransformer(SwinVariant::SwinT)),
            OldBackbone::SwinV1S => Ok(BackboneType::SwinTransformer(SwinVariant::SwinS)),
            OldBackbone::SwinV1B => Ok(BackboneType::SwinTransformer(SwinVariant::SwinB)),
            OldBackbone::SwinV1L => Ok(BackboneType::SwinTransformer(SwinVariant::SwinL)),
            OldBackbone::PvtV2B0 => Ok(BackboneType::PvtV2(PvtV2Variant::B0)),
            OldBackbone::PvtV2B1 => Ok(BackboneType::PvtV2(PvtV2Variant::B1)),
            OldBackbone::PvtV2B2 => Ok(BackboneType::PvtV2(PvtV2Variant::B2)),
            OldBackbone::PvtV2B5 => Ok(BackboneType::PvtV2(PvtV2Variant::B5)),
        }
    }
}

/// An enum to handle different multi-scale input strategies.
#[derive(Module, Debug)]
enum MulSclIptRecord<B: Backend> {
    None(Identity<B>),
    Add(Identity<B>),
    Cat(Identity<B>),
}

/// The main BiRefNet model.
#[derive(Module, Debug)]
pub struct BiRefNet<B: Backend> {
    /// The multi-scale input handling module.
    mul_scl_ipt: MulSclIptRecord<B>,
    /// Context channel sizes.
    cxt: [usize; 3],
    /// The backbone encoder.
    bb: BackboneWrapper<B>,
    /// The squeeze module applied to the deepest encoder feature.
    squeeze_module: Vec<SqueezeBlockModule<B>>,
    /// The decoder module.
    decoder: Decoder<B>,
    /// The refinement module.
    refine: RefineModule<B>,
    /// The complete BiRefNet training loss system.
    #[cfg(feature = "train")]
    loss: BiRefNetLoss<B>,
}

impl<B: Backend> BiRefNet<B> {
    /// Performs the forward pass through the encoder part of the network.
    ///
    /// # Shapes
    /// * `x` - Input tensor: `[batch_size, channels, height, width]`
    /// * Returns - Feature maps: `[[batch_size, C1, H/4, W/4], [batch_size, C2, H/8, W/8], [batch_size, C3, H/16, W/16], [batch_size, C4, H/32, W/32]]`
    ///
    /// # Arguments
    /// * `x` - The input RGB image tensor
    ///
    /// # Returns
    /// A result containing a 4-element array of hierarchical feature maps from different encoder stages with decreasing spatial resolutions
    pub fn forward_enc(&self, x: Tensor<B, 4>) -> BiRefNetResult<[Tensor<B, 4>; 4]> {
        let [x1, x2, x3, x4] = self.bb.forward(x.clone());
        let [x1, x2, x3, x4] = match self.mul_scl_ipt {
            MulSclIptRecord::None(_) => [x1, x2, x3, x4],
            MulSclIptRecord::Add(_) => {
                let [_, _, h, w] = x.dims();
                let [x1_, x2_, x3_, x4_] = self.bb.forward(interpolate(
                    x,
                    [h / 2, w / 2],
                    InterpolateOptions::new(InterpolateMode::Bilinear),
                ));

                let [_, _, h, w] = x1.dims();
                let x1 = x1
                    + interpolate(
                        x1_,
                        [h, w],
                        InterpolateOptions::new(InterpolateMode::Bilinear),
                    );
                let [_, _, h, w] = x2.dims();
                let x2 = x2
                    + interpolate(
                        x2_,
                        [h, w],
                        InterpolateOptions::new(InterpolateMode::Bilinear),
                    );
                let [_, _, h, w] = x3.dims();
                let x3 = x3
                    + interpolate(
                        x3_,
                        [h, w],
                        InterpolateOptions::new(InterpolateMode::Bilinear),
                    );
                let [_, _, h, w] = x4.dims();
                let x4 = x4
                    + interpolate(
                        x4_,
                        [h, w],
                        InterpolateOptions::new(InterpolateMode::Bilinear),
                    );

                [x1, x2, x3, x4]
            }
            MulSclIptRecord::Cat(_) => {
                let [_, _, h, w] = x.dims();
                let [x1_, x2_, x3_, x4_] = self.bb.forward(interpolate(
                    x,
                    [h / 2, w / 2],
                    InterpolateOptions::new(InterpolateMode::Bilinear),
                ));

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
            Ok([x1, x2, x3, x4])
        } else {
            Ok([x1, x2, x3, x4])
        }
    }

    /// Full forward pass that returns comprehensive output for complete training loss computation
    ///
    /// This method performs the complete BiRefNet forward pass including multi-scale predictions,
    /// classification outputs, and gradient direction tensor (GDT) outputs when configured.
    ///
    /// # Arguments
    /// * `x` - Input tensor with shape [batch_size, channels, height, width]
    ///
    /// # Returns
    /// BiRefNetFullOutput containing:
    /// - scaled_preds: Multi-scale segmentation predictions
    /// - class_preds: Classification predictions (if enabled)
    /// - gdt_outputs: Gradient direction tensor outputs (if enabled)
    /// - primary_pred: Main high-resolution prediction
    pub fn forward_full(
        &self,
        x: Tensor<B, 4>,
    ) -> BiRefNetResult<crate::training::BiRefNetFullOutput<B>> {
        // ########## Encoder ##########
        let [x1, x2, x3, x4] = self.forward_enc(x.clone())?;

        // Apply squeeze modules to the deepest feature
        let mut x4_processed = x4;
        for squeeze_module in &self.squeeze_module {
            match squeeze_module {
                SqueezeBlockModule::BasicDecBlk(model) => {
                    x4_processed = model.forward(x4_processed);
                }
                SqueezeBlockModule::ResBlk(model) => {
                    x4_processed = model.forward(x4_processed);
                }
                SqueezeBlockModule::ASPP(model) => {
                    x4_processed = model.forward(x4_processed);
                }
                SqueezeBlockModule::ASPPDeformable(model) => {
                    x4_processed = model.forward(x4_processed);
                }
            }
        }

        // ########## Decoder ##########
        let features = [x.clone(), x1, x2, x3, x4_processed];
        let decoder_output = self.decoder.forward(features);

        // Generate multi-scale predictions
        let [_orig_batch, _orig_channels, orig_h, orig_w] = x.dims();
        let mut scaled_preds = Vec::new();

        // Primary prediction (full resolution)
        let primary_pred = interpolate(
            decoder_output.clone(),
            [orig_h, orig_w],
            InterpolateOptions::new(InterpolateMode::Bilinear),
        );
        scaled_preds.push(primary_pred);

        // Additional scales for multi-scale loss computation
        // Add half resolution
        let half_res_pred = interpolate(
            decoder_output.clone(),
            [orig_h / 2, orig_w / 2],
            InterpolateOptions::new(InterpolateMode::Bilinear),
        );
        scaled_preds.push(half_res_pred);

        // Add quarter resolution
        let quarter_res_pred = interpolate(
            decoder_output,
            [orig_h / 4, orig_w / 4],
            InterpolateOptions::new(InterpolateMode::Bilinear),
        );
        scaled_preds.push(quarter_res_pred);

        // For now, we don't implement class_preds and gdt_outputs as they depend on
        // additional modules not yet implemented in this version
        let class_preds = None;
        let gdt_outputs = None;

        Ok(crate::training::BiRefNetFullOutput::new(
            scaled_preds,
            class_preds,
            gdt_outputs,
        ))
    }

    /// The original forward pass of the model.
    ///
    /// This method encapsulates the full process from encoder to decoder, including
    /// multi-scale feature extraction, squeeze operations, and hierarchical decoding.
    ///
    /// # Shapes
    /// * `x` - Input tensor: `[batch_size, channels, height, width]`
    /// * Returns - Segmentation map: `[batch_size, 1, height, width]`
    ///
    /// # Arguments
    /// * `x` - The input RGB image tensor
    ///   
    /// # Returns
    /// A result containing the final segmentation map with values representing pixel-wise probabilities
    pub fn forward_ori(&self, x: Tensor<B, 4>) -> BiRefNetResult<Tensor<B, 4>> {
        // ########## Encoder ##########
        let [x1, x2, x3, x4] = self.forward_enc(x.clone())?;
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
        let features = [x.clone(), x1, x2, x3, x4];
        let decoder_output = self.decoder.forward(features);

        // ########## Refinement ##########
        let refined_output = match &self.refine {
            RefineModule::RefUNet(refine_module) => {
                let refined_features = refine_module.forward(vec![x, decoder_output.clone()]);
                refined_features
                    .into_iter()
                    .last()
                    .unwrap_or(decoder_output)
            }
            RefineModule::Refiner(refine_module) => {
                let refined_features = refine_module.forward(x);
                refined_features
                    .into_iter()
                    .last()
                    .unwrap_or(decoder_output)
            }
            RefineModule::RefinerPVTInChannels4(refine_module) => {
                let refined_features = refine_module.forward(x);
                refined_features
                    .into_iter()
                    .last()
                    .unwrap_or(decoder_output)
            }
            RefineModule::None(identity) => identity.forward(decoder_output),
        };

        Ok(refined_output)
    }

    /// The main forward pass for the `BiRefNet` model.
    ///
    /// # Shapes
    /// * `x` - Input tensor: `[batch_size, channels, height, width]`
    /// * Returns - Segmentation map: `[batch_size, 1, height, width]`
    ///
    /// # Arguments
    /// * `x` - The input RGB image tensor
    ///
    /// # Returns
    /// A result containing the final binary segmentation map with probability values
    pub fn forward(&self, x: Tensor<B, 4>) -> BiRefNetResult<Tensor<B, 4>> {
        let scaled_preds = self.forward_ori(x)?;

        Ok(scaled_preds)
    }

    /// Forward pass for training and validation.
    #[cfg(feature = "train")]
    pub fn forward_classification(
        &self,
        batch: BiRefNetBatch<B>,
    ) -> BiRefNetResult<BiRefNetOutput<B>> {
        let logits = self.forward(batch.images)?;
        // Use simplified BiRefNet loss system for basic training
        // TODO: Implement full forward() method that returns BiRefNetFullOutput struct
        // Current: Only returning final segmentation prediction
        // Should implement:
        // - Multi-scale predictions for supervision at different resolutions
        // - Classification predictions for auxiliary loss
        // - Gradient descent target (GDT) outputs for refinement loss
        // - Proper integration with BiRefNetLoss for complete training pipeline
        let loss = self.loss.forward_simple(
            vec![logits.clone()], // Convert single output to vec for multi-scale interface
            batch.masks.clone(),
        );

        Ok(BiRefNetOutput {
            loss,
            logits,
            target: batch.masks,
        })
    }
}

#[cfg(feature = "train")]
impl<B: AutodiffBackend> TrainStep<BiRefNetBatch<B>, BiRefNetOutput<B>> for BiRefNet<B> {
    fn step(&self, batch: BiRefNetBatch<B>) -> TrainOutput<BiRefNetOutput<B>> {
        let item = self.forward_classification(batch).unwrap();
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

#[cfg(feature = "train")]
impl<B: Backend> ValidStep<BiRefNetBatch<B>, BiRefNetOutput<B>> for BiRefNet<B> {
    fn step(&self, batch: BiRefNetBatch<B>) -> BiRefNetOutput<B> {
        self.forward_classification(batch).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;
}
