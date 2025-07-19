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
};
use crate::{
    config::{ModelConfig, MulSclIpt, SqueezeBlock},
    error::{BiRefNetError, BiRefNetResult},
};
use backbones::{
    create_backbone, Backbone, BackboneType, BackboneWrapper, PVTv2Variant, ResNetVariant,
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
    losses::{CombinedLoss, CombinedLossConfig},
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

/// Configuration for the `BiRefNet` model.
#[derive(Config, Debug)]
pub struct BiRefNetConfig {
    /// The detailed model configuration.
    config: ModelConfig,
    /// The loss function configuration.
    #[cfg(feature = "train")]
    loss: CombinedLossConfig,
}

impl BiRefNetConfig {
    /// Creates a default configuration for BiRefNet with ResNet50 backbone
    pub fn default_resnet50() -> Self {
        use crate::config::*;

        let config = ModelConfig {
            path: PathConfig::new(),
            task: TaskConfig::new(),
            backbone: BackboneConfig {
                backbone: Backbone::Resnet50,
            },
            decoder: DecoderConfig {
                ms_supervision: false,
                out_ref: false,
                dec_ipt: false,
                dec_ipt_split: false,
                cxt_num: 3,
                mul_scl_ipt: MulSclIpt::None,
                dec_att: DecAtt::ASPPDeformable,
                squeeze_block: SqueezeBlock::ASPPDeformable(1),
                dec_blk: DecBlk::BasicDecBlk,
                lat_blk: LatBlk::BasicLatBlk,
                dec_channels_inter: DecChannelsInter::Fixed,
            },
            refine: RefineConfig {
                refine: Refine::None,
            },
        };

        Self {
            config,
            #[cfg(feature = "train")]
            loss: CombinedLossConfig::new(),
        }
    }

    /// Creates a default configuration for BiRefNet with VGG16 backbone
    pub fn default_vgg16() -> Self {
        let mut config = Self::default_resnet50();
        config.config.backbone.backbone = crate::config::Backbone::Vgg16;
        config
    }

    /// Creates a default configuration for BiRefNet with Swin Transformer backbone
    pub fn default_swin_t() -> Self {
        let mut config = Self::default_resnet50();
        config.config.backbone.backbone = crate::config::Backbone::SwinV1T;
        config
    }

    /// Creates a default configuration for BiRefNet with PVTv2 backbone
    pub fn default_pvt_v2_b2() -> Self {
        let mut config = Self::default_resnet50();
        config.config.backbone.backbone = crate::config::Backbone::PvtV2B2;
        config
    }
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
            MulSclIpt::None => MulSclIpt_::None(Identity::<B>::new()),
            MulSclIpt::Add => MulSclIpt_::Add(Identity::<B>::new()),
            MulSclIpt::Cat => MulSclIpt_::Cat(Identity::<B>::new()),
        };

        Ok(BiRefNet {
            mul_scl_ipt,
            cxt,
            bb,
            squeeze_module,
            decoder: DecoderConfig::new(self.config.clone(), channels).init(device)?,
            #[cfg(feature = "train")]
            loss: self.loss.init(),
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
            OldBackbone::PvtV2B0 => Ok(BackboneType::PVTv2(PVTv2Variant::B0)),
            OldBackbone::PvtV2B1 => Ok(BackboneType::PVTv2(PVTv2Variant::B1)),
            OldBackbone::PvtV2B2 => Ok(BackboneType::PVTv2(PVTv2Variant::B2)),
            OldBackbone::PvtV2B5 => Ok(BackboneType::PVTv2(PVTv2Variant::B5)),
        }
    }
}

/// An enum to handle different multi-scale input strategies.
#[derive(Module, Debug)]
enum MulSclIpt_<B: Backend> {
    None(Identity<B>),
    Add(Identity<B>),
    Cat(Identity<B>),
}

/// The main BiRefNet model.
#[derive(Module, Debug)]
pub struct BiRefNet<B: Backend> {
    /// The multi-scale input handling module.
    mul_scl_ipt: MulSclIpt_<B>,
    /// Context channel sizes.
    cxt: [usize; 3],
    /// The backbone encoder.
    bb: BackboneWrapper<B>,
    /// The squeeze module applied to the deepest encoder feature.
    squeeze_module: Vec<SqueezeBlockModule<B>>,
    /// The decoder module.
    decoder: Decoder<B>,
    /// The loss function for training.
    #[cfg(feature = "train")]
    loss: CombinedLoss<B>,
}

impl<B: Backend> BiRefNet<B> {
    /// Performs the forward pass through the encoder part of the network.
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor of shape `[B, C, H, W]`.
    ///
    /// # Returns
    ///
    /// A result containing a 4-element array of feature maps from the encoder stages.
    pub fn forward_enc(&self, x: Tensor<B, 4>) -> BiRefNetResult<[Tensor<B, 4>; 4]> {
        let [x1, x2, x3, x4] = self.bb.forward(x.clone());
        let [x1, x2, x3, x4] = match self.mul_scl_ipt {
            MulSclIpt_::None(_) => [x1, x2, x3, x4],
            MulSclIpt_::Add(_) => {
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
            MulSclIpt_::Cat(_) => {
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

    /// The original forward pass of the model.
    ///
    /// This method encapsulates the full process from encoder to decoder.
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor of shape `[B, C, H, W]`.
    ///
    /// # Returns
    ///
    /// A result containing the final segmentation map.
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
        let features = [x, x1, x2, x3, x4];

        Ok(self.decoder.forward(features))
    }

    /// The main forward pass for the `BiRefNet` model.
    ///
    /// # Arguments
    ///
    /// * `x` - The input tensor of shape `[B, C, H, W]`.
    ///
    /// # Returns
    ///
    /// A result containing the final segmentation map.
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
        let loss = self.loss.forward(logits.clone(), batch.masks.clone());

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
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_birefnet_config_creation() {
        // Test config creation without model initialization
        let resnet_config = BiRefNetConfig::default_resnet50();
        assert_eq!(
            resnet_config.config.backbone.backbone,
            crate::config::Backbone::Resnet50
        );

        let vgg_config = BiRefNetConfig::default_vgg16();
        assert_eq!(
            vgg_config.config.backbone.backbone,
            crate::config::Backbone::Vgg16
        );

        let swin_config = BiRefNetConfig::default_swin_t();
        assert_eq!(
            swin_config.config.backbone.backbone,
            crate::config::Backbone::SwinV1T
        );

        let pvt_config = BiRefNetConfig::default_pvt_v2_b2();
        assert_eq!(
            pvt_config.config.backbone.backbone,
            crate::config::Backbone::PvtV2B2
        );
    }

    #[test]
    fn test_birefnet_model_creation() {
        let config = BiRefNetConfig::default_resnet50();
        let device = Default::default();

        // Test model creation with ResNet50 backbone
        let model = config.init::<TestBackend>(&device);
        assert!(model.is_ok());

        // Don't test forward pass as it requires heavy computation
        // and complex tensor shape handling
        let _model = model.unwrap();
    }

    #[test]
    fn test_backbone_conversion_system() {
        let config = BiRefNetConfig::default_resnet50();

        // Test conversion of ResNet50
        let backbone_type = config.convert_backbone_config();
        assert!(backbone_type.is_ok());

        match backbone_type.unwrap() {
            BackboneType::ResNet(ResNetVariant::ResNet50) => {
                // Expected
            }
            _ => panic!("Expected ResNet50 backbone type"),
        }
    }

    #[test]
    fn test_different_backbone_configs() {
        // Light test of different backbone configuration types
        let resnet_config = BiRefNetConfig::default_resnet50();
        let vgg_config = BiRefNetConfig::default_vgg16();
        let swin_config = BiRefNetConfig::default_swin_t();
        let pvt_config = BiRefNetConfig::default_pvt_v2_b2();

        // Test backbone conversion system
        let resnet_backbone = resnet_config.convert_backbone_config();
        let vgg_backbone = vgg_config.convert_backbone_config();
        let swin_backbone = swin_config.convert_backbone_config();
        let pvt_backbone = pvt_config.convert_backbone_config();

        assert!(resnet_backbone.is_ok());
        assert!(vgg_backbone.is_ok());
        assert!(swin_backbone.is_ok());
        assert!(pvt_backbone.is_ok());

        // Test conversion correctness
        match resnet_backbone.unwrap() {
            BackboneType::ResNet(ResNetVariant::ResNet50) => { /* Expected */ }
            _ => panic!("Expected ResNet50"),
        }

        match vgg_backbone.unwrap() {
            BackboneType::VGG(VGGVariant::VGG16) => { /* Expected */ }
            _ => panic!("Expected VGG16"),
        }

        match swin_backbone.unwrap() {
            BackboneType::SwinTransformer(SwinVariant::SwinT) => { /* Expected */ }
            _ => panic!("Expected SwinT"),
        }

        match pvt_backbone.unwrap() {
            BackboneType::PVTv2(PVTv2Variant::B2) => { /* Expected */ }
            _ => panic!("Expected PVTv2B2"),
        }
    }
}
