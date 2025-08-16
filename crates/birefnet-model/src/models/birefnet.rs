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

use birefnet_backbones::{
    create_backbone, Backbone, BackboneType, BackboneWrapper, PvtV2Variant, ResNetVariant,
    SwinVariant, VGGVariant,
};
#[cfg(feature = "train")]
use birefnet_loss::{BiRefNetLoss, BiRefNetLossConfig};
use burn::{module::Ignored, prelude::*};
#[cfg(feature = "train")]
use burn::{
    tensor::backend::AutodiffBackend,
    train::{TrainOutput, TrainStep, ValidStep},
};

use super::{
    decoder::{Decoder, DecoderConfig},
    modules::{
        ASPPConfig, ASPPDeformable, ASPPDeformableConfig, BasicDecBlk, BasicDecBlkConfig, ResBlk,
        ResBlkConfig, ASPP,
    },
    refinement::{
        RefUNet, RefUNetConfig, Refiner, RefinerConfig, RefinerPVTInChannels4,
        RefinerPVTInChannels4Config,
    },
};
#[cfg(feature = "train")]
use crate::training::{BiRefNetBatch, BiRefNetOutput};
use crate::{
    config::{InterpolationStrategy, ModelConfig, MultiScaleInput, Refine, SqueezeBlock},
    error::{BiRefNetError, BiRefNetResult},
    models::modules::utils::intelligent_interpolate,
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
}

/// Configuration for the `BiRefNet` model.
#[derive(Config, Debug)]
pub struct BiRefNetConfig {
    /// The detailed model configuration.
    config: ModelConfig,
    /// The loss configuration for training.
    #[cfg(feature = "train")]
    loss_config: Option<BiRefNetLossConfig>,
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
                        let model = BasicDecBlkConfig::new(
                            SqueezeBlock::ASPPDeformable(0),
                            self.config.interpolation.clone(),
                        )
                        .with_in_channels(channels[0] + cxt_sum)
                        .with_out_channels(channels[0])
                        .init(device)?;
                        squeeze_module.push(SqueezeBlockModule::BasicDecBlk(model));
                    }
                    SqueezeBlock::ResBlk(_) => {
                        let cxt_sum = cxt.iter().sum::<usize>();
                        let model = ResBlkConfig::new(self.config.interpolation.clone())
                            .with_in_channels(channels[0] + cxt_sum)
                            .with_out_channels(Some(channels[0]))
                            .init(device)?;
                        squeeze_module.push(SqueezeBlockModule::ResBlk(model));
                    }
                    SqueezeBlock::ASPP(_) => {
                        let cxt_sum = cxt.iter().sum::<usize>();
                        let model = ASPPConfig::new(self.config.interpolation.clone())
                            .with_in_channels(channels[0] + cxt_sum)
                            .with_out_channels(Some(channels[0]))
                            .init(device);
                        squeeze_module.push(SqueezeBlockModule::ASPP(model));
                    }
                    SqueezeBlock::ASPPDeformable(_) => {
                        let cxt_sum = cxt.iter().sum::<usize>();
                        let model = ASPPDeformableConfig::new(self.config.interpolation.clone())
                            .with_in_channels(channels[0] + cxt_sum)
                            .with_out_channels(Some(channels[0]))
                            .init(device)?;
                        squeeze_module.push(SqueezeBlockModule::ASPPDeformable(model));
                    }
                    SqueezeBlock::None => {
                        return Err(BiRefNetError::InvalidConfiguration {
                            reason: "SqueezeBlock::None should not be processed in this loop"
                                .to_owned(),
                        });
                    }
                }
            }

            squeeze_module
        };

        let mul_scl_ipt = Ignored(self.config.decoder.mul_scl_ipt.clone());

        // Initialize refinement module based on configuration
        let refine = match self.config.refine.refine {
            Refine::RefUNet => {
                let refine_config = RefUNetConfig::new(self.config.interpolation.clone());
                Some(RefineModule::RefUNet(refine_config.init(device)))
            }
            Refine::Refiner => {
                let refine_config =
                    RefinerConfig::new(self.config.clone(), self.config.interpolation.clone());
                Some(RefineModule::Refiner(refine_config.init(device)?))
            }
            Refine::RefinerPVTInChannels4 => {
                let refine_config = RefinerPVTInChannels4Config::new(
                    self.config.clone(),
                    self.config.interpolation.clone(),
                );
                Some(RefineModule::RefinerPVTInChannels4(
                    refine_config.init(device)?,
                ))
            }
            Refine::Itself => None,
            Refine::None => None,
        };

        #[cfg(feature = "train")]
        let loss = self
            .loss_config
            .clone()
            .map(|config| BiRefNetLoss::new(config));

        let interpolation_strategy = Ignored(self.config.interpolation.clone());

        Ok(BiRefNet {
            mul_scl_ipt,
            cxt,
            bb,
            squeeze_module,
            decoder: DecoderConfig::new(
                self.config.clone(),
                channels,
                self.config.interpolation.clone(),
            )
            .init(device)?,
            refine,
            #[cfg(feature = "train")]
            loss,
            interpolation_strategy,
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

/// The main BiRefNet model.
#[derive(Module, Debug)]
pub struct BiRefNet<B: Backend> {
    /// The multi-scale input handling strategy.
    mul_scl_ipt: Ignored<MultiScaleInput>,
    /// Context channel sizes.
    cxt: [usize; 3],
    /// The backbone encoder.
    bb: BackboneWrapper<B>,
    /// The squeeze module applied to the deepest encoder feature.
    squeeze_module: Vec<SqueezeBlockModule<B>>,
    /// The decoder module.
    decoder: Decoder<B>,
    /// The refinement module.
    refine: Option<RefineModule<B>>,
    /// The loss function used for training.
    #[cfg(feature = "train")]
    loss: Option<BiRefNetLoss<B>>,
    /// Interpolation strategy for tensor resizing operations.
    interpolation_strategy: Ignored<InterpolationStrategy>,
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
        let [x1, x2, x3, x4] = match self.mul_scl_ipt.0 {
            MultiScaleInput::None => [x1, x2, x3, x4],
            MultiScaleInput::Add => {
                let [_, _, h, w] = x.dims();
                let [x1_, x2_, x3_, x4_] = self.bb.forward(intelligent_interpolate(
                    x,
                    [h / 2, w / 2],
                    &self.interpolation_strategy.0,
                ));

                let [_, _, h, w] = x1.dims();
                let x1 = x1 + intelligent_interpolate(x1_, [h, w], &self.interpolation_strategy.0);
                let [_, _, h, w] = x2.dims();
                let x2 = x2 + intelligent_interpolate(x2_, [h, w], &self.interpolation_strategy.0);
                let [_, _, h, w] = x3.dims();
                let x3 = x3 + intelligent_interpolate(x3_, [h, w], &self.interpolation_strategy.0);
                let [_, _, h, w] = x4.dims();
                let x4 = x4 + intelligent_interpolate(x4_, [h, w], &self.interpolation_strategy.0);

                [x1, x2, x3, x4]
            }
            MultiScaleInput::Cat => {
                let [_, _, h, w] = x.dims();
                let [x1_, x2_, x3_, x4_] = self.bb.forward(intelligent_interpolate(
                    x,
                    [h / 2, w / 2],
                    &self.interpolation_strategy.0,
                ));

                let [_, _, h, w] = x1.dims();
                let x1 = Tensor::cat(
                    vec![
                        x1,
                        intelligent_interpolate(x1_, [h, w], &self.interpolation_strategy.0),
                    ],
                    1,
                );
                let [_, _, h, w] = x2.dims();
                let x2 = Tensor::cat(
                    vec![
                        x2,
                        intelligent_interpolate(x2_, [h, w], &self.interpolation_strategy.0),
                    ],
                    1,
                );
                let [_, _, h, w] = x3.dims();
                let x3 = Tensor::cat(
                    vec![
                        x3,
                        intelligent_interpolate(x3_, [h, w], &self.interpolation_strategy.0),
                    ],
                    1,
                );
                let [_, _, h, w] = x4.dims();
                let x4 = Tensor::cat(
                    vec![
                        x4,
                        intelligent_interpolate(x4_, [h, w], &self.interpolation_strategy.0),
                    ],
                    1,
                );

                [x1, x2, x3, x4]
            }
        };

        if self.cxt[0] > 0 {
            let mut combined = vec![];
            let [_, _, h, w] = x4.dims();
            combined.push(intelligent_interpolate(
                x1.clone(),
                [h, w],
                &InterpolationStrategy::Nearest,
            ));
            combined.push(intelligent_interpolate(
                x2.clone(),
                [h, w],
                &InterpolationStrategy::Nearest,
            ));
            combined.push(intelligent_interpolate(
                x3.clone(),
                [h, w],
                &InterpolationStrategy::Nearest,
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
        if self.refine.is_none() {
            return Ok(decoder_output);
        }

        let refined_output = match self.refine.as_ref().unwrap() {
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
        self.forward_ori(x)
    }

    /// Forward pass for training with configurable loss computation.
    ///
    /// This method combines the model's segmentation output with ground truth
    /// to compute comprehensive training losses. The loss configuration can be
    /// customized for different training scenarios.
    ///
    /// # Arguments
    /// * `batch` - Training batch containing input images and ground truth masks
    /// * `loss_config` - Optional loss configuration (uses default if None)
    ///
    /// # Returns
    /// A result containing the training output with predictions, targets, and computed loss
    #[cfg(feature = "train")]
    pub fn forward_classification(
        &self,
        batch: BiRefNetBatch<B>,
    ) -> BiRefNetResult<BiRefNetOutput<B>> {
        // 1. Forward pass through the model to get predictions
        let prediction = self.forward(batch.images.clone())?;

        let loss = if let Some(loss_config) = &self.loss {
            loss_config
        } else {
            return Err(BiRefNetError::InvalidConfiguration {
                reason: "Loss configuration is not set".to_owned(),
            });
        };

        let loss = loss
            .forward(vec![prediction.clone()], batch.masks.clone())
            .map_err(|e| {
                BiRefNetError::General {
                    message: format!("Loss computation failed: {e}"),
                }
            })?;

        // 4. Create output item
        Ok(BiRefNetOutput::new(loss, prediction, batch.masks))
    }
}

#[cfg(feature = "train")]
impl<B: AutodiffBackend> TrainStep<BiRefNetBatch<B>, BiRefNetOutput<B>> for BiRefNet<B> {
    fn step(&self, batch: BiRefNetBatch<B>) -> TrainOutput<BiRefNetOutput<B>> {
        // Use default loss configuration for training
        let item = self.forward_classification(batch).unwrap();
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

#[cfg(feature = "train")]
impl<B: Backend> ValidStep<BiRefNetBatch<B>, BiRefNetOutput<B>> for BiRefNet<B> {
    fn step(&self, batch: BiRefNetBatch<B>) -> BiRefNetOutput<B> {
        // Use default loss configuration for validation
        self.forward_classification(batch).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use birefnet_loss::BiRefNetLossConfig;
    use burn::{
        tensor::{Distribution, Tensor},
        train::{TrainStep, ValidStep},
    };

    use super::BiRefNetConfig;
    #[cfg(feature = "train")]
    use crate::BiRefNetBatch;
    use crate::{
        config::{InterpolationStrategy, ModelConfig},
        tests::{TestAutodiffBackend, TestBackend},
    };

    #[cfg(not(feature = "train"))]
    #[test]
    fn birefnet_forward_produces_correct_output_shape() {
        let device = Default::default();
        let config = BiRefNetConfig::new(ModelConfig::new(InterpolationStrategy::Nearest));
        let model = config.init::<TestBackend>(&device).unwrap();

        // Create test input tensor [batch_size=1, channels=3, height=64, width=64]
        let input = Tensor::<TestBackend, 4>::random(
            [1, 3, 64, 64],
            Distribution::Normal(0.0, 1.0),
            &device,
        );

        // Test forward pass
        let output = model.forward(input);
        assert!(output.is_ok());

        let output_tensor = output.unwrap();
        // Output should have shape [1, 1, 64, 64] for segmentation
        assert_eq!(output_tensor.dims(), [1, 1, 64, 64]);
    }

    #[cfg(feature = "train")]
    #[test]
    fn birefnet_train_step_produces_finite_loss() {
        let device = Default::default();
        let config = BiRefNetConfig::new(ModelConfig::new(InterpolationStrategy::Nearest))
            .with_loss_config(Option::from(BiRefNetLossConfig::new()));
        let model = config.init::<TestAutodiffBackend>(&device).unwrap();

        // Create test batch
        let images = Tensor::<TestAutodiffBackend, 4>::random(
            [2, 3, 64, 64],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let masks = Tensor::<TestAutodiffBackend, 4>::random(
            [2, 1, 64, 64],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );

        let batch = BiRefNetBatch { images, masks };

        // Test training step
        let train_output = TrainStep::step(&model, batch);

        // Should have valid output
        assert!(train_output.item.loss.into_scalar().is_finite());
        assert_eq!(train_output.item.output.dims(), [2, 1, 64, 64]);
        assert_eq!(train_output.item.targets.dims(), [2, 1, 64, 64]);
    }

    #[cfg(feature = "train")]
    #[test]
    fn birefnet_valid_step_produces_finite_loss() {
        let device = Default::default();
        let config = BiRefNetConfig::new(ModelConfig::new(InterpolationStrategy::Nearest))
            .with_loss_config(Option::from(BiRefNetLossConfig::new()));
        let model = config.init::<TestBackend>(&device).unwrap();

        // Create test batch
        let images = Tensor::<TestBackend, 4>::random(
            [2, 3, 64, 64],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let masks = Tensor::<TestBackend, 4>::random(
            [2, 1, 64, 64],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );

        let batch = BiRefNetBatch { images, masks };

        // Test validation step
        let output = ValidStep::step(&model, batch);

        // Should have valid output
        assert!(output.loss.into_scalar().is_finite());
        assert_eq!(output.output.dims(), [2, 1, 64, 64]);
        assert_eq!(output.targets.dims(), [2, 1, 64, 64]);
    }
}
