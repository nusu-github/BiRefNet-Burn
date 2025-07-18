//! Core configuration structures for BiRefNet.
//!
//! This module contains the primary configuration structures that define
//! the BiRefNet model architecture and behavior.

use std::path::PathBuf;

use crate::error::{BiRefNetError, BiRefNetResult};
use burn::prelude::*;

use super::enums::*;

/// Main configuration for the BiRefNet model.
///
/// This struct aggregates all other configuration modules for the model,
/// including paths, task-specific settings, backbone, decoder, and refinement options.
#[derive(Config, Debug)]
pub struct ModelConfig {
    /// Path-related configurations.
    #[config(default = "PathConfig::new()")]
    pub path: PathConfig,
    /// Task-related configurations.
    #[config(default = "TaskConfig::new()")]
    pub task: TaskConfig,
    /// Backbone network configuration.
    #[config(default = "BackboneConfig::new()")]
    pub backbone: BackboneConfig,
    /// Decoder module configuration.
    #[config(default = "DecoderConfig::new()")]
    pub decoder: DecoderConfig,
    /// Refinement module configuration.
    #[config(default = "RefineConfig::new()")]
    pub refine: RefineConfig,
}

/// Path-related configuration.
#[derive(Config, Debug)]
pub struct PathConfig {
    /// System home directory.
    #[config(default = "None")]
    pub sys_home_dir: Option<PathBuf>,
    /// Root directory for the datasets.
    #[config(default = "PathBuf::from(\"datasets/dis\")")]
    pub data_root_dir: PathBuf,
}

/// Task-related configuration.
#[derive(Config, Debug)]
pub struct TaskConfig {
    /// The specific task the model is configured for (e.g., DIS5K, COD).
    #[config(default = "Task::DIS5K")]
    pub task: Task,
    /// The type of prompt used for localization.
    #[config(default = "Prompt4loc::Dense")]
    pub prompt4_loc: Prompt4loc,
    /// Batch size for training/inference. Controls BatchNorm vs Identity usage.
    #[config(default = "4")]
    pub batch_size: usize,
}

/// Backbone network configuration.
#[derive(Config, Debug)]
pub struct BackboneConfig {
    /// The chosen backbone architecture (e.g., SwinV1L, Resnet50).
    #[config(default = "Backbone::SwinV1L")]
    pub backbone: Backbone,
}

/// Decoder-related configuration.
///
/// Controls the behavior of the decoder part of the network, including supervision strategies,
/// input handling, attention mechanisms, and block types.
#[derive(Config, Debug)]
pub struct DecoderConfig {
    /// Enable multi-scale supervision during training.
    #[config(default = "true")]
    pub ms_supervision: bool,
    /// Enable output refinement. Depends on `ms_supervision`.
    #[config(default = "true")]
    pub out_ref: bool,
    /// Enable decoder input from the main input image.
    #[config(default = "true")]
    pub dec_ipt: bool,
    /// Split the decoder input. Depends on `dec_ipt`.
    #[config(default = "true")]
    pub dec_ipt_split: bool,
    /// Number of context features to use from earlier encoder stages. Range: [0, 3].
    #[config(default = "3")]
    pub cxt_num: usize,
    /// Method for handling multi-scale input ('Cat', 'Add', or 'None').
    #[config(default = "MulSclIpt::Cat")]
    pub mul_scl_ipt: MulSclIpt,
    /// Type of attention mechanism in the decoder.
    #[config(default = "DecAtt::ASPPDeformable")]
    pub dec_att: DecAtt,
    /// The block type and count for the squeeze module.
    #[config(default = "SqueezeBlock::BasicDecBlk(1)")]
    pub squeeze_block: SqueezeBlock,
    /// The type of block to use in the decoder.
    #[config(default = "DecBlk::BasicDecBlk")]
    pub dec_blk: DecBlk,
    /// The type of lateral connection block to use.
    #[config(default = "LatBlk::BasicLatBlk")]
    pub lat_blk: LatBlk,
    /// Strategy for intermediate channel sizes in the decoder.
    #[config(default = "DecChannelsInter::Fixed")]
    pub dec_channels_inter: DecChannelsInter,
}

/// Refinement-related configuration.
#[derive(Config, Debug)]
pub struct RefineConfig {
    /// The refinement strategy to use.
    #[config(default = "Refine::None")]
    pub refine: Refine,
}

impl ModelConfig {
    /// Validate the configuration and return appropriate errors for invalid settings.
    ///
    /// This function checks for logical inconsistencies and unsupported features based on
    /// constraints from the original PyTorch implementation.
    ///
    /// # Errors
    ///
    /// Returns `Err(BiRefNetError::InvalidConfiguration)` if any validation rule is violated.
    /// Returns `Err(BiRefNetError::UnsupportedBackbone)` if the backbone is not implemented.
    pub fn validate(&self) -> BiRefNetResult<()> {
        // 1. Check if backbone is supported (only Swin Transformer variants currently implemented)
        match self.backbone.backbone {
            Backbone::SwinV1T | Backbone::SwinV1S | Backbone::SwinV1B | Backbone::SwinV1L => {
                // These are supported
            }
            _ => {
                return Err(BiRefNetError::UnsupportedBackbone {
                    backbone: format!("{:?}", self.backbone.backbone),
                })
            }
        }

        // 2. Check context number is valid (本家では [0, 3] のみ)
        if self.decoder.cxt_num > 3 {
            return Err(BiRefNetError::InvalidConfiguration {
                reason: format!("Context number must be <= 3, got {}", self.decoder.cxt_num),
            });
        }

        // 3. Check squeeze block count is reasonable
        if self.decoder.squeeze_block.count() > 10 {
            return Err(BiRefNetError::InvalidConfiguration {
                reason: format!(
                    "Squeeze block count is too high: {}",
                    self.decoder.squeeze_block.count()
                ),
            });
        }

        // 4. Check logical dependencies (similar to original implementation)
        // out_ref depends on ms_supervision (本家: self.out_ref = self.ms_supervision and True)
        if self.decoder.out_ref && !self.decoder.ms_supervision {
            return Err(BiRefNetError::InvalidConfiguration {
                reason: "out_ref can only be enabled when ms_supervision is true".to_string(),
            });
        }

        // 5. Check dec_ipt_split depends on dec_ipt
        if self.decoder.dec_ipt_split && !self.decoder.dec_ipt {
            return Err(BiRefNetError::InvalidConfiguration {
                reason: "dec_ipt_split can only be enabled when dec_ipt is true".to_string(),
            });
        }

        // 6. Check squeeze_block compatibility with dec_att
        if self.decoder.squeeze_block != SqueezeBlock::None {
            match self.decoder.dec_att {
                DecAtt::None => {
                    return Err(BiRefNetError::InvalidConfiguration {
                        reason: "dec_att should not be None when squeeze_block is enabled"
                            .to_string(),
                    });
                }
                DecAtt::ASPP => {
                    if !matches!(self.decoder.squeeze_block, SqueezeBlock::ASPP(_)) {
                        return Err(BiRefNetError::InvalidConfiguration {
                            reason: "dec_att ASPP should be used with SqueezeBlock::ASPP"
                                .to_string(),
                        });
                    }
                }
                DecAtt::ASPPDeformable => {
                    if !matches!(
                        self.decoder.squeeze_block,
                        SqueezeBlock::ASPPDeformable(_) | SqueezeBlock::BasicDecBlk(_)
                    ) {
                        return Err(BiRefNetError::InvalidConfiguration {
                            reason: "dec_att ASPPDeformable should be used with SqueezeBlock::ASPPDeformable or BasicDecBlk".to_string()
                        });
                    }
                }
            }
        }

        // 7. Check refine configuration
        if self.refine.refine != Refine::None {
            return Err(BiRefNetError::InvalidConfiguration {
                reason: "Refine is not yet implemented".to_string(),
            });
        }

        Ok(())
    }

    /// Get lateral channels for the backbone network.
    ///
    /// The number of channels depends on the chosen backbone and whether multi-scale
    /// inputs are concatenated (`MulSclIpt::Cat`).
    #[must_use]
    pub fn lateral_channels_in_collection(&self) -> [usize; 4] {
        let channels = match self.backbone.backbone {
            Backbone::Vgg16 | Backbone::Vgg16bn => [512, 256, 128, 64],
            Backbone::Resnet50 => [1024, 512, 256, 64],
            Backbone::SwinV1T | Backbone::SwinV1S => [768, 384, 192, 96],
            Backbone::SwinV1B => [1024, 512, 256, 128],
            Backbone::SwinV1L => [1536, 768, 384, 192],
            Backbone::PvtV2B0 => [256, 160, 64, 32],
            Backbone::PvtV2B1 | Backbone::PvtV2B2 | Backbone::PvtV2B5 => [512, 320, 128, 64],
        };
        if self.decoder.mul_scl_ipt == MulSclIpt::Cat {
            let [c1, c2, c3, c4] = channels;
            [c1 * 2, c2 * 2, c3 * 2, c4 * 2]
        } else {
            channels
        }
    }

    /// Get context channels for skip connections.
    ///
    /// Retrieves the channel counts from the last `cxt_num` encoder features
    /// (excluding the final one) to be used as context in the decoder.
    ///
    /// # Errors
    ///
    /// Returns `Err(BiRefNetError::InvalidConfiguration)` if `cxt_num` is invalid.
    #[must_use]
    pub fn cxt(&self) -> BiRefNetResult<[usize; 3]> {
        if self.decoder.cxt_num > 0 {
            let reversed: Vec<usize> = self.lateral_channels_in_collection()[1..]
                .iter()
                .rev()
                .copied()
                .collect();
            reversed[reversed.len().saturating_sub(self.decoder.cxt_num)..]
                .try_into()
                .map_err(|_| BiRefNetError::InvalidConfiguration {
                    reason: format!("Invalid context number: {}", self.decoder.cxt_num),
                })
        } else {
            Ok([0, 0, 0])
        }
    }
}

impl BackboneConfig {
    /// Get lateral channels for the backbone network.
    #[must_use]
    pub const fn lateral_channels_in_collection(&self) -> [usize; 4] {
        match self.backbone {
            Backbone::Vgg16 | Backbone::Vgg16bn => [512, 256, 128, 64],
            Backbone::Resnet50 => [1024, 512, 256, 64],
            Backbone::SwinV1T | Backbone::SwinV1S => [768, 384, 192, 96],
            Backbone::SwinV1B => [1024, 512, 256, 128],
            Backbone::SwinV1L => [1536, 768, 384, 192],
            Backbone::PvtV2B0 => [256, 160, 64, 32],
            Backbone::PvtV2B1 | Backbone::PvtV2B2 | Backbone::PvtV2B5 => [512, 320, 128, 64],
        }
    }
}

impl DecoderConfig {
    /// Get context channels for skip connections.
    #[must_use]
    pub fn cxt(&self, lateral_channels: &[usize; 4]) -> BiRefNetResult<[usize; 3]> {
        if self.cxt_num > 0 {
            let reversed: Vec<usize> = lateral_channels[1..].iter().rev().copied().collect();
            reversed[reversed.len().saturating_sub(self.cxt_num)..]
                .try_into()
                .map_err(|_| BiRefNetError::InvalidConfiguration {
                    reason: format!("Invalid context number: {}", self.cxt_num),
                })
        } else {
            Ok([0, 0, 0])
        }
    }
}
