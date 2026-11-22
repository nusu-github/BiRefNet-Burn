//! Core configuration structures for `BiRefNet`.

use burn::prelude::*;

use super::enums::{
    Backbone, DecBlk, DecChannelsInter, DecoderAttention, InterpolationStrategy, LateralBlock,
    MultiScaleInput, Refine, SqueezeBlock, Task,
};
use crate::error::ModelError;

/// Top-level configuration for the `BiRefNet` model.
#[derive(Config, Debug, PartialEq, Eq, Hash)]
pub struct ModelConfig {
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
    /// Interpolation strategy for tensor resizing operations.
    pub interpolation: InterpolationStrategy,
}

/// Task-related configuration.
#[derive(Config, Debug, PartialEq, Eq, Hash)]
pub struct TaskConfig {
    /// The target dataset or task preset (for example `DIS5K` or `COD`).
    #[config(default = "Task::DIS5K")]
    pub task: Task,
    /// Batch size for training/inference; controls `BatchNorm` vs identity normalization.
    #[config(default = 4)]
    pub batch_size: usize,
}

/// Backbone network configuration.
#[derive(Config, Debug, PartialEq, Eq, Hash)]
pub struct BackboneConfig {
    /// Chosen backbone encoder architecture.
    #[config(default = "Backbone::SwinV1L")]
    pub backbone: Backbone,
}

/// Decoder module configuration.
#[derive(Config, Debug, PartialEq, Eq, Hash)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "mirrors upstream BiRefNet boolean decoder configuration flags"
)]
pub struct DecoderConfig {
    /// Enable multi-scale supervision outputs during training.
    #[config(default = true)]
    pub ms_supervision: bool,
    /// Enable gradient-guided output refinement; requires `ms_supervision`.
    #[config(default = true)]
    pub out_ref: bool,
    /// Enable image-pyramid input branches in the decoder.
    #[config(default = true)]
    pub dec_ipt: bool,
    /// Split the decoder image input into spatial patches; requires `dec_ipt`.
    #[config(default = true)]
    pub dec_ipt_split: bool,
    /// Number of encoder context features concatenated into the deepest stage (`0..=3`).
    #[config(default = 3)]
    pub cxt_num: usize,
    /// Method for handling multi-scale encoder inputs.
    #[config(default = "MultiScaleInput::Cat")]
    pub mul_scl_ipt: MultiScaleInput,
    /// Attention mechanism inside decoder and squeeze blocks.
    #[config(default = "DecoderAttention::ASPPDeformable")]
    pub dec_att: DecoderAttention,
    /// Squeeze block architecture and depth.
    #[config(default = "SqueezeBlock::BasicDecBlk(1)")]
    pub squeeze_block: SqueezeBlock,
    /// Decoder block architecture.
    #[config(default = "DecBlk::BasicDecBlk")]
    pub dec_blk: DecBlk,
    /// Lateral connection block architecture.
    #[config(default = "LateralBlock::BasicLatBlk")]
    pub lat_blk: LateralBlock,
    /// Intermediate channel sizing strategy inside decoder blocks.
    #[config(default = "DecChannelsInter::Fixed")]
    pub dec_channels_inter: DecChannelsInter,
}

/// Refinement module configuration.
#[derive(Config, Debug, PartialEq, Eq, Hash)]
pub struct RefineConfig {
    /// Refinement strategy appended after the decoder.
    #[config(default = "Refine::None")]
    pub refine: Refine,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for BackboneConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for RefineConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelConfig {
    /// Validates logical consistency across model configuration fields.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `cxt_num > 3`, `squeeze_block.count() > 10`,
    /// dependent flags (`out_ref` without `ms_supervision`, `dec_ipt_split` without `dec_ipt`)
    /// are inconsistent, `dec_att` mismatches `squeeze_block`, or `refine` is the
    /// unimplemented [`Refine::Itself`].
    pub fn validate(&self) -> Result<(), ModelError> {
        // Upstream BiRefNet supports only 0..=3 context stages.
        if self.decoder.cxt_num > 3 {
            return Err(ModelError::invalid_config(format!(
                "context number must be <= 3, got {}",
                self.decoder.cxt_num
            )));
        }

        if self.decoder.squeeze_block.count() > 10 {
            return Err(ModelError::invalid_config(format!(
                "squeeze block count is too high: {}",
                self.decoder.squeeze_block.count()
            )));
        }

        // `out_ref` depends on `ms_supervision`.
        if self.decoder.out_ref && !self.decoder.ms_supervision {
            return Err(ModelError::invalid_config(
                "out_ref can only be enabled when ms_supervision is true",
            ));
        }

        if self.decoder.dec_ipt_split && !self.decoder.dec_ipt {
            return Err(ModelError::invalid_config(
                "dec_ipt_split can only be enabled when dec_ipt is true",
            ));
        }

        if self.refine.refine == Refine::Itself {
            return Err(ModelError::invalid_config(
                "refine = itself is not implemented; use none, RefUNet, Refiner, or RefinerPVTInChannels4",
            ));
        }

        if self.decoder.squeeze_block != SqueezeBlock::None {
            match self.decoder.dec_att {
                DecoderAttention::None => {
                    if matches!(
                        self.decoder.squeeze_block,
                        SqueezeBlock::ASPP(_) | SqueezeBlock::ASPPDeformable(_)
                    ) {
                        return Err(ModelError::invalid_config(
                            "dec_att cannot be None when an ASPP squeeze_block is selected",
                        ));
                    }
                }
                DecoderAttention::ASPP => {
                    if !matches!(self.decoder.squeeze_block, SqueezeBlock::ASPP(_)) {
                        return Err(ModelError::invalid_config(
                            "dec_att ASPP should be used with SqueezeBlock::ASPP",
                        ));
                    }
                }
                DecoderAttention::ASPPDeformable => {
                    if !matches!(
                        self.decoder.squeeze_block,
                        SqueezeBlock::ASPPDeformable(_) | SqueezeBlock::BasicDecBlk(_)
                    ) {
                        return Err(ModelError::invalid_config(
                            "dec_att ASPPDeformable requires SqueezeBlock::ASPPDeformable or BasicDecBlk",
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    /// Returns decoder-ordered `[x4, x3, x2, x1]` channel counts after multi-scale input fusion.
    #[must_use]
    pub fn lateral_channels_in_collection(&self) -> [usize; 4] {
        let channels = self.backbone.lateral_channels_in_collection();
        if self.decoder.mul_scl_ipt == MultiScaleInput::Cat {
            let [c4, c3, c2, c1] = channels;
            [c4 * 2, c3 * 2, c2 * 2, c1 * 2]
        } else {
            channels
        }
    }

    /// Returns context channel counts `[c1, c2, c3]` active for skip concatenation into `x4`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `cxt_num > 3`.
    pub fn cxt(&self) -> Result<[usize; 3], ModelError> {
        let lateral_channels = self.lateral_channels_in_collection();
        self.decoder.cxt(&lateral_channels)
    }
}

impl BackboneConfig {
    /// Returns raw backbone output channels ordered deep-to-shallow `[x4, x3, x2, x1]`.
    #[must_use]
    pub const fn lateral_channels_in_collection(&self) -> [usize; 4] {
        match self.backbone {
            Backbone::Vgg16 | Backbone::Vgg16Bn => [512, 512, 256, 128],
            Backbone::ResNet50 => [2048, 1024, 512, 256],
            Backbone::SwinV1T | Backbone::SwinV1S => [768, 384, 192, 96],
            Backbone::SwinV1B => [1024, 512, 256, 128],
            Backbone::SwinV1L => [1536, 768, 384, 192],
            Backbone::PvtV2B0 => [256, 160, 64, 32],
            Backbone::PvtV2B1 | Backbone::PvtV2B2 | Backbone::PvtV2B5 => [512, 320, 128, 64],
        }
    }
}

impl DecoderConfig {
    /// Returns context channel counts for the deepest stage given `[x4, x3, x2, x1]` widths.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `cxt_num > 3`.
    pub fn cxt(&self, lateral_channels: &[usize; 4]) -> Result<[usize; 3], ModelError> {
        match self.cxt_num {
            0 => Ok([0, 0, 0]),
            1 => Ok([0, 0, lateral_channels[1]]),
            2 => Ok([0, lateral_channels[2], lateral_channels[1]]),
            3 => Ok([
                lateral_channels[3],
                lateral_channels[2],
                lateral_channels[1],
            ]),
            other => Err(ModelError::invalid_config(format!(
                "context number must be <= 3, got {other}"
            ))),
        }
    }
}
