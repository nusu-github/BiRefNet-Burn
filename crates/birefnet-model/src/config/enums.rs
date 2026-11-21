//! Enumeration types for `BiRefNet` configuration.

use burn::serde::{Deserialize, Serialize};

/// Interpolation strategy for tensor resizing operations.
///
/// Burn's autodiff backward pass supports only `Nearest` interpolation, whereas
/// inference can use `Bilinear` for smoother masks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(crate = "burn::serde")]
pub enum InterpolationStrategy {
    /// Bilinear interpolation (inference only).
    #[default]
    Bilinear,
    /// Nearest-neighbor interpolation (works for both training and inference).
    Nearest,
}

/// Target dataset or segmentation task preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(crate = "burn::serde")]
pub enum Task {
    /// Dichotomous Image Segmentation 5K dataset.
    DIS5K,
    /// Camouflaged Object Detection dataset.
    COD,
    /// High-Resolution Salient Object Detection dataset.
    HRSOD,
    /// General-purpose segmentation.
    General,
    /// General-purpose 2K resolution segmentation.
    General2k,
    /// Image matting task.
    Matting,
}

/// Prompt type used for localization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(crate = "burn::serde")]
pub enum PromptForLocation {
    /// Dense prompt.
    Dense,
    /// Sparse prompt.
    Sparse,
}

/// Method for combining multi-scale inputs in the encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(crate = "burn::serde")]
pub enum MultiScaleInput {
    /// Single-scale input only.
    None,
    /// Add half-resolution features to full-resolution features.
    Add,
    /// Concatenate half-resolution features with full-resolution features.
    Cat,
}

/// Attention mechanism used inside decoder and squeeze blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(crate = "burn::serde")]
pub enum DecoderAttention {
    /// No squeeze attention block.
    None,
    /// Atrous Spatial Pyramid Pooling (`ASPP`).
    ASPP,
    /// Deformable Atrous Spatial Pyramid Pooling (`ASPPDeformable`).
    ASPPDeformable,
}

/// Type and repetition count of blocks in the encoder-decoder squeeze module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(crate = "burn::serde")]
pub enum SqueezeBlock {
    /// No squeeze block.
    None,
    /// Sequence of `BasicDecBlk` squeeze blocks.
    BasicDecBlk(usize),
    /// Sequence of `ResBlk` squeeze blocks.
    ResBlk(usize),
    /// Sequence of `ASPP` squeeze blocks.
    ASPP(usize),
    /// Sequence of `ASPPDeformable` squeeze blocks.
    ASPPDeformable(usize),
}

impl SqueezeBlock {
    /// Returns the number of repeated blocks configured for the squeeze module.
    #[must_use]
    pub const fn count(&self) -> usize {
        match self {
            Self::None => 0,
            Self::BasicDecBlk(count)
            | Self::ResBlk(count)
            | Self::ASPP(count)
            | Self::ASPPDeformable(count) => *count,
        }
    }
}

/// Decoder block architecture variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(crate = "burn::serde")]
pub enum DecBlk {
    /// Basic decoder block.
    BasicDecBlk,
    /// Residual decoder block.
    ResBlk,
}

/// Backbone encoder architecture variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(crate = "burn::serde")]
pub enum Backbone {
    /// VGG-16 without batch normalization.
    Vgg16,
    /// VGG-16 with batch normalization.
    Vgg16bn,
    /// `ResNet-50`.
    Resnet50,
    /// Swin Transformer Tiny.
    SwinV1T,
    /// Swin Transformer Small.
    SwinV1S,
    /// Swin Transformer Base.
    SwinV1B,
    /// Swin Transformer Large.
    SwinV1L,
    /// Pyramid Vision Transformer v2 B0.
    PvtV2B0,
    /// Pyramid Vision Transformer v2 B1.
    PvtV2B1,
    /// Pyramid Vision Transformer v2 B2.
    PvtV2B2,
    /// Pyramid Vision Transformer v2 B5.
    PvtV2B5,
}

/// Lateral connection block architecture variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(crate = "burn::serde")]
pub enum LateralBlock {
    /// Single `1x1` convolution lateral block.
    BasicLatBlk,
}

/// Strategy for intermediate channel widths inside decoder blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(crate = "burn::serde")]
pub enum DecChannelsInter {
    /// Fixed intermediate width (`64` channels).
    Fixed,
    /// Adaptive intermediate width (`in_channels / 4`).
    Adap,
}

/// Output refinement module strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(crate = "burn::serde")]
pub enum Refine {
    /// No refinement stage.
    None,
    /// Self-refinement (reserved).
    Itself,
    /// U-Net refinement module (`RefUNet`).
    RefUNet,
    /// Stem + PVTv2 refinement module (`Refiner`).
    Refiner,
    /// 4-channel PVTv2 refinement module (`RefinerPVTInChannels4`).
    RefinerPVTInChannels4,
}

/// Optimizer selection for model training configurations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(crate = "burn::serde")]
pub enum Optimizer {
    /// Adam optimizer.
    Adam,
    /// `AdamW` decoupled weight-decay optimizer.
    AdamW,
}
