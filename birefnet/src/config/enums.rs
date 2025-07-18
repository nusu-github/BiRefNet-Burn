//! Enumeration types for BiRefNet configuration.
//!
//! This module contains all the enumeration types that define various
//! configuration options for the BiRefNet model.

use burn::prelude::*;

/// Defines the target dataset or task.
#[derive(Config, Debug)]
pub enum Task {
    /// Dichotomous Image Segmentation 5K dataset.
    DIS5K,
    /// Camouflaged Object Detection dataset.
    COD,
    /// High-Resolution Salient Object Detection dataset.
    HRSOD,
    /// General purpose segmentation.
    General,
    /// General purpose segmentation (2k version).
    General2k,
    /// Image Matting task.
    Matting,
}

/// Defines the type of prompt for localization.
#[derive(Config, Debug)]
pub enum Prompt4loc {
    /// Dense prompt.
    Dense,
    /// Sparse prompt.
    Sparse,
}

/// Defines the method for combining multi-scale inputs.
#[derive(Config, Debug, PartialEq, Eq)]
pub enum MulSclIpt {
    /// No multi-scale input.
    None,
    /// Add multi-scale features.
    Add,
    /// Concatenate multi-scale features.
    Cat,
}

/// Defines the attention mechanism used in the decoder.
#[derive(Config, Debug)]
pub enum DecAtt {
    /// No attention mechanism.
    None,
    /// Atrous Spatial Pyramid Pooling.
    ASPP,
    /// ASPP with Deformable Convolutions.
    ASPPDeformable,
}

/// Defines the type and number of blocks in the squeeze module.
#[derive(Config, Debug, PartialEq, Eq)]
pub enum SqueezeBlock {
    /// No squeeze block.
    None,
    /// A sequence of `BasicDecBlk`.
    BasicDecBlk(usize),
    /// A sequence of `ResBlk`.
    ResBlk(usize),
    /// A sequence of `ASPP` blocks.
    ASPP(usize),
    /// A sequence of `ASPPDeformable` blocks.
    ASPPDeformable(usize),
}

impl SqueezeBlock {
    /// Returns the number of blocks configured for the squeeze module.
    #[must_use]
    pub const fn count(&self) -> usize {
        match self {
            Self::None => 0,
            Self::BasicDecBlk(x) | Self::ResBlk(x) | Self::ASPP(x) | Self::ASPPDeformable(x) => *x,
        }
    }
}

/// Defines the type of decoder block.
#[derive(Config, Debug)]
pub enum DecBlk {
    /// Basic decoder block.
    BasicDecBlk,
    /// Residual decoder block.
    ResBlk,
}

/// Defines the backbone architecture.
#[derive(Config, Debug, PartialEq, Eq)]
pub enum Backbone {
    /// VGG-16.
    Vgg16,
    /// VGG-16 with Batch Normalization.
    Vgg16bn,
    /// ResNet-50.
    Resnet50, // 0, 1, 2
    /// Swin Transformer Tiny.
    SwinV1T,
    /// Swin Transformer Small.
    SwinV1S, // 3, 4
    /// Swin Transformer Base.
    SwinV1B,
    /// Swin Transformer Large.
    SwinV1L, // 5-bs9, 6-bs4
    /// Pyramid Vision Transformer v2 B0.
    PvtV2B0,
    /// Pyramid Vision Transformer v2 B1.
    PvtV2B1, // 7, 8
    /// Pyramid Vision Transformer v2 B2.
    PvtV2B2,
    /// Pyramid Vision Transformer v2 B5.
    PvtV2B5, // 9-bs10, 10-bs5
}

/// Defines the type of lateral connection block.
#[derive(Config, Debug)]
pub enum LatBlk {
    /// Basic lateral block.
    BasicLatBlk,
}

/// Defines the strategy for intermediate channel sizes in the decoder.
#[derive(Config, Debug)]
pub enum DecChannelsInter {
    /// Fixed channel sizes.
    Fixed,
    /// Adaptive channel sizes.
    Adap,
}

/// Defines the refinement strategy.
#[derive(Config, Debug, PartialEq, Eq)]
pub enum Refine {
    /// No refinement.
    None,
    /// Refine with the model itself.
    Itself,
    /// Refine with a U-Net like structure.
    RefUNet,
    /// Refine with the `Refiner` module.
    Refiner,
    /// Refiner with 4 input channels for PVT backbone.
    RefinerPVTInChannels4,
}

/// Defines preprocessing methods.
#[derive(Config, Debug)]
pub enum PreprocMethods {
    Flip,
    Enhance,
    Rotate,
    Pepper,
    Crop,
}

/// Defines the optimizer type.
#[derive(Config, Debug)]
pub enum Optimizer {
    Adam,
    AdamW,
}
