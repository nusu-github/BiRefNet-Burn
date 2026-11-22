//! Enumeration types for `BiRefNet` configuration.
//!
//! `Display` renders the option strings used by the upstream PyTorch `config.py`.

use std::fmt;

use burn::serde::{Deserialize, Serialize};

/// Interpolation strategy for tensor resizing operations.
///
/// The Flex backend differentiates `Nearest`, `Bilinear`, and `Bicubic` interpolation, but
/// CubeCL backends (`cpu`, `wgpu`, `cuda`, ...) implement the backward pass for `Nearest` only.
/// Use `Nearest` for training on those backends and `Bilinear` for smoother inference masks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(crate = "burn::serde")]
pub enum InterpolationStrategy {
    /// Bilinear interpolation (default; differentiable on Flex only).
    #[default]
    Bilinear,
    /// Nearest-neighbor interpolation (differentiable on every backend).
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
    #[serde(alias = "Vgg16bn")]
    Vgg16Bn,
    /// `ResNet-50`.
    #[serde(alias = "Resnet50")]
    ResNet50,
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
    /// Upstream "refine with itself" mode; not implemented and rejected by validation.
    Itself,
    /// U-Net refinement module (`RefUNet`).
    RefUNet,
    /// Stem + PVTv2 refinement module (`Refiner`).
    Refiner,
    /// 4-channel PVTv2 refinement module (`RefinerPVTInChannels4`).
    RefinerPVTInChannels4,
}

macro_rules! display_as {
    ($ty:ty { $($variant:pat => $name:expr),+ $(,)? }) => {
        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    $($variant => f.write_str($name)),+
                }
            }
        }
    };
}

display_as!(InterpolationStrategy {
    Self::Bilinear => "bilinear",
    Self::Nearest => "nearest",
});

display_as!(Task {
    Self::DIS5K => "DIS5K",
    Self::COD => "COD",
    Self::HRSOD => "HRSOD",
    Self::General => "General",
    Self::General2k => "General-2K",
    Self::Matting => "Matting",
});

display_as!(MultiScaleInput {
    Self::None => "none",
    Self::Add => "add",
    Self::Cat => "cat",
});

display_as!(DecoderAttention {
    Self::None => "none",
    Self::ASPP => "ASPP",
    Self::ASPPDeformable => "ASPPDeformable",
});

impl fmt::Display for SqueezeBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::None => return f.write_str("none"),
            Self::BasicDecBlk(_) => "BasicDecBlk",
            Self::ResBlk(_) => "ResBlk",
            Self::ASPP(_) => "ASPP",
            Self::ASPPDeformable(_) => "ASPPDeformable",
        };
        write!(f, "{name}_x{}", self.count())
    }
}

display_as!(DecBlk {
    Self::BasicDecBlk => "BasicDecBlk",
    Self::ResBlk => "ResBlk",
});

display_as!(Backbone {
    Self::Vgg16 => "vgg16",
    Self::Vgg16Bn => "vgg16bn",
    Self::ResNet50 => "resnet50",
    Self::SwinV1T => "swin_v1_t",
    Self::SwinV1S => "swin_v1_s",
    Self::SwinV1B => "swin_v1_b",
    Self::SwinV1L => "swin_v1_l",
    Self::PvtV2B0 => "pvt_v2_b0",
    Self::PvtV2B1 => "pvt_v2_b1",
    Self::PvtV2B2 => "pvt_v2_b2",
    Self::PvtV2B5 => "pvt_v2_b5",
});

display_as!(LateralBlock {
    Self::BasicLatBlk => "BasicLatBlk",
});

display_as!(DecChannelsInter {
    Self::Fixed => "fixed",
    Self::Adap => "adap",
});

display_as!(Refine {
    Self::None => "none",
    Self::Itself => "itself",
    Self::RefUNet => "RefUNet",
    Self::Refiner => "Refiner",
    Self::RefinerPVTInChannels4 => "RefinerPVTInChannels4",
});
