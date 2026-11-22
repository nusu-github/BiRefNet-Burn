//! Multi-scale encoder backbones for `BiRefNet` on Burn.
//!
//! | Family | Variants | Configuration |
//! | --- | --- | --- |
//! | Swin Transformer v1 | [`SwinVariant`]: T, S, B, L | [`SwinTransformerConfig`] |
//! | Pyramid Vision Transformer v2 | [`PvtV2Variant`]: B0–B5 | [`PvtV2Config`] |
//! | `ResNet` | [`ResNetVariant`]: 18, 34, 50, 101, 152 | [`ResNetConfig`] |
//! | VGG | [`VggVariant`]: 16, 16-BN | [`VggConfig`] |
//!
//! Every backbone implements [`Backbone`] and returns four feature maps at strides
//! `4, 8, 16, 32`, shallowest first. [`BackboneType`] selects a preset and builds a
//! [`BackboneModel`]; configuration or shape issues surface as [`BackboneError`].
//!
//! # Examples
//!
//! ```rust
//! use birefnet_backbones::{Backbone, BackboneType, PvtV2Variant};
//! use burn::prelude::*;
//!
//! let device = Device::flex();
//! let backbone = BackboneType::PvtV2(PvtV2Variant::B0).init(&device)?;
//! let features = backbone.forward(Tensor::zeros([1, 3, 64, 64], &device));
//!
//! assert_eq!(features[0].dims(), [1, 32, 16, 16]);
//! assert_eq!(features[3].dims()[1], backbone.output_channels()[3]);
//! # Ok::<(), birefnet_backbones::BackboneError>(())
//! ```
//!
//! `PVTv2` accepts any number of input channels, which the refiner uses for RGB + mask input:
//!
//! ```rust
//! use birefnet_backbones::{PvtV2Config, PvtV2Variant};
//! use burn::prelude::*;
//!
//! let device = Device::flex();
//! let pvt = PvtV2Config::from_variant(PvtV2Variant::B0, 4).init(&device)?;
//! let [stride4, ..] = pvt.forward(Tensor::zeros([1, 4, 64, 64], &device));
//! assert_eq!(stride4.dims(), [1, 32, 16, 16]);
//! # Ok::<(), birefnet_backbones::BackboneError>(())
//! ```

mod backbones;
mod error;

use std::fmt;

use burn::{
    prelude::*,
    serde::{Deserialize, Serialize},
};

#[doc(inline)]
pub use backbones::{
    pvt_v2::{PvtV2Config, PyramidVisionTransformerImpr},
    resnet::{ResNetBackbone, ResNetConfig},
    swin::{SwinTransformer, SwinTransformerConfig},
    vgg::{VggBackbone, VggConfig},
};
#[doc(inline)]
pub use error::BackboneError;

/// Unified interface implemented by every multi-scale `BiRefNet` feature extractor.
pub trait Backbone {
    /// Extracts 4 hierarchical feature maps from `[B, 3, H, W]` ordered shallowest to deepest.
    fn forward(&self, input: Tensor<4>) -> [Tensor<4>; 4];

    /// Returns the output channel counts `[C1, C2, C3, C4]` for the 4 returned feature maps.
    fn output_channels(&self) -> [usize; 4];
}

impl Backbone for ResNetBackbone {
    fn forward(&self, input: Tensor<4>) -> [Tensor<4>; 4] {
        self.forward(input)
    }

    fn output_channels(&self) -> [usize; 4] {
        self.output_channels()
    }
}

impl Backbone for VggBackbone {
    fn forward(&self, input: Tensor<4>) -> [Tensor<4>; 4] {
        self.forward(input)
    }

    fn output_channels(&self) -> [usize; 4] {
        self.output_channels()
    }
}

impl Backbone for SwinTransformer {
    fn forward(&self, input: Tensor<4>) -> [Tensor<4>; 4] {
        let input_dims = input.dims();
        self.forward(input).unwrap_or_else(|err| {
            panic!("SwinTransformer forward pass failed with input shape {input_dims:?}: {err}")
        })
    }

    fn output_channels(&self) -> [usize; 4] {
        self.output_channels()
    }
}

impl Backbone for PyramidVisionTransformerImpr {
    fn forward(&self, input: Tensor<4>) -> [Tensor<4>; 4] {
        self.forward(input)
    }

    fn output_channels(&self) -> [usize; 4] {
        self.output_channels()
    }
}

/// Available backbone architecture families and variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(crate = "burn::serde")]
pub enum BackboneType {
    /// `ResNet` family backbone.
    ResNet(ResNetVariant),
    /// VGG family backbone.
    Vgg(VggVariant),
    /// Swin Transformer v1 family backbone.
    SwinTransformer(SwinVariant),
    /// Pyramid Vision Transformer v2 family backbone.
    PvtV2(PvtV2Variant),
}

impl BackboneType {
    /// Initializes the selected [`BackboneModel`] on `device`.
    ///
    /// # Errors
    ///
    /// Returns [`BackboneError`] if the underlying backbone configuration is invalid.
    pub fn init(self, device: &Device) -> Result<BackboneModel, BackboneError> {
        match self {
            Self::ResNet(variant) => {
                let model = ResNetConfig::from_variant(variant).init(device)?;
                Ok(BackboneModel::ResNet(model))
            }
            Self::Vgg(variant) => {
                let model = VggConfig::from_variant(variant).init(device);
                Ok(BackboneModel::Vgg(model))
            }
            Self::SwinTransformer(variant) => {
                let config = SwinTransformerConfig::from_variant(variant);
                Ok(BackboneModel::SwinTransformer(config.init(device)?))
            }
            Self::PvtV2(variant) => {
                let config = PvtV2Config::from_variant(variant, 3);
                Ok(BackboneModel::PvtV2(config.init(device)?))
            }
        }
    }
}

impl fmt::Display for BackboneType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ResNet(variant) => variant.fmt(f),
            Self::Vgg(variant) => variant.fmt(f),
            Self::SwinTransformer(variant) => variant.fmt(f),
            Self::PvtV2(variant) => variant.fmt(f),
        }
    }
}

impl fmt::Display for ResNetVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::ResNet18 => "resnet18",
            Self::ResNet34 => "resnet34",
            Self::ResNet50 => "resnet50",
            Self::ResNet101 => "resnet101",
            Self::ResNet152 => "resnet152",
        })
    }
}

impl fmt::Display for VggVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Vgg16 => "vgg16",
            Self::Vgg16Bn => "vgg16bn",
        })
    }
}

impl fmt::Display for SwinVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::SwinT => "swin_v1_t",
            Self::SwinS => "swin_v1_s",
            Self::SwinB => "swin_v1_b",
            Self::SwinL => "swin_v1_l",
        })
    }
}

impl fmt::Display for PvtV2Variant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::B0 => "pvt_v2_b0",
            Self::B1 => "pvt_v2_b1",
            Self::B2 => "pvt_v2_b2",
            Self::B3 => "pvt_v2_b3",
            Self::B4 => "pvt_v2_b4",
            Self::B5 => "pvt_v2_b5",
        })
    }
}

/// Supported `ResNet` depths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(crate = "burn::serde")]
pub enum ResNetVariant {
    /// ResNet-18.
    ResNet18,
    /// ResNet-34.
    ResNet34,
    /// ResNet-50.
    ResNet50,
    /// ResNet-101.
    ResNet101,
    /// ResNet-152.
    ResNet152,
}

/// Supported VGG variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(crate = "burn::serde")]
pub enum VggVariant {
    /// VGG-16 without batch normalization.
    Vgg16,
    /// VGG-16 with batch normalization.
    Vgg16Bn,
}

/// Supported Swin Transformer v1 variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(crate = "burn::serde")]
pub enum SwinVariant {
    /// Swin-T (Tiny).
    SwinT,
    /// Swin-S (Small).
    SwinS,
    /// Swin-B (Base).
    SwinB,
    /// Swin-L (Large).
    SwinL,
}

/// Supported Pyramid Vision Transformer v2 (`PVTv2`) variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(crate = "burn::serde")]
pub enum PvtV2Variant {
    /// PVTv2-B0.
    B0,
    /// PVTv2-B1.
    B1,
    /// PVTv2-B2.
    B2,
    /// PVTv2-B3.
    B3,
    /// PVTv2-B4.
    B4,
    /// PVTv2-B5.
    B5,
}

/// Dynamic wrapper over every supported [`Backbone`] implementation.
#[derive(Module, Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "Burn Module requires inline storage because Box<M> does not implement Module"
)]
pub enum BackboneModel {
    /// `ResNet` backbone.
    ResNet(ResNetBackbone),
    /// VGG backbone.
    Vgg(VggBackbone),
    /// Swin Transformer backbone.
    SwinTransformer(SwinTransformer),
    /// `PVTv2` backbone.
    PvtV2(PyramidVisionTransformerImpr),
}

impl BackboneModel {
    /// Runs feature extraction, returning an error if input shape or internal invariants fail.
    ///
    /// # Errors
    ///
    /// Returns [`BackboneError`] when a Swin Transformer rejects an invalid input shape.
    pub fn try_forward(&self, input: Tensor<4>) -> Result<[Tensor<4>; 4], BackboneError> {
        match self {
            Self::ResNet(backbone) => Ok(backbone.forward(input)),
            Self::Vgg(backbone) => Ok(backbone.forward(input)),
            Self::SwinTransformer(backbone) => backbone.forward(input),
            Self::PvtV2(backbone) => Ok(backbone.forward(input)),
        }
    }
}

impl Backbone for BackboneModel {
    fn forward(&self, input: Tensor<4>) -> [Tensor<4>; 4] {
        match self {
            Self::ResNet(backbone) => backbone.forward(input),
            Self::Vgg(backbone) => backbone.forward(input),
            Self::SwinTransformer(backbone) => Backbone::forward(backbone, input),
            Self::PvtV2(backbone) => backbone.forward(input),
        }
    }

    fn output_channels(&self) -> [usize; 4] {
        match self {
            Self::ResNet(backbone) => backbone.output_channels(),
            Self::Vgg(backbone) => backbone.output_channels(),
            Self::SwinTransformer(backbone) => backbone.output_channels(),
            Self::PvtV2(backbone) => backbone.output_channels(),
        }
    }
}
