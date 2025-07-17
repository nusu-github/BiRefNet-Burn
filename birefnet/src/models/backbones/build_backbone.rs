//! # Backbone Builder
//!
//! This module provides a factory function `build_backbone` to construct
//! a backbone model based on the configuration.

use burn::prelude::*;

use super::{
    swin_v1_b, swin_v1_l, swin_v1_s, swin_v1_t, ResNetBackbone, SwinTransformer, VGGBackbone,
};
use crate::config::{Backbone, ModelConfig};
use crate::error::{BiRefNetError, BiRefNetResult};

/// An enum to encapsulate different backbone architectures.
///
/// This allows for a single type to represent any of the supported backbones,
/// simplifying the model construction process.
#[derive(Module, Debug)]
pub enum BackboneEnum<B: Backend> {
    /// The Swin Transformer model.
    SwinTransformer(SwinTransformer<B>),
    /// The ResNet model family.
    ResNet(ResNetBackbone<B>),
    /// The VGG model family.
    VGG(VGGBackbone<B>),
}

/// Constructs a backbone model based on the provided configuration.
///
/// This function acts as a factory, instantiating the correct backbone architecture
/// as specified in `config.backbone.backbone`.
///
/// # Arguments
///
/// * `config` - The main model configuration.
/// * `device` - The device to create the model on.
///
/// # Returns
///
/// A `BiRefNetResult` containing the constructed `BackboneEnum`.
///
/// # Errors
///
/// Returns `BiRefNetError::UnsupportedBackbone` if the specified backbone
/// is not yet implemented in this crate.
pub fn build_backbone<B: Backend>(
    config: &ModelConfig,
    device: &Device<B>,
) -> BiRefNetResult<BackboneEnum<B>> {
    match config.backbone.backbone {
        Backbone::Vgg16 => Ok(BackboneEnum::VGG(VGGBackbone::vgg16(device))),
        Backbone::Vgg16bn => Ok(BackboneEnum::VGG(VGGBackbone::vgg16_bn(device))),
        Backbone::Resnet50 => Ok(BackboneEnum::ResNet(ResNetBackbone::resnet50(device))),
        Backbone::SwinV1T => Ok(BackboneEnum::SwinTransformer(swin_v1_t(device)?)),
        Backbone::SwinV1S => Ok(BackboneEnum::SwinTransformer(swin_v1_s(device)?)),
        Backbone::SwinV1B => Ok(BackboneEnum::SwinTransformer(swin_v1_b(device)?)),
        Backbone::SwinV1L => Ok(BackboneEnum::SwinTransformer(swin_v1_l(device)?)),
        Backbone::PvtV2B0 => Err(BiRefNetError::UnsupportedBackbone {
            backbone: "PVT v2 B0".to_string(),
        }),
        Backbone::PvtV2B1 => Err(BiRefNetError::UnsupportedBackbone {
            backbone: "PVT v2 B1".to_string(),
        }),
        Backbone::PvtV2B2 => Err(BiRefNetError::UnsupportedBackbone {
            backbone: "PVT v2 B2".to_string(),
        }),
        Backbone::PvtV2B5 => Err(BiRefNetError::UnsupportedBackbone {
            backbone: "PVT v2 B5".to_string(),
        }),
    }
}
