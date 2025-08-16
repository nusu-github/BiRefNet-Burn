//! Backbone implementations for BiRefNet
//!
//! This crate provides unified interfaces for different backbone architectures
//! used in BiRefNet, including ResNet, VGG, and Swin Transformer.

mod backbones;

pub use backbones::{
    pvt_v2::{PvtV2Config, PyramidVisionTransformerImpr},
    resnet::{ResNetBackbone, ResNetConfig},
    swin_transformer::{SwinTransformer, SwinTransformerConfig},
    vgg::{VGGBackbone, VggConfig},
};
use burn::prelude::*;

/// Unified backbone trait for BiRefNet
pub trait Backbone<B: Backend> {
    /// Forward pass through the backbone
    ///
    /// # Arguments
    /// * `input` - Input tensor of shape `[batch_size, channels, height, width]`
    ///
    /// # Returns
    /// Array of 4 feature maps at different scales
    fn forward(&self, input: Tensor<B, 4>) -> [Tensor<B, 4>; 4];

    /// Get output channels for each scale
    fn output_channels(&self) -> [usize; 4];
}

/// Implement Backbone trait for ResNet
impl<B: Backend> Backbone<B> for ResNetBackbone<B> {
    fn forward(&self, input: Tensor<B, 4>) -> [Tensor<B, 4>; 4] {
        self.forward(input)
    }

    fn output_channels(&self) -> [usize; 4] {
        // ResNet50 channels: [256, 512, 1024, 2048]
        // ResNet18 channels: [64, 128, 256, 512]
        // For now, return ResNet50 channels as default
        [256, 512, 1024, 2048]
    }
}

/// Implement Backbone trait for VGG
impl<B: Backend> Backbone<B> for VGGBackbone<B> {
    fn forward(&self, input: Tensor<B, 4>) -> [Tensor<B, 4>; 4] {
        self.forward(input)
    }

    fn output_channels(&self) -> [usize; 4] {
        // VGG16 channels: [128, 256, 512, 512]
        [128, 256, 512, 512]
    }
}

/// Implement Backbone trait for Swin Transformer
impl<B: Backend> Backbone<B> for SwinTransformer<B> {
    fn forward(&self, input: Tensor<B, 4>) -> [Tensor<B, 4>; 4] {
        let input_dims = input.dims();
        self.forward(input).unwrap_or_else(|err| {
            panic!("SwinTransformer forward pass failed with input shape {input_dims:?}: {err}")
        })
    }

    fn output_channels(&self) -> [usize; 4] {
        // Swin-T channels: [96, 192, 384, 768]
        [96, 192, 384, 768]
    }
}

/// Implement Backbone trait for PVTv2
impl<B: Backend> Backbone<B> for PyramidVisionTransformerImpr<B> {
    fn forward(&self, input: Tensor<B, 4>) -> [Tensor<B, 4>; 4] {
        self.forward(input)
    }

    fn output_channels(&self) -> [usize; 4] {
        self.output_channels()
    }
}

/// Enumeration of supported backbone types
#[derive(Debug, Clone)]
pub enum BackboneType {
    /// ResNet backbone
    ResNet(ResNetVariant),
    /// VGG backbone
    VGG(VGGVariant),
    /// Swin Transformer backbone
    SwinTransformer(SwinVariant),
    /// PvtV2 backbone
    PvtV2(PvtV2Variant),
}

/// ResNet variants
#[derive(Debug, Clone)]
pub enum ResNetVariant {
    /// ResNet-18
    ResNet18,
    /// ResNet-34
    ResNet34,
    /// ResNet-50
    ResNet50,
    /// ResNet-101
    ResNet101,
    /// ResNet-152
    ResNet152,
}

/// VGG variants
#[derive(Debug, Clone)]
pub enum VGGVariant {
    /// VGG-16
    VGG16,
    /// VGG-16 with batch normalization
    VGG16BN,
}

/// Swin Transformer variants
#[derive(Debug, Clone)]
pub enum SwinVariant {
    /// Swin-T (Tiny)
    SwinT,
    /// Swin-S (Small)
    SwinS,
    /// Swin-B (Base)
    SwinB,
    /// Swin-L (Large)
    SwinL,
}

/// PvtV2 variants
#[derive(Config, Debug)]
pub enum PvtV2Variant {
    /// PVTv2-B0
    B0,
    /// PVTv2-B1
    B1,
    /// PVTv2-B2
    B2,
    /// PVTv2-B3
    B3,
    /// PVTv2-B4
    B4,
    /// PVTv2-B5
    B5,
}

/// Enum to wrap different backbone implementations
#[derive(Module, Debug)]
pub enum BackboneWrapper<B: Backend> {
    /// ResNet backbone
    ResNet(ResNetBackbone<B>),
    /// VGG backbone
    VGG(VGGBackbone<B>),
    /// Swin Transformer backbone
    SwinTransformer(SwinTransformer<B>),
    /// PvtV2 backbone
    PvtV2(PyramidVisionTransformerImpr<B>),
}

impl<B: Backend> Backbone<B> for BackboneWrapper<B> {
    fn forward(&self, input: Tensor<B, 4>) -> [Tensor<B, 4>; 4] {
        match self {
            Self::ResNet(backbone) => backbone.forward(input),
            Self::VGG(backbone) => backbone.forward(input),
            Self::SwinTransformer(backbone) => {
                let input_dims = input.dims();
                backbone.forward(input).unwrap_or_else(|err| {
                    panic!(
                        "SwinTransformer backbone forward pass failed with input shape {input_dims:?}: {err}"
                    )
                })
            }
            Self::PvtV2(backbone) => backbone.forward(input),
        }
    }

    fn output_channels(&self) -> [usize; 4] {
        match self {
            Self::ResNet(backbone) => backbone.output_channels(),
            Self::VGG(backbone) => backbone.output_channels(),
            Self::SwinTransformer(backbone) => backbone.output_channels(),
            Self::PvtV2(backbone) => backbone.output_channels(),
        }
    }
}

/// Factory function to create backbone architectures.
///
/// # Arguments
/// * `backbone_type` - The type and variant of backbone to create
/// * `device` - The device on which to initialize the backbone
///
/// # Returns
/// A wrapped backbone instance implementing the unified `Backbone` trait
///
/// # Panics
/// * When Swin Transformer initialization fails due to invalid configuration parameters
/// * When PvtV2 initialization fails due to invalid configuration parameters  
/// * When the device is not compatible with the selected backend
///
/// # Examples
/// ```rust,ignore
/// let backbone = create_backbone(BackboneType::ResNet(ResNetVariant::ResNet50), &device);
/// let features = backbone.forward(input_tensor);
/// ```
pub fn create_backbone<B: Backend>(
    backbone_type: BackboneType,
    device: &Device<B>,
) -> BackboneWrapper<B> {
    match backbone_type {
        BackboneType::ResNet(variant) => {
            let backbone = match variant {
                ResNetVariant::ResNet18 => ResNetBackbone::resnet18(device),
                ResNetVariant::ResNet34 => ResNetBackbone::resnet34(device),
                ResNetVariant::ResNet50 => ResNetBackbone::resnet50(device),
                ResNetVariant::ResNet101 => ResNetBackbone::resnet101(device),
                ResNetVariant::ResNet152 => ResNetBackbone::resnet152(device),
            };
            BackboneWrapper::ResNet(backbone)
        }
        BackboneType::VGG(variant) => {
            let backbone = match variant {
                VGGVariant::VGG16 => VGGBackbone::vgg16(device),
                VGGVariant::VGG16BN => VGGBackbone::vgg16_bn(device),
            };
            BackboneWrapper::VGG(backbone)
        }
        BackboneType::SwinTransformer(variant) => {
            let config = match variant {
                SwinVariant::SwinT => {
                    SwinTransformerConfig::new()
                        .with_embed_dim(96)
                        .with_depths([2, 2, 6, 2])
                        .with_num_heads([3, 6, 12, 24])
                        .with_window_size(7)
                }
                SwinVariant::SwinS => {
                    SwinTransformerConfig::new()
                        .with_embed_dim(96)
                        .with_depths([2, 2, 18, 2])
                        .with_num_heads([3, 6, 12, 24])
                        .with_window_size(7)
                }
                SwinVariant::SwinB => {
                    SwinTransformerConfig::new()
                        .with_embed_dim(128)
                        .with_depths([2, 2, 18, 2])
                        .with_num_heads([4, 8, 16, 32])
                        .with_window_size(12)
                }
                SwinVariant::SwinL => {
                    SwinTransformerConfig::new()
                        .with_embed_dim(192)
                        .with_depths([2, 2, 18, 2])
                        .with_num_heads([6, 12, 24, 48])
                        .with_window_size(12)
                }
            };
            let backbone = config.init(device).unwrap_or_else(|err| {
                panic!("Failed to initialize SwinTransformer with variant {variant:?}: {err}")
            });
            BackboneWrapper::SwinTransformer(backbone)
        }
        BackboneType::PvtV2(variant) => {
            let config = match variant {
                PvtV2Variant::B0 => PvtV2Config::b0(3),
                PvtV2Variant::B1 => PvtV2Config::b1(3),
                PvtV2Variant::B2 => PvtV2Config::b2(3),
                PvtV2Variant::B3 => PvtV2Config::b3(3),
                PvtV2Variant::B4 => PvtV2Config::b4(3),
                PvtV2Variant::B5 => PvtV2Config::b5(3),
            };
            let backbone = config.init(device);
            BackboneWrapper::PvtV2(backbone)
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    pub type TestBackend = NdArray;
}
