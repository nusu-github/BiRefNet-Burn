//! Backbone implementations for BiRefNet
//!
//! This crate provides unified interfaces for different backbone architectures
//! used in BiRefNet, including ResNet, VGG, and Swin Transformer.

use burn::prelude::*;

pub use pvt_v2::{PvtV2Config, PyramidVisionTransformerImpr};
pub use resnet::{ResNetBackbone, ResNetConfig};
pub use swin_transformer::{SwinTransformer, SwinTransformerConfig};
pub use vgg::{VGGBackbone, VggConfig};

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
        self.forward(input).expect("SwinTransformer forward failed")
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
    /// PVTv2 backbone
    PVTv2(PVTv2Variant),
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

/// PVTv2 variants
#[derive(Config, Debug)]
pub enum PVTv2Variant {
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
    /// PVTv2 backbone
    PVTv2(PyramidVisionTransformerImpr<B>),
}

impl<B: Backend> Backbone<B> for BackboneWrapper<B> {
    fn forward(&self, input: Tensor<B, 4>) -> [Tensor<B, 4>; 4] {
        match self {
            Self::ResNet(backbone) => backbone.forward(input),
            Self::VGG(backbone) => backbone.forward(input),
            Self::SwinTransformer(backbone) => backbone
                .forward(input)
                .expect("SwinTransformer forward failed"),
            Self::PVTv2(backbone) => backbone.forward(input),
        }
    }

    fn output_channels(&self) -> [usize; 4] {
        match self {
            Self::ResNet(backbone) => backbone.output_channels(),
            Self::VGG(backbone) => backbone.output_channels(),
            Self::SwinTransformer(backbone) => backbone.output_channels(),
            Self::PVTv2(backbone) => backbone.output_channels(),
        }
    }
}

/// Factory function to create backbones
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
                SwinVariant::SwinT => SwinTransformerConfig::new()
                    .with_embed_dim(96)
                    .with_depths([2, 2, 6, 2])
                    .with_num_heads([3, 6, 12, 24]),
                SwinVariant::SwinS => SwinTransformerConfig::new()
                    .with_embed_dim(96)
                    .with_depths([2, 2, 18, 2])
                    .with_num_heads([3, 6, 12, 24]),
                SwinVariant::SwinB => SwinTransformerConfig::new()
                    .with_embed_dim(128)
                    .with_depths([2, 2, 18, 2])
                    .with_num_heads([4, 8, 16, 32]),
                SwinVariant::SwinL => SwinTransformerConfig::new()
                    .with_embed_dim(192)
                    .with_depths([2, 2, 18, 2])
                    .with_num_heads([6, 12, 24, 48]),
            };
            let backbone = config
                .init(device)
                .expect("Failed to initialize SwinTransformer");
            BackboneWrapper::SwinTransformer(backbone)
        }
        BackboneType::PVTv2(variant) => {
            let config = match variant {
                PVTv2Variant::B0 => PvtV2Config::b0(3),
                PVTv2Variant::B1 => PvtV2Config::b1(3),
                PVTv2Variant::B2 => PvtV2Config::b2(3),
                PVTv2Variant::B3 => PvtV2Config::b3(3),
                PVTv2Variant::B4 => PvtV2Config::b4(3),
                PVTv2Variant::B5 => PvtV2Config::b5(3),
            };
            let backbone = config.init(device);
            BackboneWrapper::PVTv2(backbone)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_resnet_backbone() {
        let device = Default::default();
        let backbone = create_backbone(BackboneType::ResNet(ResNetVariant::ResNet50), &device);

        let input = Tensor::<TestBackend, 4>::random(
            [1, 3, 224, 224],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = backbone.forward(input);
        let channels = backbone.output_channels();

        assert_eq!(output.len(), 4);
        assert_eq!(channels.len(), 4);

        // Check ResNet50 output shapes
        assert_eq!(output[0].dims(), [1, 256, 56, 56]);
        assert_eq!(output[1].dims(), [1, 512, 28, 28]);
        assert_eq!(output[2].dims(), [1, 1024, 14, 14]);
        assert_eq!(output[3].dims(), [1, 2048, 7, 7]);
    }

    #[test]
    fn test_vgg_backbone() {
        let device = Default::default();
        let backbone = create_backbone(BackboneType::VGG(VGGVariant::VGG16), &device);

        let input = Tensor::<TestBackend, 4>::random(
            [1, 3, 224, 224],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = backbone.forward(input);
        let channels = backbone.output_channels();

        assert_eq!(output.len(), 4);
        assert_eq!(channels.len(), 4);

        // Check VGG16 output shapes
        assert_eq!(output[0].dims(), [1, 128, 112, 112]);
        assert_eq!(output[1].dims(), [1, 256, 56, 56]);
        assert_eq!(output[2].dims(), [1, 512, 28, 28]);
        assert_eq!(output[3].dims(), [1, 512, 14, 14]);
    }

    #[test]
    fn test_swin_transformer_backbone() {
        let device = Default::default();
        let backbone = create_backbone(BackboneType::SwinTransformer(SwinVariant::SwinT), &device);

        let input = Tensor::<TestBackend, 4>::random(
            [1, 3, 224, 224],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = backbone.forward(input);
        let channels = backbone.output_channels();

        assert_eq!(output.len(), 4);
        assert_eq!(channels.len(), 4);

        // Check Swin-T output shapes
        assert_eq!(output[0].dims(), [1, 96, 56, 56]);
        assert_eq!(output[1].dims(), [1, 192, 28, 28]);
        assert_eq!(output[2].dims(), [1, 384, 14, 14]);
        assert_eq!(output[3].dims(), [1, 768, 7, 7]);
    }

    #[test]
    fn test_pvt_v2_backbone() {
        let device = Default::default();
        let backbone = create_backbone(BackboneType::PVTv2(PVTv2Variant::B2), &device);

        let input = Tensor::<TestBackend, 4>::random(
            [1, 3, 224, 224],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let output = backbone.forward(input);
        let channels = backbone.output_channels();

        assert_eq!(output.len(), 4);
        assert_eq!(channels.len(), 4);

        // Check PVTv2-B2 output shapes
        assert_eq!(output[0].dims(), [1, 64, 56, 56]);
        assert_eq!(output[1].dims(), [1, 128, 28, 28]);
        assert_eq!(output[2].dims(), [1, 320, 14, 14]);
        assert_eq!(output[3].dims(), [1, 512, 7, 7]);
    }
}
