//! VGG implementation for BiRefNet backbone.
//!
//! This module provides VGG models adapted for use as backbones in BiRefNet.
//! The implementation is based on the torchvision VGG implementation.

use core::f64::consts::SQRT_2;

use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
        BatchNorm, BatchNormConfig, Initializer, PaddingConfig2d, Relu,
    },
    prelude::*,
};

/// VGG configuration type for layer specification.
#[derive(Debug, Clone)]
pub enum VGGLayer {
    /// Convolution layer with specified output channels.
    Conv(usize),
    /// Max pooling layer.
    MaxPool,
}

/// VGG backbone implementation for BiRefNet.
///
/// This provides the 4 feature levels (conv1-4) needed for BiRefNet.
/// Based on the torchvision VGG implementation.
#[derive(Module, Debug)]
pub struct VGGBackbone<B: Backend> {
    /// First feature level: layers 0-10 (no BN) or 0-14 (with BN)
    pub conv1: VGGFeatureBlock<B>,
    /// Second feature level: layers 10-17 (no BN) or 14-24 (with BN)
    pub conv2: VGGFeatureBlock<B>,
    /// Third feature level: layers 17-24 (no BN) or 24-34 (with BN)
    pub conv3: VGGFeatureBlock<B>,
    /// Fourth feature level: layers 24-31 (no BN) or 34-44 (with BN)
    pub conv4: VGGFeatureBlock<B>,
}

impl<B: Backend> VGGBackbone<B> {
    /// Forward pass that returns the 4 feature levels required by BiRefNet.
    pub fn forward(&self, input: Tensor<B, 4>) -> [Tensor<B, 4>; 4] {
        let conv1 = self.conv1.forward(input);
        let conv2 = self.conv2.forward(conv1.clone());
        let conv3 = self.conv3.forward(conv2.clone());
        let conv4 = self.conv4.forward(conv3.clone());

        [conv1, conv2, conv3, conv4]
    }

    /// Create VGG16 backbone without batch normalization.
    pub fn vgg16(device: &Device<B>) -> Self {
        Self::new(false, device)
    }

    /// Create VGG16 backbone with batch normalization.
    pub fn vgg16_bn(device: &Device<B>) -> Self {
        Self::new(true, device)
    }

    /// Create a new VGG backbone with the specified configuration.
    fn new(batch_norm: bool, device: &Device<B>) -> Self {
        // Based on PyTorch VGG16 feature extraction for BiRefNet
        // VGG16: conv1: [:10], conv2: [10:17], conv3: [17:24], conv4: [24:31]
        // VGG16_BN: conv1: [:14], conv2: [14:24], conv3: [24:34], conv4: [34:44]

        if batch_norm {
            let conv1 = VGGFeatureBlock::new(
                &[
                    VGGLayer::Conv(64),
                    VGGLayer::Conv(64),
                    VGGLayer::MaxPool,
                    VGGLayer::Conv(128),
                    VGGLayer::Conv(128),
                    VGGLayer::MaxPool,
                ],
                3,
                true,
                device,
            );

            let conv2 = VGGFeatureBlock::new(
                &[
                    VGGLayer::Conv(256),
                    VGGLayer::Conv(256),
                    VGGLayer::Conv(256),
                    VGGLayer::MaxPool,
                ],
                128,
                true,
                device,
            );

            let conv3 = VGGFeatureBlock::new(
                &[
                    VGGLayer::Conv(512),
                    VGGLayer::Conv(512),
                    VGGLayer::Conv(512),
                    VGGLayer::MaxPool,
                ],
                256,
                true,
                device,
            );

            let conv4 = VGGFeatureBlock::new(
                &[
                    VGGLayer::Conv(512),
                    VGGLayer::Conv(512),
                    VGGLayer::Conv(512),
                    VGGLayer::MaxPool,
                ],
                512,
                true,
                device,
            );

            Self {
                conv1,
                conv2,
                conv3,
                conv4,
            }
        } else {
            let conv1 = VGGFeatureBlock::new(
                &[
                    VGGLayer::Conv(64),
                    VGGLayer::Conv(64),
                    VGGLayer::MaxPool,
                    VGGLayer::Conv(128),
                    VGGLayer::Conv(128),
                    VGGLayer::MaxPool,
                ],
                3,
                false,
                device,
            );

            let conv2 = VGGFeatureBlock::new(
                &[
                    VGGLayer::Conv(256),
                    VGGLayer::Conv(256),
                    VGGLayer::Conv(256),
                    VGGLayer::MaxPool,
                ],
                128,
                false,
                device,
            );

            let conv3 = VGGFeatureBlock::new(
                &[
                    VGGLayer::Conv(512),
                    VGGLayer::Conv(512),
                    VGGLayer::Conv(512),
                    VGGLayer::MaxPool,
                ],
                256,
                false,
                device,
            );

            let conv4 = VGGFeatureBlock::new(
                &[
                    VGGLayer::Conv(512),
                    VGGLayer::Conv(512),
                    VGGLayer::Conv(512),
                    VGGLayer::MaxPool,
                ],
                512,
                false,
                device,
            );

            Self {
                conv1,
                conv2,
                conv3,
                conv4,
            }
        }
    }
}

/// A feature block in VGG architecture.
#[derive(Module, Debug)]
pub struct VGGFeatureBlock<B: Backend> {
    layers: Vec<VGGBlockLayer<B>>,
}

impl<B: Backend> VGGFeatureBlock<B> {
    /// Forward pass through the feature block.
    pub fn forward(&self, mut input: Tensor<B, 4>) -> Tensor<B, 4> {
        for layer in &self.layers {
            input = layer.forward(input);
        }
        input
    }

    /// Create a new VGG feature block.
    pub fn new(
        config: &[VGGLayer],
        in_channels: usize,
        batch_norm: bool,
        device: &Device<B>,
    ) -> Self {
        let mut layers = Vec::new();
        let mut current_channels = in_channels;

        for layer_config in config {
            match layer_config {
                VGGLayer::Conv(out_channels) => {
                    let conv_layer =
                        VGGConvLayer::new(current_channels, *out_channels, batch_norm, device);
                    layers.push(VGGBlockLayer::Conv(conv_layer));
                    current_channels = *out_channels;
                }
                VGGLayer::MaxPool => {
                    let maxpool = MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init();
                    layers.push(VGGBlockLayer::MaxPool(maxpool));
                }
            }
        }

        Self { layers }
    }
}

/// Individual layer types in VGG feature blocks.
#[derive(Module, Debug)]
pub enum VGGBlockLayer<B: Backend> {
    /// Convolution layer with optional batch normalization.
    Conv(VGGConvLayer<B>),
    /// Max pooling layer.
    MaxPool(MaxPool2d),
}

impl<B: Backend> VGGBlockLayer<B> {
    /// Forward pass through the layer.
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Self::Conv(conv) => conv.forward(input),
            Self::MaxPool(pool) => pool.forward(input),
        }
    }
}

/// VGG convolution layer with optional batch normalization.
#[derive(Module, Debug)]
pub struct VGGConvLayer<B: Backend> {
    conv: Conv2d<B>,
    batch_norm: Option<BatchNorm<B, 2>>,
    relu: Relu,
}

impl<B: Backend> VGGConvLayer<B> {
    /// Forward pass through the convolution layer.
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let out = self.conv.forward(input);
        let out = if let Some(bn) = &self.batch_norm {
            bn.forward(out)
        } else {
            out
        };
        self.relu.forward(out)
    }

    /// Create a new VGG convolution layer.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        batch_norm: bool,
        device: &Device<B>,
    ) -> Self {
        let initializer = Initializer::KaimingNormal {
            gain: SQRT_2,
            fan_out_only: true,
        };

        // 3x3 conv, stride=1, padding=1
        let conv = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(!batch_norm) // No bias when using batch norm
            .with_initializer(initializer)
            .init(device);

        let batch_norm = batch_norm.then(|| BatchNormConfig::new(out_channels).init(device));

        Self {
            conv,
            batch_norm,
            relu: Relu::new(),
        }
    }
}

/// VGG configuration
#[derive(Config, Debug)]
pub struct VggConfig {
    /// VGG variant (16 or 19)
    pub variant: VggVariant,
    /// Use batch normalization
    pub batch_norm: bool,
    /// Number of classes for the final layer (None for backbone usage)
    pub num_classes: Option<usize>,
}

/// VGG variant enumeration
#[derive(Config, Debug)]
pub enum VggVariant {
    /// VGG16
    Vgg16,
    /// VGG19
    Vgg19,
}

impl VggConfig {
    /// VGG16 configuration
    pub const fn vgg16() -> Self {
        Self {
            variant: VggVariant::Vgg16,
            batch_norm: false,
            num_classes: None,
        }
    }

    /// VGG19 configuration
    pub const fn vgg19() -> Self {
        Self {
            variant: VggVariant::Vgg19,
            batch_norm: false,
            num_classes: None,
        }
    }
}

/// VGG model output containing multi-scale features
#[derive(Debug, Clone)]
pub struct VggOutput<B: Backend> {
    /// Block1 output (1/2 scale)
    pub block1: Tensor<B, 4>,
    /// Block2 output (1/4 scale)
    pub block2: Tensor<B, 4>,
    /// Block3 output (1/8 scale)
    pub block3: Tensor<B, 4>,
    /// Block4 output (1/16 scale)
    pub block4: Tensor<B, 4>,
}

impl VggConfig {
    /// Initialize VGG model
    pub fn init<B: Backend>(&self, device: &B::Device) -> VGGBackbone<B> {
        match self.variant {
            VggVariant::Vgg16 => {
                if self.batch_norm {
                    VGGBackbone::vgg16_bn(device)
                } else {
                    VGGBackbone::vgg16(device)
                }
            }
            VggVariant::Vgg19 => {
                // For now, implement VGG19 as VGG16
                if self.batch_norm {
                    VGGBackbone::vgg16_bn(device)
                } else {
                    VGGBackbone::vgg16(device)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;

    use super::*;
    use crate::tests::TestBackend;

    #[test]
    fn vgg16_config_disables_batch_norm_by_default() {
        let config = VggConfig::vgg16();
        assert!(!config.batch_norm);
    }

    #[test]
    fn vgg16_config_sets_num_classes_none_for_backbone() {
        let config = VggConfig::vgg16();
        assert_eq!(config.num_classes, None);
    }

    #[test]
    fn vgg16_forward_returns_correct_feature_shapes() {
        let device = Default::default();
        let model = VGGBackbone::vgg16(&device);

        let input = Tensor::<TestBackend, 4>::random(
            [1, 3, 224, 224],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = model.forward(input);

        // Check output shapes for VGG16
        assert_eq!(output[0].dims(), [1, 128, 56, 56]); // After conv1 block
        assert_eq!(output[1].dims(), [1, 256, 28, 28]); // After conv2 block
        assert_eq!(output[2].dims(), [1, 512, 14, 14]); // After conv3 block
        assert_eq!(output[3].dims(), [1, 512, 7, 7]); // After conv4 block
    }

    #[test]
    fn vgg16_bn_forward_returns_correct_feature_shapes() {
        let device = Default::default();
        let model = VGGBackbone::vgg16_bn(&device);

        let input = Tensor::<TestBackend, 4>::random(
            [1, 3, 224, 224],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = model.forward(input);

        // Check output shapes for VGG16-BN
        assert_eq!(output[0].dims(), [1, 128, 56, 56]); // After conv1 block
        assert_eq!(output[1].dims(), [1, 256, 28, 28]); // After conv2 block
        assert_eq!(output[2].dims(), [1, 512, 14, 14]); // After conv3 block
        assert_eq!(output[3].dims(), [1, 512, 7, 7]); // After conv4 block
    }
}
