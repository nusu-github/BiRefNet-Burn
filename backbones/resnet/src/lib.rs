//! ResNet implementation for BiRefNet backbone.
//!
//! This module provides ResNet models adapted for use as backbones in BiRefNet.
//! The implementation is based on the official torchvision ResNet implementation.

use burn::nn::{
    conv::{Conv2d, Conv2dConfig},
    pool::{MaxPool2d, MaxPool2dConfig},
    BatchNorm, BatchNormConfig, Initializer, PaddingConfig2d, Relu,
};
use burn::prelude::*;
use core::f64::consts::SQRT_2;

mod blocks;
pub use blocks::*;

// ResNet residual layer block configs
const RESNET18_BLOCKS: [usize; 4] = [2, 2, 2, 2];
const RESNET34_BLOCKS: [usize; 4] = [3, 4, 6, 3];
const RESNET50_BLOCKS: [usize; 4] = [3, 4, 6, 3];
const RESNET101_BLOCKS: [usize; 4] = [3, 4, 23, 3];
const RESNET152_BLOCKS: [usize; 4] = [3, 8, 36, 3];

/// ResNet backbone implementation for BiRefNet.
///
/// This provides the 4 feature levels (conv1-4) needed for BiRefNet.
/// Derived from torchvision.models.resnet.ResNet
#[derive(Module, Debug)]
pub struct ResNetBackbone<B: Backend> {
    // First feature level: conv1 + bn1 + relu + maxpool + layer1
    pub conv1_block: Conv1Block<B>,
    pub layer1: LayerBlock<B>,

    // Second feature level: layer2
    pub layer2: LayerBlock<B>,

    // Third feature level: layer3
    pub layer3: LayerBlock<B>,

    // Fourth feature level: layer4
    pub layer4: LayerBlock<B>,
}

impl<B: Backend> ResNetBackbone<B> {
    /// Forward pass that returns the 4 feature levels required by BiRefNet.
    pub fn forward(&self, input: Tensor<B, 4>) -> [Tensor<B, 4>; 4] {
        // First feature level: conv1 + bn1 + relu + maxpool + layer1
        let conv1 = self.conv1_block.forward(input);
        let conv1 = self.layer1.forward(conv1);

        // Second feature level: layer2
        let conv2 = self.layer2.forward(conv1.clone());

        // Third feature level: layer3
        let conv3 = self.layer3.forward(conv2.clone());

        // Fourth feature level: layer4
        let conv4 = self.layer4.forward(conv3.clone());

        [conv1, conv2, conv3, conv4]
    }

    /// Create ResNet-18 backbone.
    pub fn resnet18(device: &Device<B>) -> Self {
        Self::new(RESNET18_BLOCKS, 1, device)
    }

    /// Create ResNet-34 backbone.
    pub fn resnet34(device: &Device<B>) -> Self {
        Self::new(RESNET34_BLOCKS, 1, device)
    }

    /// Create ResNet-50 backbone.
    pub fn resnet50(device: &Device<B>) -> Self {
        Self::new(RESNET50_BLOCKS, 4, device)
    }

    /// Create ResNet-101 backbone.
    pub fn resnet101(device: &Device<B>) -> Self {
        Self::new(RESNET101_BLOCKS, 4, device)
    }

    /// Create ResNet-152 backbone.
    pub fn resnet152(device: &Device<B>) -> Self {
        Self::new(RESNET152_BLOCKS, 4, device)
    }

    /// Create a new ResNet backbone with the specified configuration.
    fn new(blocks: [usize; 4], expansion: usize, device: &Device<B>) -> Self {
        // Validate expansion
        assert!(
            expansion == 1 || expansion == 4,
            "ResNet backbone only supports expansion values [1, 4] for residual blocks"
        );

        // First conv block: 7x7 conv, 64, stride=2, padding=3
        let conv1_block = Conv1Block::new(3, 64, device);

        // Residual blocks
        let bottleneck = expansion > 1;
        let layer1 = LayerBlock::new(blocks[0], 64, 64 * expansion, 1, bottleneck, device);
        let layer2 = LayerBlock::new(
            blocks[1],
            64 * expansion,
            128 * expansion,
            2,
            bottleneck,
            device,
        );
        let layer3 = LayerBlock::new(
            blocks[2],
            128 * expansion,
            256 * expansion,
            2,
            bottleneck,
            device,
        );
        let layer4 = LayerBlock::new(
            blocks[3],
            256 * expansion,
            512 * expansion,
            2,
            bottleneck,
            device,
        );

        Self {
            conv1_block,
            layer1,
            layer2,
            layer3,
            layer4,
        }
    }
}

/// First conv block: conv1 + bn1 + relu + maxpool
#[derive(Module, Debug)]
pub struct Conv1Block<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    relu: Relu,
    maxpool: MaxPool2d,
}

impl<B: Backend> Conv1Block<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let out = self.conv1.forward(input);
        let out = self.bn1.forward(out);
        let out = self.relu.forward(out);
        self.maxpool.forward(out)
    }

    /// Create a new Conv1Block.
    pub fn new(in_channels: usize, out_channels: usize, device: &Device<B>) -> Self {
        let initializer = Initializer::KaimingNormal {
            gain: SQRT_2,
            fan_out_only: true,
        };

        // 7x7 conv, stride=2, padding=3
        let conv1 = Conv2dConfig::new([in_channels, out_channels], [7, 7])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(3, 3))
            .with_bias(false)
            .with_initializer(initializer)
            .init(device);

        let bn1 = BatchNormConfig::new(out_channels).init(device);

        // 3x3 maxpool, stride=2, padding=1
        let maxpool = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init();

        Self {
            conv1,
            bn1,
            relu: Relu::new(),
            maxpool,
        }
    }
}

/// Legacy alias for backwards compatibility
pub type ResNet<B> = ResNetBackbone<B>;

/// ResNet configuration
#[derive(Config)]
pub struct ResNetConfig {
    /// Number of layers in each block [3, 4, 6, 3] for ResNet50
    pub layers: Vec<usize>,
    /// Number of classes for the final layer (None for backbone usage)
    pub num_classes: Option<usize>,
    /// Zero-initialize the last BN in each residual branch
    pub zero_init_residual: bool,
}

impl ResNetConfig {
    /// ResNet50 configuration
    pub fn resnet50() -> Self {
        Self {
            layers: vec![3, 4, 6, 3],
            num_classes: None,
            zero_init_residual: false,
        }
    }

    /// ResNet101 configuration
    pub fn resnet101() -> Self {
        Self {
            layers: vec![3, 4, 23, 3],
            num_classes: None,
            zero_init_residual: false,
        }
    }
}

/// ResNet model output containing multi-scale features
#[derive(Debug, Clone)]
pub struct ResNetOutput<B: Backend> {
    /// Layer1 output (1/4 scale)
    pub layer1: Tensor<B, 4>,
    /// Layer2 output (1/8 scale)
    pub layer2: Tensor<B, 4>,
    /// Layer3 output (1/16 scale)
    pub layer3: Tensor<B, 4>,
    /// Layer4 output (1/32 scale)
    pub layer4: Tensor<B, 4>,
}

impl ResNetConfig {
    /// Initialize ResNet model
    pub fn init<B: Backend>(&self, device: &B::Device) -> ResNetBackbone<B> {
        match self.layers.as_slice() {
            [3, 4, 6, 3] => ResNetBackbone::resnet50(device),
            [3, 4, 23, 3] => ResNetBackbone::resnet101(device),
            _ => panic!("Unsupported ResNet configuration"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_resnet_config() {
        let config = ResNetConfig::resnet50();
        assert_eq!(config.layers, vec![3, 4, 6, 3]);
        assert_eq!(config.num_classes, None);
        assert!(!config.zero_init_residual);
    }

    #[test]
    fn test_resnet_forward() {
        let device = Default::default();
        let model = ResNetBackbone::resnet50(&device);

        let input = Tensor::<TestBackend, 4>::random(
            [1, 3, 224, 224],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = model.forward(input);

        // Check output shapes for ResNet50
        assert_eq!(output[0].dims(), [1, 256, 56, 56]); // 224/4 = 56
        assert_eq!(output[1].dims(), [1, 512, 28, 28]); // 56/2 = 28
        assert_eq!(output[2].dims(), [1, 1024, 14, 14]); // 28/2 = 14
        assert_eq!(output[3].dims(), [1, 2048, 7, 7]); // 14/2 = 7
    }

    #[test]
    fn test_resnet18_forward() {
        let device = Default::default();
        let model = ResNetBackbone::resnet18(&device);

        let input = Tensor::<TestBackend, 4>::random(
            [1, 3, 224, 224],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = model.forward(input);

        // Check output shapes for ResNet18 (expansion=1)
        assert_eq!(output[0].dims(), [1, 64, 56, 56]); // 224/4 = 56
        assert_eq!(output[1].dims(), [1, 128, 28, 28]); // 56/2 = 28
        assert_eq!(output[2].dims(), [1, 256, 14, 14]); // 28/2 = 14
        assert_eq!(output[3].dims(), [1, 512, 7, 7]); // 14/2 = 7
    }
}
