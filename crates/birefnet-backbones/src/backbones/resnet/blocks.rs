//! ResNet block implementations.
//!
//! This module contains the building blocks for ResNet: BasicBlock, Bottleneck, and LayerBlock.

use core::f64::consts::SQRT_2;

use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        BatchNorm, BatchNormConfig, Initializer, PaddingConfig2d, Relu,
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub enum ResidualBlock<B: Backend> {
    /// A bottleneck residual block.
    Bottleneck(Bottleneck<B>),
    /// A basic residual block.
    Basic(BasicBlock<B>),
}

impl<B: Backend> ResidualBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Self::Basic(block) => block.forward(input),
            Self::Bottleneck(block) => block.forward(input),
        }
    }
}

/// ResNet basic residual block implementation.
/// Derived from torchvision.models.resnet.BasicBlock
#[derive(Module, Debug)]
pub struct BasicBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    relu: Relu,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    downsample: Option<Downsample<B>>,
}

impl<B: Backend> BasicBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let identity = input.clone();

        // Conv block
        let out = self.conv1.forward(input);
        let out = self.bn1.forward(out);
        let out = self.relu.forward(out);
        let out = self.conv2.forward(out);
        let out = self.bn2.forward(out);

        // Skip connection
        let out = match &self.downsample {
            Some(downsample) => out + downsample.forward(identity),
            None => out + identity,
        };

        // Activation
        self.relu.forward(out)
    }

    /// Create a new BasicBlock.
    pub fn new(in_channels: usize, out_channels: usize, stride: usize, device: &Device<B>) -> Self {
        let initializer = Initializer::KaimingNormal {
            gain: SQRT_2,
            fan_out_only: true,
        };

        // conv3x3
        let conv1 = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false)
            .with_initializer(initializer.clone())
            .init(device);
        let bn1 = BatchNormConfig::new(out_channels).init(device);

        // conv3x3
        let conv2 = Conv2dConfig::new([out_channels, out_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false)
            .with_initializer(initializer)
            .init(device);
        let bn2 = BatchNormConfig::new(out_channels).init(device);

        let downsample = (stride != 1 || in_channels != out_channels)
            .then(|| Downsample::new(in_channels, out_channels, stride, device));

        Self {
            conv1,
            bn1,
            relu: Relu::new(),
            conv2,
            bn2,
            downsample,
        }
    }
}

/// ResNet bottleneck residual block implementation.
/// Derived from torchvision.models.resnet.Bottleneck
///
/// **NOTE:** Following common practice, this bottleneck block places the stride for downsampling
/// to the second 3x3 convolution while the original paper places it to the first 1x1 convolution.
/// This variant improves the accuracy and is known as ResNet V1.5.
#[derive(Module, Debug)]
pub struct Bottleneck<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
    relu: Relu,
    conv2: Conv2d<B>,
    bn2: BatchNorm<B, 2>,
    conv3: Conv2d<B>,
    bn3: BatchNorm<B, 2>,
    downsample: Option<Downsample<B>>,
}

impl<B: Backend> Bottleneck<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let identity = input.clone();

        // Conv block
        let out = self.conv1.forward(input);
        let out = self.bn1.forward(out);
        let out = self.relu.forward(out);
        let out = self.conv2.forward(out);
        let out = self.bn2.forward(out);
        let out = self.relu.forward(out);
        let out = self.conv3.forward(out);
        let out = self.bn3.forward(out);

        // Skip connection
        let out = match &self.downsample {
            Some(downsample) => out + downsample.forward(identity),
            None => out + identity,
        };

        // Activation
        self.relu.forward(out)
    }

    /// Create a new Bottleneck.
    pub fn new(in_channels: usize, out_channels: usize, stride: usize, device: &Device<B>) -> Self {
        // Intermediate output channels w/ expansion = 4
        let int_out_channels = out_channels / 4;

        let initializer = Initializer::KaimingNormal {
            gain: SQRT_2,
            fan_out_only: true,
        };

        // conv1x1
        let conv1 = Conv2dConfig::new([in_channels, int_out_channels], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .with_bias(false)
            .with_initializer(initializer.clone())
            .init(device);
        let bn1 = BatchNormConfig::new(int_out_channels).init(device);

        // conv3x3
        let conv2 = Conv2dConfig::new([int_out_channels, int_out_channels], [3, 3])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .with_bias(false)
            .with_initializer(initializer.clone())
            .init(device);
        let bn2 = BatchNormConfig::new(int_out_channels).init(device);

        // conv1x1
        let conv3 = Conv2dConfig::new([int_out_channels, out_channels], [1, 1])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .with_bias(false)
            .with_initializer(initializer)
            .init(device);
        let bn3 = BatchNormConfig::new(out_channels).init(device);

        let downsample = (stride != 1 || in_channels != out_channels)
            .then(|| Downsample::new(in_channels, out_channels, stride, device));

        Self {
            conv1,
            bn1,
            relu: Relu::new(),
            conv2,
            bn2,
            conv3,
            bn3,
            downsample,
        }
    }
}

/// Downsample layer applies a 1x1 conv to reduce the resolution (H, W) and adjust the number of channels.
#[derive(Module, Debug)]
pub struct Downsample<B: Backend> {
    conv: Conv2d<B>,
    bn: BatchNorm<B, 2>,
}

impl<B: Backend> Downsample<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let out = self.conv.forward(input);
        self.bn.forward(out)
    }

    /// Create a new Downsample.
    pub fn new(in_channels: usize, out_channels: usize, stride: usize, device: &Device<B>) -> Self {
        let initializer = Initializer::KaimingNormal {
            gain: SQRT_2,
            fan_out_only: true,
        };

        // conv1x1
        let conv = Conv2dConfig::new([in_channels, out_channels], [1, 1])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .with_bias(false)
            .with_initializer(initializer)
            .init(device);
        let bn = BatchNormConfig::new(out_channels).init(device);

        Self { conv, bn }
    }
}

/// Collection of sequential residual blocks.
#[derive(Module, Debug)]
pub struct LayerBlock<B: Backend> {
    blocks: Vec<ResidualBlock<B>>,
}

impl<B: Backend> LayerBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let mut out = input;
        for block in &self.blocks {
            out = block.forward(out);
        }
        out
    }

    /// Create a new LayerBlock.
    pub fn new(
        num_blocks: usize,
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        bottleneck: bool,
        device: &Device<B>,
    ) -> Self {
        let blocks = (0..num_blocks)
            .map(|b| {
                if b == 0 {
                    // First block uses the specified stride
                    if bottleneck {
                        ResidualBlock::Bottleneck(Bottleneck::new(
                            in_channels,
                            out_channels,
                            stride,
                            device,
                        ))
                    } else {
                        ResidualBlock::Basic(BasicBlock::new(
                            in_channels,
                            out_channels,
                            stride,
                            device,
                        ))
                    }
                } else {
                    // Other blocks use a stride of 1
                    if bottleneck {
                        ResidualBlock::Bottleneck(Bottleneck::new(
                            out_channels,
                            out_channels,
                            1,
                            device,
                        ))
                    } else {
                        ResidualBlock::Basic(BasicBlock::new(out_channels, out_channels, 1, device))
                    }
                }
            })
            .collect();

        Self { blocks }
    }
}
