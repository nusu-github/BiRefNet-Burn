//! `ResNet` backbone implementations for `BiRefNet`.

use core::f64::consts::SQRT_2;

use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Initializer, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    prelude::*,
};

mod blocks;
use self::blocks::LayerBlock;
use crate::{ResNetVariant, error::BackboneError};

const RESNET18_BLOCKS: [usize; 4] = [2, 2, 2, 2];
const RESNET34_BLOCKS: [usize; 4] = [3, 4, 6, 3];
const RESNET50_BLOCKS: [usize; 4] = [3, 4, 6, 3];
const RESNET101_BLOCKS: [usize; 4] = [3, 4, 23, 3];
const RESNET152_BLOCKS: [usize; 4] = [3, 8, 36, 3];

/// `ResNet` multi-scale backbone returning 4 feature levels.
#[derive(Module, Debug)]
pub struct ResNetBackbone {
    conv1_block: Conv1Block,
    layer1: LayerBlock,
    layer2: LayerBlock,
    layer3: LayerBlock,
    layer4: LayerBlock,
    out_channels: [usize; 4],
}

impl ResNetBackbone {
    /// Extracts 4 hierarchical feature maps from `[B, 3, H, W]`.
    #[must_use]
    pub fn forward(&self, input: Tensor<4>) -> [Tensor<4>; 4] {
        let conv1 = self.layer1.forward(self.conv1_block.forward(input));
        let conv2 = self.layer2.forward(conv1.clone());
        let conv3 = self.layer3.forward(conv2.clone());
        let conv4 = self.layer4.forward(conv3.clone());
        [conv1, conv2, conv3, conv4]
    }

    /// Returns the output channel counts `[C1, C2, C3, C4]` for this `ResNet` variant.
    #[must_use]
    pub const fn output_channels(&self) -> [usize; 4] {
        self.out_channels
    }

    /// Initializes a ResNet-18 backbone.
    #[must_use]
    pub fn resnet18(device: &Device) -> Self {
        Self::new(RESNET18_BLOCKS, 1, device)
    }

    /// Initializes a ResNet-34 backbone.
    #[must_use]
    pub fn resnet34(device: &Device) -> Self {
        Self::new(RESNET34_BLOCKS, 1, device)
    }

    /// Initializes a ResNet-50 backbone.
    #[must_use]
    pub fn resnet50(device: &Device) -> Self {
        Self::new(RESNET50_BLOCKS, 4, device)
    }

    /// Initializes a ResNet-101 backbone.
    #[must_use]
    pub fn resnet101(device: &Device) -> Self {
        Self::new(RESNET101_BLOCKS, 4, device)
    }

    /// Initializes a ResNet-152 backbone.
    #[must_use]
    pub fn resnet152(device: &Device) -> Self {
        Self::new(RESNET152_BLOCKS, 4, device)
    }

    fn new(blocks: [usize; 4], expansion: usize, device: &Device) -> Self {
        debug_assert!(expansion == 1 || expansion == 4);
        let bottleneck = expansion > 1;
        let out_channels = [
            64 * expansion,
            128 * expansion,
            256 * expansion,
            512 * expansion,
        ];

        Self {
            conv1_block: Conv1Block::new(3, 64, device),
            layer1: LayerBlock::new(blocks[0], 64, out_channels[0], 1, bottleneck, device),
            layer2: LayerBlock::new(
                blocks[1],
                out_channels[0],
                out_channels[1],
                2,
                bottleneck,
                device,
            ),
            layer3: LayerBlock::new(
                blocks[2],
                out_channels[1],
                out_channels[2],
                2,
                bottleneck,
                device,
            ),
            layer4: LayerBlock::new(
                blocks[3],
                out_channels[2],
                out_channels[3],
                2,
                bottleneck,
                device,
            ),
            out_channels,
        }
    }
}

/// `ResNet` stem (`7x7` conv -> batch norm -> `ReLU` -> `3x3` max-pool).
#[derive(Module, Debug)]
pub struct Conv1Block {
    conv1: Conv2d,
    bn1: BatchNorm,
    relu: Relu,
    maxpool: MaxPool2d,
}

impl Conv1Block {
    /// Applies the `ResNet` stem block.
    #[must_use]
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let out = self
            .relu
            .forward(self.bn1.forward(self.conv1.forward(input)));
        self.maxpool.forward(out)
    }

    /// Initializes the `ResNet` stem block.
    #[must_use]
    pub fn new(in_channels: usize, out_channels: usize, device: &Device) -> Self {
        let initializer = Initializer::KaimingNormal {
            gain: SQRT_2,
            fan_out_only: true,
        };
        let conv1 = Conv2dConfig::new([in_channels, out_channels], [7, 7])
            .with_stride([2, 2])
            .with_padding(PaddingConfig2d::Explicit(3, 3, 3, 3))
            .with_bias(false)
            .with_initializer(initializer)
            .init(device);
        let bn1 = BatchNormConfig::new(out_channels).init(device);
        let maxpool = MaxPool2dConfig::new([3, 3])
            .with_strides([2, 2])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init();

        Self {
            conv1,
            bn1,
            relu: Relu::new(),
            maxpool,
        }
    }
}

/// Configuration for [`ResNetBackbone`].
#[derive(Config, Debug, PartialEq, Eq)]
pub struct ResNetConfig {
    /// Block depths per stage.
    pub layers: [usize; 4],
    /// Channel expansion factor (`1` for `BasicBlock`, `4` for `Bottleneck`).
    #[config(default = 4)]
    pub expansion: usize,
}

impl ResNetConfig {
    /// Returns the configuration matching a [`ResNetVariant`].
    #[must_use]
    pub const fn from_variant(variant: ResNetVariant) -> Self {
        match variant {
            ResNetVariant::ResNet18 => Self::resnet18(),
            ResNetVariant::ResNet34 => Self::resnet34(),
            ResNetVariant::ResNet50 => Self::resnet50(),
            ResNetVariant::ResNet101 => Self::resnet101(),
            ResNetVariant::ResNet152 => Self::resnet152(),
        }
    }

    /// ResNet-18 configuration (`[2, 2, 2, 2]`, expansion 1).
    #[must_use]
    pub const fn resnet18() -> Self {
        Self {
            layers: RESNET18_BLOCKS,
            expansion: 1,
        }
    }

    /// ResNet-34 configuration (`[3, 4, 6, 3]`, expansion 1).
    #[must_use]
    pub const fn resnet34() -> Self {
        Self {
            layers: RESNET34_BLOCKS,
            expansion: 1,
        }
    }

    /// ResNet-50 configuration (`[3, 4, 6, 3]`, expansion 4).
    #[must_use]
    pub const fn resnet50() -> Self {
        Self {
            layers: RESNET50_BLOCKS,
            expansion: 4,
        }
    }

    /// ResNet-101 configuration (`[3, 4, 23, 3]`, expansion 4).
    #[must_use]
    pub const fn resnet101() -> Self {
        Self {
            layers: RESNET101_BLOCKS,
            expansion: 4,
        }
    }

    /// ResNet-152 configuration (`[3, 8, 36, 3]`, expansion 4).
    #[must_use]
    pub const fn resnet152() -> Self {
        Self {
            layers: RESNET152_BLOCKS,
            expansion: 4,
        }
    }

    /// Validates configuration parameters and initializes a [`ResNetBackbone`].
    ///
    /// # Errors
    ///
    /// Returns [`BackboneError`] if `expansion` is not `1` or `4` or any stage depth is `0`.
    pub fn init(&self, device: &Device) -> Result<ResNetBackbone, BackboneError> {
        if self.expansion != 1 && self.expansion != 4 {
            return Err(BackboneError::invalid_config(format!(
                "ResNet expansion must be 1 or 4, got {}",
                self.expansion
            )));
        }
        if self.layers.contains(&0) {
            return Err(BackboneError::invalid_config(format!(
                "ResNet stage block counts must be positive, got {:?}",
                self.layers
            )));
        }
        Ok(ResNetBackbone::new(self.layers, self.expansion, device))
    }
}
