//! VGG backbone implementations for `BiRefNet`.

use core::f64::consts::SQRT_2;

use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Initializer, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    prelude::*,
};

use crate::VggVariant;

#[derive(Clone, Copy, Debug)]
enum VggLayerSpec {
    Conv(usize),
    MaxPool,
}

const CONV1_SPEC: [VggLayerSpec; 6] = [
    VggLayerSpec::Conv(64),
    VggLayerSpec::Conv(64),
    VggLayerSpec::MaxPool,
    VggLayerSpec::Conv(128),
    VggLayerSpec::Conv(128),
    VggLayerSpec::MaxPool,
];
const CONV2_SPEC: [VggLayerSpec; 4] = [
    VggLayerSpec::Conv(256),
    VggLayerSpec::Conv(256),
    VggLayerSpec::Conv(256),
    VggLayerSpec::MaxPool,
];
const CONV3_SPEC: [VggLayerSpec; 4] = [
    VggLayerSpec::Conv(512),
    VggLayerSpec::Conv(512),
    VggLayerSpec::Conv(512),
    VggLayerSpec::MaxPool,
];
const CONV4_SPEC: [VggLayerSpec; 4] = [
    VggLayerSpec::Conv(512),
    VggLayerSpec::Conv(512),
    VggLayerSpec::Conv(512),
    VggLayerSpec::MaxPool,
];

/// VGG-16 multi-scale backbone returning 4 feature levels.
#[derive(Module, Debug)]
pub struct VggBackbone {
    conv1: VggFeatureBlock,
    conv2: VggFeatureBlock,
    conv3: VggFeatureBlock,
    conv4: VggFeatureBlock,
}

impl VggBackbone {
    /// Extracts 4 hierarchical feature maps from `[B, 3, H, W]`.
    #[must_use]
    pub fn forward(&self, input: Tensor<4>) -> [Tensor<4>; 4] {
        let conv1 = self.conv1.forward(input);
        let conv2 = self.conv2.forward(conv1.clone());
        let conv3 = self.conv3.forward(conv2.clone());
        let conv4 = self.conv4.forward(conv3.clone());
        [conv1, conv2, conv3, conv4]
    }

    /// Returns the fixed output channel counts `[128, 256, 512, 512]` for VGG-16.
    #[must_use]
    pub const fn output_channels(&self) -> [usize; 4] {
        [128, 256, 512, 512]
    }

    /// Initializes VGG-16 without batch normalization.
    #[must_use]
    pub fn vgg16(device: &Device) -> Self {
        Self::new(false, device)
    }

    /// Initializes VGG-16 with batch normalization.
    #[must_use]
    pub fn vgg16_bn(device: &Device) -> Self {
        Self::new(true, device)
    }

    fn new(batch_norm: bool, device: &Device) -> Self {
        Self {
            conv1: VggFeatureBlock::new(&CONV1_SPEC, 3, batch_norm, device),
            conv2: VggFeatureBlock::new(&CONV2_SPEC, 128, batch_norm, device),
            conv3: VggFeatureBlock::new(&CONV3_SPEC, 256, batch_norm, device),
            conv4: VggFeatureBlock::new(&CONV4_SPEC, 512, batch_norm, device),
        }
    }
}

/// Sequence of convolutional and pooling layers forming one VGG feature level.
#[derive(Module, Debug)]
struct VggFeatureBlock {
    layers: Vec<VggBlockLayer>,
}

impl VggFeatureBlock {
    fn forward(&self, mut input: Tensor<4>) -> Tensor<4> {
        for layer in &self.layers {
            input = layer.forward(input);
        }
        input
    }

    fn new(spec: &[VggLayerSpec], in_channels: usize, batch_norm: bool, device: &Device) -> Self {
        let mut layers = Vec::with_capacity(spec.len());
        let mut current_channels = in_channels;

        for layer_spec in spec {
            match layer_spec {
                VggLayerSpec::Conv(out_channels) => {
                    layers.push(VggBlockLayer::Conv(VggConvLayer::new(
                        current_channels,
                        *out_channels,
                        batch_norm,
                        device,
                    )));
                    current_channels = *out_channels;
                }
                VggLayerSpec::MaxPool => {
                    layers.push(VggBlockLayer::MaxPool(
                        MaxPool2dConfig::new([2, 2]).with_strides([2, 2]).init(),
                    ));
                }
            }
        }

        Self { layers }
    }
}

#[derive(Module, Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "Burn Module requires inline storage because Box<M> does not implement Module"
)]
enum VggBlockLayer {
    Conv(VggConvLayer),
    MaxPool(MaxPool2d),
}

impl VggBlockLayer {
    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        match self {
            Self::Conv(conv) => conv.forward(input),
            Self::MaxPool(pool) => pool.forward(input),
        }
    }
}

#[derive(Module, Debug)]
struct VggConvLayer {
    conv: Conv2d,
    batch_norm: Option<BatchNorm>,
    relu: Relu,
}

impl VggConvLayer {
    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let out = self.conv.forward(input);
        let out = match &self.batch_norm {
            Some(bn) => bn.forward(out),
            None => out,
        };
        self.relu.forward(out)
    }

    fn new(in_channels: usize, out_channels: usize, batch_norm: bool, device: &Device) -> Self {
        let initializer = Initializer::KaimingNormal {
            gain: SQRT_2,
            fan_out_only: true,
        };
        let conv = Conv2dConfig::new([in_channels, out_channels], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_bias(!batch_norm)
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

/// Configuration for [`VggBackbone`].
#[derive(Config, Debug, PartialEq, Eq)]
pub struct VggConfig {
    /// Whether batch normalization follows each convolution.
    #[config(default = false)]
    pub batch_norm: bool,
}

impl Default for VggConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl VggConfig {
    /// Returns the configuration matching a [`VggVariant`].
    #[must_use]
    pub const fn from_variant(variant: VggVariant) -> Self {
        match variant {
            VggVariant::Vgg16 => Self::vgg16(),
            VggVariant::Vgg16Bn => Self::vgg16_bn(),
        }
    }

    /// VGG-16 configuration without batch normalization.
    #[must_use]
    pub const fn vgg16() -> Self {
        Self { batch_norm: false }
    }

    /// VGG-16 configuration with batch normalization.
    #[must_use]
    pub const fn vgg16_bn() -> Self {
        Self { batch_norm: true }
    }

    /// Initializes a [`VggBackbone`].
    #[must_use]
    pub fn init(&self, device: &Device) -> VggBackbone {
        VggBackbone::new(self.batch_norm, device)
    }
}
