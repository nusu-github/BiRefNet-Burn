//! Lateral connection blocks used to merge encoder features into the decoder.

use burn::{
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
};

/// Configuration for [`BasicLatBlk`].
#[derive(Config, Debug)]
pub struct BasicLatBlkConfig {
    /// Number of input channels.
    #[config(default = 64)]
    in_channels: usize,
    /// Number of output channels.
    #[config(default = 64)]
    out_channels: usize,
}

impl BasicLatBlkConfig {
    /// Initializes a new [`BasicLatBlk`] module.
    #[must_use]
    pub fn init(&self, device: &Device) -> BasicLatBlk {
        BasicLatBlk {
            conv: Conv2dConfig::new([self.in_channels, self.out_channels], [1, 1])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Valid)
                .init(device),
        }
    }
}

/// Basic lateral block consisting of a single `1x1` convolution.
#[derive(Module, Debug)]
pub struct BasicLatBlk {
    conv: Conv2d,
}

impl BasicLatBlk {
    /// Projects encoder skip features with a `1x1` convolution.
    #[must_use]
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        self.conv.forward(x)
    }
}
