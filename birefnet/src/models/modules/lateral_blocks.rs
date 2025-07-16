//! # Lateral Blocks
//!
//! This module defines the lateral connection blocks used in the decoder of BiRefNet.
//! These blocks are typically simple convolutions that process features from the encoder
//! before they are merged with the decoder path.

use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        PaddingConfig2d,
    },
    prelude::*,
};

/// Configuration for the `BasicLatBlk` module.
#[derive(Config, Debug)]
pub struct BasicLatBlkConfig {
    /// Number of input channels.
    #[config(default = "64")]
    in_channels: usize,
    /// Number of output channels.
    #[config(default = "64")]
    out_channels: usize,
    /// Number of intermediate channels (not used in this block).
    #[config(default = "64")]
    inter_channels: usize,
}

impl BasicLatBlkConfig {
    /// Initializes a new `BasicLatBlk` module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> BasicLatBlk<B> {
        BasicLatBlk {
            conv: Conv2dConfig::new([self.in_channels, self.out_channels], [1, 1])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Valid)
                .init(device),
        }
    }
}

/// A basic lateral block, consisting of a single 1x1 convolution.
#[derive(Module, Debug)]
pub struct BasicLatBlk<B: Backend> {
    conv: Conv2d<B>,
}

impl<B: Backend> BasicLatBlk<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv.forward(x)
    }
}
