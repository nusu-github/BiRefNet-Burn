//! Lightweight two-layer convolutional block used for image-pyramid input paths.
#![expect(
    clippy::struct_field_names,
    reason = "Burn Config macro generates shadow builder structs mirroring *_channels fields"
)]

use burn::{
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
};

/// Configuration for [`SimpleConvs`].
#[derive(Config, Debug)]
pub struct SimpleConvsConfig {
    in_channels: usize,
    out_channels: usize,
    #[config(default = "64")]
    inter_channels: usize,
}

impl SimpleConvsConfig {
    /// Initializes a [`SimpleConvs`] module.
    #[must_use]
    pub fn init(&self, device: &Device) -> SimpleConvs {
        let conv1 = Conv2dConfig::new([self.in_channels, self.inter_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);

        let conv_out = Conv2dConfig::new([self.inter_channels, self.out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .init(device);

        SimpleConvs { conv1, conv_out }
    }
}

/// Two-layer convolutional block for decoder image-pyramid branches.
#[derive(Module, Debug)]
pub struct SimpleConvs {
    conv1: Conv2d,
    conv_out: Conv2d,
}

impl SimpleConvs {
    /// Applies both `3x3` convolution layers sequentially.
    #[must_use]
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        self.conv_out.forward(self.conv1.forward(x))
    }
}
