use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        PaddingConfig2d,
    },
    prelude::*,
};

/// A simple two-layer convolutional block.
#[derive(Config, Debug)]
pub struct SimpleConvsConfig {
    in_channels: usize,
    out_channels: usize,
    #[config(default = "64")]
    inter_channels: usize,
}

impl SimpleConvsConfig {
    /// Initializes a `SimpleConvs` module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> SimpleConvs<B> {
        let conv1 = Conv2dConfig::new([self.in_channels, self.inter_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        let conv_out = Conv2dConfig::new([self.inter_channels, self.out_channels], [3, 3])
            .with_padding(PaddingConfig2d::Explicit(1, 1))
            .init(device);

        SimpleConvs { conv1, conv_out }
    }
}

/// A simple two-layer convolutional block used for decoder input processing.
#[derive(Module, Debug)]
pub struct SimpleConvs<B: Backend> {
    conv1: Conv2d<B>,
    conv_out: Conv2d<B>,
}

impl<B: Backend> SimpleConvs<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        self.conv_out.forward(self.conv1.forward(x))
    }
}
