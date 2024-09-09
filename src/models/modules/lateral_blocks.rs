use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        PaddingConfig2d,
    },
    prelude::*,
};

#[derive(Config, Debug)]
pub struct BasicLatBlkConfig {
    #[config(default = "64")]
    in_channels: usize,
    #[config(default = "64")]
    out_channels: usize,
    #[config(default = "64")]
    inter_channels: usize,
}

impl BasicLatBlkConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> BasicLatBlk<B> {
        BasicLatBlk {
            conv: Conv2dConfig::new([self.in_channels, self.out_channels], [1, 1])
                .with_stride([1, 1])
                .with_padding(PaddingConfig2d::Valid)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct BasicLatBlk<B: Backend> {
    conv: Conv2d<B>,
}

impl<B: Backend> BasicLatBlk<B> {
    pub fn forward(&self, x: Tensor<B, 4, Float>) -> Tensor<B, 4, Float> {
        self.conv.forward(x)
    }
}
