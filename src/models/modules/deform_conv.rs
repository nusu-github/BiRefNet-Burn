use std::ops::Deref;

use burn::{
    module::Param,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        PaddingConfig2d,
    },
    prelude::*,
    tensor::activation::sigmoid,
};

use crate::special::deform_conv2d;

#[derive(Config, Debug)]
pub struct DeformableConv2dConfig {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    bias: bool,
}

impl DeformableConv2dConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> DeformableConv2d<B> {
        let mut offset_conv = Conv2dConfig::new(
            [self.in_channels, 2 * self.kernel_size * self.kernel_size],
            [self.kernel_size, self.kernel_size],
        )
        .with_stride([self.stride, self.stride])
        .with_padding(PaddingConfig2d::Explicit(self.padding, self.padding))
        .with_bias(true)
        .init(device);

        offset_conv.weight = Param::from_tensor(offset_conv.weight.deref().zeros_like());
        offset_conv.bias = Some(Param::from_tensor(
            offset_conv.bias.unwrap().deref().zeros_like(),
        ));

        let mut modulator_conv = Conv2dConfig::new(
            [self.in_channels, self.kernel_size * self.kernel_size],
            [self.kernel_size, self.kernel_size],
        )
        .with_stride([self.stride, self.stride])
        .with_padding(PaddingConfig2d::Explicit(self.padding, self.padding))
        .with_bias(true)
        .init(device);

        modulator_conv.weight = Param::from_tensor(modulator_conv.weight.deref().zeros_like());
        modulator_conv.bias = Some(Param::from_tensor(
            modulator_conv.bias.unwrap().deref().zeros_like(),
        ));

        let regular_conv = Conv2dConfig::new(
            [self.in_channels, self.out_channels],
            [self.kernel_size, self.kernel_size],
        )
        .with_stride([self.stride, self.stride])
        .with_padding(PaddingConfig2d::Explicit(self.padding, self.padding))
        .with_bias(self.bias)
        .init(device);

        DeformableConv2d {
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            kernel_size: self.kernel_size,
            stride: self.stride,
            padding: self.padding,
            bias: self.bias,
            modulator_conv,
            offset_conv,
            regular_conv,
        }
    }
}

#[derive(Module, Debug)]
pub struct DeformableConv2d<B: Backend> {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    bias: bool,
    modulator_conv: Conv2d<B>,
    offset_conv: Conv2d<B>,
    regular_conv: Conv2d<B>,
}

impl<B: Backend> DeformableConv2d<B> {
    pub fn forward(&self, x: Tensor<B, 4, Float>) -> Tensor<B, 4, Float> {
        let offset = self.offset_conv.forward(x.clone());
        let modulator = sigmoid(self.modulator_conv.forward(x.clone())).mul_scalar(2.0);

        // TODO: 公式実装待ち
        // ダミー [b, c*4, h, w]
        let [b, c, h, w] = x.dims();
        let dammy = Tensor::<B, 4, Float>::zeros([b, c * 4, h, w], &x.device());

        return dammy;

        todo!();

        let x = deform_conv2d(
            x,
            offset,
            self.regular_conv.weight.deref().clone(),
            Option::from(self.regular_conv.bias.clone().unwrap().deref().clone()),
            (self.stride, self.stride),
            (self.padding, self.padding),
            (1, 1),
            Some(modulator),
        );

        x
    }
}
