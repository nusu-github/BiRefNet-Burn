//! # Deformable Convolution v2
//!
//! This module provides an implementation of Deformable Convolution v2, which can
//! enhance the model's ability to handle geometric transformations.

use burn::{
    module::Param,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        PaddingConfig2d,
    },
    prelude::*,
    tensor::{activation::sigmoid, module::deform_conv2d, ops::DeformConvOptions},
};

use crate::error::{BiRefNetError, BiRefNetResult};

/// Configuration for the `DeformableConv2d` module.
#[derive(Config, Debug)]
pub struct DeformableConv2dConfig {
    /// Number of input channels.
    in_channels: usize,
    /// Number of output channels.
    out_channels: usize,
    /// Size of the convolutional kernel.
    #[config(default = "3")]
    kernel_size: usize,
    /// Stride of the convolution.
    #[config(default = "1")]
    stride: usize,
    /// Padding added to all four sides of the input.
    #[config(default = "1")]
    padding: usize,
    /// Whether to include a bias term.
    #[config(default = "false")]
    bias: bool,
}

impl DeformableConv2dConfig {
    /// Initializes a new `DeformableConv2d` module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> BiRefNetResult<DeformableConv2d<B>> {
        let mut offset_conv = Conv2dConfig::new(
            [self.in_channels, 2 * self.kernel_size * self.kernel_size],
            [self.kernel_size, self.kernel_size],
        )
        .with_stride([self.stride, self.stride])
        .with_padding(PaddingConfig2d::Explicit(self.padding, self.padding))
        .init(device);

        offset_conv.weight = Param::from_tensor(offset_conv.weight.val().zeros_like());
        offset_conv.bias = Some(Param::from_tensor(
            offset_conv
                .bias
                .ok_or_else(|| {
                    BiRefNetError::ModelInitializationFailed {
                        reason: "Offset conv bias parameter is missing".to_owned(),
                    }
                })?
                .val()
                .zeros_like(),
        ));

        let mut modulator_conv =
            Conv2dConfig::new([self.in_channels, self.kernel_size * self.kernel_size], [
                self.kernel_size,
                self.kernel_size,
            ])
            .with_stride([self.stride, self.stride])
            .with_padding(PaddingConfig2d::Explicit(self.padding, self.padding))
            .init(device);

        modulator_conv.weight = Param::from_tensor(modulator_conv.weight.val().zeros_like());
        modulator_conv.bias = Some(Param::from_tensor(
            modulator_conv
                .bias
                .ok_or_else(|| {
                    BiRefNetError::ModelInitializationFailed {
                        reason: "Modulator conv bias parameter is missing".to_owned(),
                    }
                })?
                .val()
                .zeros_like(),
        ));

        let regular_conv = Conv2dConfig::new([self.in_channels, self.out_channels], [
            self.kernel_size,
            self.kernel_size,
        ])
        .with_stride([self.stride, self.stride])
        .with_padding(PaddingConfig2d::Explicit(self.padding, self.padding))
        .with_bias(self.bias)
        .init(device);

        Ok(DeformableConv2d {
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            kernel_size: self.kernel_size,
            stride: self.stride,
            padding: self.padding,
            bias: self.bias,
            modulator_conv,
            offset_conv,
            regular_conv,
        })
    }
}

/// Deformable Convolution v2 layer.
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
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let offset = self.offset_conv.forward(x.clone());
        let modulator = sigmoid(self.modulator_conv.forward(x.clone())).mul_scalar(2.0);

        // Calculate groups dynamically like PyTorch implementation
        let [_, _, weights_h, weights_w] = self.regular_conv.weight.dims();
        let [_, n_in_channels, _, _] = x.dims();

        // PyTorch calculation: n_offset_grps = offset.shape[1] // (2 * weights_h * weights_w)
        let n_offset_grps = offset.dims()[1] / (2 * weights_h * weights_w);

        // PyTorch calculation: n_weight_grps = n_in_channels // weight.shape[1]
        let n_weight_grps = n_in_channels / self.regular_conv.weight.dims()[1];

        deform_conv2d(
            x,
            offset,
            self.regular_conv.weight.val(),
            Option::from(modulator),
            self.regular_conv.bias.as_ref().map(Param::val),
            DeformConvOptions {
                stride: [self.stride, self.stride],
                padding: [self.padding, self.padding],
                dilation: [1, 1],
                weight_groups: n_weight_grps,
                offset_groups: n_offset_grps,
            },
        )
    }
}
