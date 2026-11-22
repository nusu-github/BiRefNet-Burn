//! Deformable Convolution v2 layer used in deformable ASPP branches.

use burn::{
    module::Param,
    nn::{
        Initializer, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::{activation::sigmoid, module::deform_conv2d, ops::DeformConvOptions},
};

/// Configuration for [`DeformableConv2d`].
#[derive(Config, Debug)]
pub struct DeformableConv2dConfig {
    /// Number of input channels.
    in_channels: usize,
    /// Number of output channels.
    out_channels: usize,
    /// Size of the convolutional kernel.
    #[config(default = 3)]
    kernel_size: usize,
    /// Stride of the convolution.
    #[config(default = 1)]
    stride: usize,
    /// Padding added to all four sides of the input.
    #[config(default = 1)]
    padding: usize,
    /// Whether to include a bias term on the main convolution.
    #[config(default = false)]
    bias: bool,
}

/// `DCNv2` modulation range: `2 * sigmoid(.)` maps zero-initialized logits to `1.0` and lets
/// each sampling location be scaled into `(0, 2)`.
const MODULATOR_SCALE: f64 = 2.0;

impl DeformableConv2dConfig {
    /// Initializes a [`DeformableConv2d`] module with zero-initialized offsets and modulators.
    #[must_use]
    pub fn init(&self, device: &Device) -> DeformableConv2d {
        let padding =
            PaddingConfig2d::Explicit(self.padding, self.padding, self.padding, self.padding);

        // `Initializer::Zeros` also zero-initializes the bias, so offsets start at 0 and
        // modulators at `MODULATOR_SCALE * sigmoid(0) = 1`, i.e. a regular convolution.
        let offset_conv = Conv2dConfig::new(
            [self.in_channels, 2 * self.kernel_size * self.kernel_size],
            [self.kernel_size, self.kernel_size],
        )
        .with_stride([self.stride, self.stride])
        .with_padding(padding.clone())
        .with_initializer(Initializer::Zeros)
        .init(device);

        let modulator_conv = Conv2dConfig::new(
            [self.in_channels, self.kernel_size * self.kernel_size],
            [self.kernel_size, self.kernel_size],
        )
        .with_stride([self.stride, self.stride])
        .with_padding(padding.clone())
        .with_initializer(Initializer::Zeros)
        .init(device);

        let regular_conv = Conv2dConfig::new(
            [self.in_channels, self.out_channels],
            [self.kernel_size, self.kernel_size],
        )
        .with_stride([self.stride, self.stride])
        .with_padding(padding)
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

/// Deformable Convolution v2 layer.
#[derive(Module, Debug)]
pub struct DeformableConv2d {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    bias: bool,
    modulator_conv: Conv2d,
    offset_conv: Conv2d,
    regular_conv: Conv2d,
}

impl DeformableConv2d {
    /// Applies deformable convolution over `[B, in_channels, H, W]`.
    #[must_use]
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let offset = self.offset_conv.forward(x.clone());
        let modulator = sigmoid(self.modulator_conv.forward(x.clone())).mul_scalar(MODULATOR_SCALE);

        let [_, _, weights_h, weights_w] = self.regular_conv.weight.dims();
        let [_, n_in_channels, _, _] = x.dims();

        let n_offset_grps = offset.dims()[1] / (2 * weights_h * weights_w);
        let n_weight_grps = n_in_channels / self.regular_conv.weight.dims()[1];

        deform_conv2d(
            x,
            offset,
            self.regular_conv.weight.val(),
            Some(modulator),
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
