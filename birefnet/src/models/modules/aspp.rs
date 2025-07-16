//! # Atrous Spatial Pyramid Pooling (ASPP)
//!
//! This module implements the ASPP and Deformable ASPP blocks, which are crucial
//! for capturing multi-scale contextual information in semantic segmentation tasks.

use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, PaddingConfig2d, Relu,
    },
    prelude::*,
    tensor::{
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};

use super::DeformableConv2d;
use crate::error::BiRefNetResult;
use crate::models::DeformableConv2dConfig;

/// Configuration for the `_ASPPModule`.
#[derive(Debug)]
pub struct _ASPPModuleConfig {
    in_channels: usize,
    planes: usize,
    kernel_size: usize,
    padding: usize,
    dilation: usize,
}

impl _ASPPModuleConfig {
    /// Create a new instance of the ASPP module [config](_ASPPModuleConfig).
    pub const fn new(
        in_channels: usize,
        planes: usize,
        kernel_size: usize,
        padding: usize,
        dilation: usize,
    ) -> Self {
        Self {
            in_channels,
            planes,
            kernel_size,
            padding,
            dilation,
        }
    }

    /// Initializes a new `_ASPPModule`.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> _ASPPModule<B> {
        let atrous_conv = Conv2dConfig::new(
            [self.in_channels, self.planes],
            [self.kernel_size, self.kernel_size],
        )
        .with_stride([1, 1])
        .with_padding(PaddingConfig2d::Explicit(self.padding, self.padding))
        .with_dilation([self.dilation, self.dilation])
        .with_bias(false)
        .init(device);

        // TODO: Add batch_size condition like PyTorch (batch_size > 1)
        let bn = Some(BatchNormConfig::new(self.planes).init(device));
        let relu = Relu::new();

        _ASPPModule {
            atrous_conv,
            bn,
            relu,
        }
    }
}

/// A single branch of the ASPP module, consisting of an atrous convolution.
#[derive(Module, Debug)]
pub struct _ASPPModule<B: Backend> {
    atrous_conv: Conv2d<B>,
    bn: Option<BatchNorm<B, 2>>,
    relu: Relu,
}

impl<B: Backend> _ASPPModule<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.atrous_conv.forward(x);
        let x = match &self.bn {
            Some(bn) => bn.forward(x),
            None => x,
        };

        self.relu.forward(x)
    }
}

/// Configuration for the `ASPP` module.
#[derive(Config, Debug)]
pub struct ASPPConfig {
    /// Number of input channels.
    #[config(default = "64")]
    in_channels: usize,
    /// Number of output channels. Defaults to `in_channels`.
    #[config(default = "None")]
    out_channels: Option<usize>,
    /// The output stride of the backbone.
    #[config(default = "16")]
    output_stride: usize,
}

impl ASPPConfig {
    /// Initializes a new `ASPP` module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> ASPP<B> {
        let out_channels = self.out_channels.unwrap_or(self.in_channels);
        let in_channelster = 256; // Equivalent to PyTorch's self.in_channelster = 256 // self.down_scale

        // Determine dilations based on output_stride
        let dilations = if self.output_stride == 16 {
            [1, 6, 12, 18]
        } else if self.output_stride == 8 {
            [1, 12, 24, 36]
        } else {
            panic!(
                "Unsupported output_stride: {}. Only 8 and 16 are supported.",
                self.output_stride
            );
        };

        let aspp1 = _ASPPModuleConfig::new(self.in_channels, in_channelster, 1, 0, dilations[0])
            .init(device);
        let aspp2 = _ASPPModuleConfig::new(
            self.in_channels,
            in_channelster,
            3,
            dilations[1],
            dilations[1],
        )
        .init(device);
        let aspp3 = _ASPPModuleConfig::new(
            self.in_channels,
            in_channelster,
            3,
            dilations[2],
            dilations[2],
        )
        .init(device);
        let aspp4 = _ASPPModuleConfig::new(
            self.in_channels,
            in_channelster,
            3,
            dilations[3],
            dilations[3],
        )
        .init(device);

        // Global average pooling branch
        let global_avg_pool_0 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let global_avg_pool_1 = Conv2dConfig::new([self.in_channels, in_channelster], [1, 1])
            .with_stride([1, 1])
            .with_bias(false)
            .init(device);
        let global_avg_pool_2 = Some(BatchNormConfig::new(in_channelster).init(device));
        let global_avg_pool_3 = Relu::new();

        // Final convolution
        let conv1 = Conv2dConfig::new([in_channelster * 5, out_channels], [1, 1])
            .with_bias(false)
            .init(device);
        let bn1 = Some(BatchNormConfig::new(out_channels).init(device));
        let relu = Relu::new();
        let dropout = DropoutConfig::new(0.5).init();

        ASPP {
            aspp1,
            aspp2,
            aspp3,
            aspp4,
            global_avg_pool_0,
            global_avg_pool_1,
            global_avg_pool_2,
            global_avg_pool_3,
            conv1,
            bn1,
            relu,
            dropout,
        }
    }
}

/// Atrous Spatial Pyramid Pooling (ASPP) module.
#[derive(Module, Debug)]
pub struct ASPP<B: Backend> {
    aspp1: _ASPPModule<B>,
    aspp2: _ASPPModule<B>,
    aspp3: _ASPPModule<B>,
    aspp4: _ASPPModule<B>,
    // nn.Sequential
    global_avg_pool_0: AdaptiveAvgPool2d,
    global_avg_pool_1: Conv2d<B>,
    global_avg_pool_2: Option<BatchNorm<B, 2>>,
    global_avg_pool_3: Relu,
    conv1: Conv2d<B>,
    bn1: Option<BatchNorm<B, 2>>,
    relu: Relu,
    dropout: Dropout,
}

impl<B: Backend> ASPP<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x1 = self.aspp1.forward(x.clone());
        let x2 = self.aspp2.forward(x.clone());
        let x3 = self.aspp3.forward(x.clone());
        let x4 = self.aspp4.forward(x.clone());

        // Global average pooling branch
        let x5 = self.global_avg_pool_0.forward(x);
        let x5 = self.global_avg_pool_1.forward(x5);
        let x5 = match &self.global_avg_pool_2 {
            Some(bn) => bn.forward(x5),
            None => x5,
        };
        let x5 = self.global_avg_pool_3.forward(x5);

        // Interpolate to match x1 size
        let [_, _, d3, d4] = x1.dims();
        let x5 = interpolate(
            x5,
            [d3, d4],
            InterpolateOptions::new(InterpolateMode::Bilinear),
            // Note: PyTorch uses align_corners=True, but not available in current Burn version
        );

        // Concatenate all branches
        let x = Tensor::cat(vec![x1, x2, x3, x4, x5], 1);

        // Final convolution
        let x = self.conv1.forward(x);
        let x = match &self.bn1 {
            Some(bn) => bn.forward(x),
            None => x,
        };
        let x = self.relu.forward(x);

        self.dropout.forward(x)
    }
}

/// Configuration for the `_ASPPModuleDeformable`.
#[derive(Config, Debug)]
pub struct _ASPPModuleDeformableConfig {
    in_channels: usize,
    planes: usize,
    kernel_size: usize,
    padding: usize,
}

impl _ASPPModuleDeformableConfig {
    /// Initializes a new `_ASPPModuleDeformable`.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> BiRefNetResult<_ASPPModuleDeformable<B>> {
        let atrous_conv = DeformableConv2dConfig::new(self.in_channels, self.planes)
            .with_kernel_size(self.kernel_size)
            .with_stride(1)
            .with_padding(self.padding)
            .with_bias(false)
            .init(device)?;
        let bn = BatchNormConfig::new(self.planes).init(device);
        let relu = Relu::new();
        Ok(_ASPPModuleDeformable {
            atrous_conv,
            bn: Some(bn),
            relu,
        })
    }
}

/// A single branch of the Deformable ASPP module.
#[derive(Module, Debug)]
pub struct _ASPPModuleDeformable<B: Backend> {
    atrous_conv: DeformableConv2d<B>,
    bn: Option<BatchNorm<B, 2>>,
    relu: Relu,
}

impl<B: Backend> _ASPPModuleDeformable<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.atrous_conv.forward(x);
        let x = match &self.bn {
            Some(bn) => bn.forward(x),
            None => x,
        };

        self.relu.forward(x)
    }
}

/// Configuration for the `ASPPDeformable` module.
#[derive(Config, Debug)]
pub struct ASPPDeformableConfig {
    /// Number of input channels.
    #[config(default = "64")]
    in_channels: usize,
    /// Number of output channels. Defaults to `in_channels`.
    #[config(default = "None")]
    out_channels: Option<usize>,
    /// Kernel sizes for the parallel deformable convolution blocks.
    #[config(default = "[1, 3, 7]")]
    parallel_block_sizes: [usize; 3],
}

impl ASPPDeformableConfig {
    /// Initializes a new `ASPPDeformable` module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> BiRefNetResult<ASPPDeformable<B>> {
        let out_channels = self.out_channels.unwrap_or(self.in_channels);
        let in_channelster = 256;

        let aspp1 = _ASPPModuleDeformableConfig::new(self.in_channels, in_channelster, 1, 0)
            .init(device)?;
        let aspp_deforms = self
            .parallel_block_sizes
            .iter()
            .map(|&conv_size| {
                _ASPPModuleDeformableConfig::new(
                    self.in_channels,
                    in_channelster,
                    conv_size,
                    conv_size / 2,
                )
                .init(device)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let global_avg_pool_0 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let global_avg_pool_1 = Conv2dConfig::new([self.in_channels, in_channelster], [1, 1])
            .with_stride([1, 1])
            .with_bias(false)
            .init(device);
        let global_avg_pool_2 = BatchNormConfig::new(in_channelster).init(device);
        let global_avg_pool_3 = Relu::new();
        let conv1 = Conv2dConfig::new(
            [
                in_channelster * (2 + self.parallel_block_sizes.len()),
                out_channels,
            ],
            [1, 1],
        )
        .with_bias(false)
        .init(device);
        let bn1 = BatchNormConfig::new(out_channels).init(device);
        let relu = Relu::new();
        let dropout = DropoutConfig::new(0.5).init();

        Ok(ASPPDeformable {
            aspp1,
            aspp_deforms,
            global_avg_pool_0,
            global_avg_pool_1,
            global_avg_pool_2: Some(global_avg_pool_2),
            global_avg_pool_3,
            conv1,
            bn1: Some(bn1),
            relu,
            dropout,
        })
    }
}

/// ASPP module with Deformable Convolutions.
#[derive(Module, Debug)]
pub struct ASPPDeformable<B: Backend> {
    aspp1: _ASPPModuleDeformable<B>,
    aspp_deforms: Vec<_ASPPModuleDeformable<B>>,
    global_avg_pool_0: AdaptiveAvgPool2d,
    global_avg_pool_1: Conv2d<B>,
    global_avg_pool_2: Option<BatchNorm<B, 2>>,
    global_avg_pool_3: Relu,
    conv1: Conv2d<B>,
    bn1: Option<BatchNorm<B, 2>>,
    relu: Relu,
    dropout: Dropout,
}

impl<B: Backend> ASPPDeformable<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x1 = self.aspp1.forward(x.clone());
        let x_aspp_deforms = self
            .aspp_deforms
            .iter()
            .map(|aspp| aspp.forward(x.clone()))
            .collect::<Vec<_>>();
        let x5 = self.global_avg_pool_0.forward(x);
        let x5 = self.global_avg_pool_1.forward(x5);
        let x5 = match &self.global_avg_pool_2 {
            Some(bn) => bn.forward(x5),
            None => x5,
        };
        let x5 = self.global_avg_pool_3.forward(x5);
        let [_, _, d3, d4] = x1.dims();
        let x5 = interpolate(
            x5,
            [d3, d4],
            InterpolateOptions::new(InterpolateMode::Bilinear),
        );
        let mut x_ = vec![x1];
        x_.extend(x_aspp_deforms);
        x_.push(x5);
        let x = Tensor::cat(x_, 1);

        let x = self.conv1.forward(x);
        let x = match &self.bn1 {
            Some(bn) => bn.forward(x),
            None => x,
        };
        let x = self.relu.forward(x);

        self.dropout.forward(x)
    }
}
