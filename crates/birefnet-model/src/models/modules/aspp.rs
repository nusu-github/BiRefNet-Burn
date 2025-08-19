//! Atrous Spatial Pyramid Pooling (ASPP) modules for BiRefNet.
//!
//! This module provides standard and deformable ASPP implementations that capture
//! multi-scale contextual information for semantic segmentation tasks.

use burn::{
    module::Ignored,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, PaddingConfig2d, Relu,
    },
    prelude::*,
};

use super::{decoder_blocks::NormLayer, utils::intelligent_interpolate, DeformableConv2d};
use crate::{
    config::InterpolationStrategy, error::BiRefNetResult,
    models::modules::deform_conv::DeformableConv2dConfig,
};

/// Configuration for the individual ASPP module branch.
#[derive(Config, Debug)]
pub struct _ASPPModuleConfig {
    in_channels: usize,
    planes: usize,
    kernel_size: usize,
    padding: usize,
    dilation: usize,
    /// Batch size for conditional BatchNorm usage.
    #[config(default = "4")]
    batch_size: usize,
}

impl _ASPPModuleConfig {
    /// Creates a new `_ASPPModule` following Burn conventions.
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

        let bn = NormLayer::new(self.planes, self.batch_size, device);
        let relu = Relu::new();

        _ASPPModule {
            atrous_conv,
            bn,
            relu,
        }
    }
}

/// ASPP module branch sequential block matching PyTorch _ASPPModule structure.
///
/// Corresponds to PyTorch's _ASPPModule:
/// ```python
/// self.atrous_conv = nn.Conv2d(in_channels, planes, kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
/// self.bn = nn.BatchNorm2d(planes) if config.batch_size > 1 else nn.Identity()
/// self.relu = nn.ReLU(inplace=True)
/// ```
///
/// # Shapes
///   - input: `[batch_size, in_channels, height, width]`  
///   - output: `[batch_size, planes, height, width]`
#[derive(Module, Debug)]
pub struct _ASPPModule<B: Backend> {
    /// Component 0: Conv2d with atrous/dilation
    atrous_conv: Conv2d<B>,
    /// Component 1: BatchNorm2d or Identity (conditional)
    bn: NormLayer<B>,
    /// Component 2: ReLU(inplace=True)
    relu: Relu,
}

impl<B: Backend> _ASPPModule<B> {
    /// Sequential forward pass matching PyTorch _ASPPModule.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Component 0: Atrous convolution
        let x = self.atrous_conv.forward(x);
        // Component 1: Conditional normalization
        let x = self.bn.forward(x);
        // Component 2: ReLU activation
        self.relu.forward(x)
    }
}

/// Global average pooling sequential block matching PyTorch nn.Sequential structure.
///
/// Corresponds to PyTorch's:
/// ```python
/// nn.Sequential(
///     nn.AdaptiveAvgPool2d((1, 1)),
///     nn.Conv2d(in_channels, inter_channels, 1, stride=1, bias=False),
///     nn.BatchNorm2d(inter_channels) if config.batch_size > 1 else nn.Identity(),
///     nn.ReLU(inplace=True)
/// )
/// ```
#[derive(Module, Debug)]
pub struct GlobalAvgPool<B: Backend> {
    /// Component 0: AdaptiveAvgPool2d((1, 1))
    pool: AdaptiveAvgPool2d,
    /// Component 1: Conv2d(in_channels, inter_channels, 1)
    conv: Conv2d<B>,
    /// Component 2: BatchNorm2d or Identity
    bn: NormLayer<B>,
    /// Component 3: ReLU(inplace=True)
    relu: Relu,
}

impl<B: Backend> GlobalAvgPool<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        batch_size: usize,
        device: &Device<B>,
    ) -> Self {
        let pool = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let conv = Conv2dConfig::new([in_channels, out_channels], [1, 1])
            .with_stride([1, 1])
            .with_bias(false)
            .init(device);
        let bn = NormLayer::new(out_channels, batch_size, device);
        let relu = Relu::new();

        Self {
            pool,
            conv,
            bn,
            relu,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.pool.forward(x);
        let x = self.conv.forward(x);
        let x = self.bn.forward(x);
        self.relu.forward(x)
    }
}

/// Configuration for the main ASPP module.
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
    /// Batch size for conditional BatchNorm usage.
    #[config(default = "4")]
    batch_size: usize,
    /// Interpolation strategy for tensor resizing operations.
    interpolation_strategy: InterpolationStrategy,
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
            .with_batch_size(self.batch_size)
            .init(device);
        let aspp2 = _ASPPModuleConfig::new(
            self.in_channels,
            in_channelster,
            3,
            dilations[1],
            dilations[1],
        )
        .with_batch_size(self.batch_size)
        .init(device);
        let aspp3 = _ASPPModuleConfig::new(
            self.in_channels,
            in_channelster,
            3,
            dilations[2],
            dilations[2],
        )
        .with_batch_size(self.batch_size)
        .init(device);
        let aspp4 = _ASPPModuleConfig::new(
            self.in_channels,
            in_channelster,
            3,
            dilations[3],
            dilations[3],
        )
        .with_batch_size(self.batch_size)
        .init(device);

        // Global average pooling sequential branch
        let global_avg_pool =
            GlobalAvgPool::new(self.in_channels, in_channelster, self.batch_size, device);

        // Final convolution
        let conv1 = Conv2dConfig::new([in_channelster * 5, out_channels], [1, 1])
            .with_bias(false)
            .init(device);
        let bn1 = NormLayer::new(out_channels, self.batch_size, device);
        let relu = Relu::new();
        let dropout = DropoutConfig::new(0.5).init();

        let interpolation_strategy = Ignored(self.interpolation_strategy.clone());

        ASPP {
            aspp1,
            aspp2,
            aspp3,
            aspp4,
            global_avg_pool,
            conv1,
            bn1,
            relu,
            dropout,
            interpolation_strategy,
        }
    }
}

/// Atrous Spatial Pyramid Pooling (ASPP) module matching PyTorch structure.
///
/// Contains 4 parallel ASPP branches (aspp1-4) + 1 global avg pool branch,
/// followed by feature fusion and final processing.
///
/// # Shapes
///   - input: `[batch_size, in_channels, height, width]`
///   - output: `[batch_size, out_channels, height, width]`
#[derive(Module, Debug)]
pub struct ASPP<B: Backend> {
    aspp1: _ASPPModule<B>,
    aspp2: _ASPPModule<B>,
    aspp3: _ASPPModule<B>,
    aspp4: _ASPPModule<B>,
    /// Branch 5: Global average pooling sequential block (nn.Sequential in PyTorch)
    global_avg_pool: GlobalAvgPool<B>,
    conv1: Conv2d<B>,
    bn1: NormLayer<B>,
    relu: Relu,
    dropout: Dropout,
    /// Interpolation strategy for tensor resizing operations.
    interpolation_strategy: Ignored<InterpolationStrategy>,
}

impl<B: Backend> ASPP<B> {
    /// Forward pass through the ASPP module.
    ///
    /// # Shapes
    /// * `x` - Input tensor: `[batch_size, channels, height, width]`
    /// * Returns - Multi-scale feature tensor: `[batch_size, out_channels, height, width]`
    ///
    /// # Arguments
    /// * `x` - Input feature map from the backbone encoder
    ///
    /// # Returns
    /// Enhanced feature map combining multiple atrous convolution scales and global context
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x1 = self.aspp1.forward(x.clone());
        let x2 = self.aspp2.forward(x.clone());
        let x3 = self.aspp3.forward(x.clone());
        let x4 = self.aspp4.forward(x.clone());

        // Global average pooling sequential branch
        let x5 = self.global_avg_pool.forward(x);

        // Interpolate to match x1 size using intelligent interpolation
        let [_, _, d3, d4] = x1.dims();
        let x5 = intelligent_interpolate(x5, [d3, d4], &self.interpolation_strategy.0);

        // Concatenate all branches
        let x = Tensor::cat(vec![x1, x2, x3, x4, x5], 1);

        // Final convolution
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.relu.forward(x);

        self.dropout.forward(x)
    }
}

/// Configuration for the deformable ASPP module branch.
#[derive(Config, Debug)]
pub struct _ASPPModuleDeformableConfig {
    in_channels: usize,
    planes: usize,
    kernel_size: usize,
    padding: usize,
    /// Batch size for conditional BatchNorm usage.
    #[config(default = "4")]
    batch_size: usize,
}

impl _ASPPModuleDeformableConfig {
    /// Creates a new `_ASPPModuleDeformable` following Burn conventions.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> BiRefNetResult<_ASPPModuleDeformable<B>> {
        let atrous_conv = DeformableConv2dConfig::new(self.in_channels, self.planes)
            .with_kernel_size(self.kernel_size)
            .with_stride(1)
            .with_padding(self.padding)
            .with_bias(false)
            .init(device)?;
        let bn = NormLayer::new(self.planes, self.batch_size, device);
        let relu = Relu::new();
        Ok(_ASPPModuleDeformable {
            atrous_conv,
            bn,
            relu,
        })
    }
}

/// Deformable ASPP module branch sequential block matching PyTorch _ASPPModuleDeformable structure.
///
/// Corresponds to PyTorch's _ASPPModuleDeformable:
/// ```python
/// self.atrous_conv = DeformableConv2d(in_channels, planes, kernel_size, stride=1, padding=padding, bias=False)
/// self.bn = nn.BatchNorm2d(planes) if config.batch_size > 1 else nn.Identity()
/// self.relu = nn.ReLU(inplace=True)
/// ```
///
/// # Shapes
///   - input: `[batch_size, in_channels, height, width]`  
///   - output: `[batch_size, planes, height, width]`
#[derive(Module, Debug)]
pub struct _ASPPModuleDeformable<B: Backend> {
    /// Component 0: DeformableConv2d
    atrous_conv: DeformableConv2d<B>,
    /// Component 1: BatchNorm2d or Identity (conditional)
    bn: NormLayer<B>,
    /// Component 2: ReLU(inplace=True)
    relu: Relu,
}

impl<B: Backend> _ASPPModuleDeformable<B> {
    /// Sequential forward pass matching PyTorch _ASPPModuleDeformable.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        // Component 0: Deformable convolution
        let x = self.atrous_conv.forward(x);
        // Component 1: Conditional normalization
        let x = self.bn.forward(x);
        // Component 2: ReLU activation
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
    /// Batch size for conditional BatchNorm usage.
    #[config(default = "4")]
    batch_size: usize,
    /// Interpolation strategy for tensor resizing operations.
    interpolation_strategy: InterpolationStrategy,
}

impl ASPPDeformableConfig {
    /// Initializes a new `ASPPDeformable` module.
    pub fn init<B: Backend>(&self, device: &Device<B>) -> BiRefNetResult<ASPPDeformable<B>> {
        let out_channels = self.out_channels.unwrap_or(self.in_channels);
        let in_channelster = 256;

        let aspp1 = _ASPPModuleDeformableConfig::new(self.in_channels, in_channelster, 1, 0)
            .with_batch_size(self.batch_size)
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
                .with_batch_size(self.batch_size)
                .init(device)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let global_avg_pool =
            GlobalAvgPool::new(self.in_channels, in_channelster, self.batch_size, device);
        let conv1 = Conv2dConfig::new(
            [
                in_channelster * (2 + self.parallel_block_sizes.len()),
                out_channels,
            ],
            [1, 1],
        )
        .with_bias(false)
        .init(device);
        let bn1 = NormLayer::new(out_channels, self.batch_size, device);
        let relu = Relu::new();
        let dropout = DropoutConfig::new(0.5).init();

        let interpolation_strategy = Ignored(self.interpolation_strategy.clone());

        Ok(ASPPDeformable {
            aspp1,
            aspp_deforms,
            global_avg_pool,
            conv1,
            bn1,
            relu,
            dropout,
            interpolation_strategy,
        })
    }
}

/// ASPP module with Deformable Convolutions matching PyTorch structure.
///
/// # Shapes
///   - input: `[batch_size, in_channels, height, width]`
///   - output: `[batch_size, out_channels, height, width]`
#[derive(Module, Debug)]
pub struct ASPPDeformable<B: Backend> {
    aspp1: _ASPPModuleDeformable<B>,
    aspp_deforms: Vec<_ASPPModuleDeformable<B>>,
    /// Global average pooling sequential block (nn.Sequential in PyTorch)
    global_avg_pool: GlobalAvgPool<B>,
    conv1: Conv2d<B>,
    bn1: NormLayer<B>,
    relu: Relu,
    dropout: Dropout,
    /// Interpolation strategy for tensor resizing operations.
    interpolation_strategy: Ignored<InterpolationStrategy>,
}

impl<B: Backend> ASPPDeformable<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let x1 = self.aspp1.forward(x.clone());
        let x_aspp_deforms = self
            .aspp_deforms
            .iter()
            .map(|aspp| aspp.forward(x.clone()))
            .collect::<Vec<_>>();
        let x5 = self.global_avg_pool.forward(x);
        let [_, _, d3, d4] = x1.dims();
        let x5 = intelligent_interpolate(x5, [d3, d4], &self.interpolation_strategy.0);
        let mut x_ = vec![x1];
        x_.extend(x_aspp_deforms);
        x_.push(x5);
        let x = Tensor::cat(x_, 1);

        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.relu.forward(x);

        self.dropout.forward(x)
    }
}
