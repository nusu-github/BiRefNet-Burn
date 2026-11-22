//! Atrous Spatial Pyramid Pooling (`ASPP`) and deformable `ASPP` modules for `BiRefNet`.

use burn::{
    nn::{
        Dropout, DropoutConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::*,
};

use super::{DeformableConv2d, decoder_blocks::NormLayer, utils::intelligent_interpolate};
use crate::{
    config::InterpolationStrategy, error::ModelError,
    models::modules::deform_conv::DeformableConv2dConfig,
};

/// Width of every ASPP branch (upstream `inter_channels = 256`).
const BRANCH_CHANNELS: usize = 256;
/// Dropout applied after the fused projection (upstream `nn.Dropout(0.5)`).
const FUSION_DROPOUT: f64 = 0.5;

#[derive(Config, Debug)]
struct AsppBranchConfig {
    in_channels: usize,
    planes: usize,
    kernel_size: usize,
    padding: usize,
    dilation: usize,
    #[config(default = 4)]
    batch_size: usize,
}

impl AsppBranchConfig {
    #[must_use]
    fn init(&self, device: &Device) -> AsppBranch {
        let atrous_conv = Conv2dConfig::new(
            [self.in_channels, self.planes],
            [self.kernel_size, self.kernel_size],
        )
        .with_stride([1, 1])
        .with_padding(PaddingConfig2d::Explicit(
            self.padding,
            self.padding,
            self.padding,
            self.padding,
        ))
        .with_dilation([self.dilation, self.dilation])
        .with_bias(false)
        .init(device);

        let bn = NormLayer::new(self.planes, self.batch_size, device);
        let relu = Relu::new();

        AsppBranch {
            atrous_conv,
            bn,
            relu,
        }
    }
}

#[derive(Module, Debug)]
struct AsppBranch {
    atrous_conv: Conv2d,
    bn: NormLayer,
    relu: Relu,
}

impl AsppBranch {
    #[must_use]
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.atrous_conv.forward(x);
        let x = self.bn.forward(x);
        self.relu.forward(x)
    }
}

/// Global average pooling branch matching the `ASPP` sequential block layout.
#[derive(Module, Debug)]
struct GlobalAvgPool {
    pool: AdaptiveAvgPool2d,
    conv: Conv2d,
    bn: NormLayer,
    relu: Relu,
}

impl GlobalAvgPool {
    #[must_use]
    fn new(in_channels: usize, out_channels: usize, batch_size: usize, device: &Device) -> Self {
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

    #[must_use]
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.pool.forward(x);
        let x = self.conv.forward(x);
        let x = self.bn.forward(x);
        self.relu.forward(x)
    }
}

/// Configuration for [`ASPP`].
#[derive(Config, Debug)]
pub struct ASPPConfig {
    /// Number of input channels.
    #[config(default = 64)]
    in_channels: usize,
    /// Number of output channels; defaults to `in_channels`.
    #[config(default = "None")]
    out_channels: Option<usize>,
    /// Output stride of the encoder (`8` or `16`).
    #[config(default = 16)]
    output_stride: usize,
    /// Batch size for conditional `BatchNorm` usage.
    #[config(default = 4)]
    batch_size: usize,
    /// Interpolation strategy for tensor resizing operations.
    interpolation_strategy: InterpolationStrategy,
}

impl ASPPConfig {
    /// Initializes a new [`ASPP`] module.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if `output_stride` is neither `8` nor `16`.
    pub fn init(&self, device: &Device) -> Result<ASPP, ModelError> {
        let out_channels = self.out_channels.unwrap_or(self.in_channels);

        let dilations = match self.output_stride {
            16 => [1, 6, 12, 18],
            8 => [1, 12, 24, 36],
            other => {
                return Err(ModelError::invalid_config(format!(
                    "unsupported ASPP output_stride {other}; expected 8 or 16"
                )));
            }
        };

        // Branch 1 is a 1x1 conv; branches 2-4 are dilated 3x3 convs padded to keep the size.
        let [aspp1, aspp2, aspp3, aspp4] = dilations.map(|dilation| {
            let (kernel_size, padding) = if dilation == dilations[0] {
                (1, 0)
            } else {
                (3, dilation)
            };
            AsppBranchConfig::new(
                self.in_channels,
                BRANCH_CHANNELS,
                kernel_size,
                padding,
                dilation,
            )
            .with_batch_size(self.batch_size)
            .init(device)
        });

        let global_avg_pool =
            GlobalAvgPool::new(self.in_channels, BRANCH_CHANNELS, self.batch_size, device);

        let conv1 = Conv2dConfig::new([BRANCH_CHANNELS * 5, out_channels], [1, 1])
            .with_bias(false)
            .init(device);
        let bn1 = NormLayer::new(out_channels, self.batch_size, device);
        let relu = Relu::new();
        let dropout = DropoutConfig::new(FUSION_DROPOUT).init();

        Ok(ASPP {
            aspp1,
            aspp2,
            aspp3,
            aspp4,
            global_avg_pool,
            conv1,
            bn1,
            relu,
            dropout,
            interpolation_strategy: self.interpolation_strategy,
        })
    }
}

/// Atrous Spatial Pyramid Pooling (`ASPP`) module.
///
/// # Shapes
/// - input: `[batch_size, in_channels, height, width]`
/// - output: `[batch_size, out_channels, height, width]`
#[derive(Module, Debug)]
#[expect(
    clippy::upper_case_acronyms,
    reason = "established deep-learning acronym matching BiRefNet public API"
)]
pub struct ASPP {
    aspp1: AsppBranch,
    aspp2: AsppBranch,
    aspp3: AsppBranch,
    aspp4: AsppBranch,
    global_avg_pool: GlobalAvgPool,
    conv1: Conv2d,
    bn1: NormLayer,
    relu: Relu,
    dropout: Dropout,
    #[module(skip)]
    interpolation_strategy: InterpolationStrategy,
}

impl ASPP {
    /// Computes multi-scale atrous spatial pyramid features.
    ///
    /// # Shapes
    /// - `x`: `[batch_size, in_channels, height, width]`
    /// - returns: `[batch_size, out_channels, height, width]`
    #[must_use]
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x1 = self.aspp1.forward(x.clone());
        let x2 = self.aspp2.forward(x.clone());
        let x3 = self.aspp3.forward(x.clone());
        let x4 = self.aspp4.forward(x.clone());

        let x5 = self.global_avg_pool.forward(x);
        let [_, _, d3, d4] = x1.dims();
        let x5 = intelligent_interpolate(x5, [d3, d4], self.interpolation_strategy);

        let x = Tensor::cat(vec![x1, x2, x3, x4, x5], 1);
        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.relu.forward(x);

        self.dropout.forward(x)
    }
}

#[derive(Config, Debug)]
struct AsppDeformableBranchConfig {
    in_channels: usize,
    planes: usize,
    kernel_size: usize,
    padding: usize,
    #[config(default = 4)]
    batch_size: usize,
}

impl AsppDeformableBranchConfig {
    #[must_use]
    fn init(&self, device: &Device) -> AsppDeformableBranch {
        let atrous_conv = DeformableConv2dConfig::new(self.in_channels, self.planes)
            .with_kernel_size(self.kernel_size)
            .with_stride(1)
            .with_padding(self.padding)
            .with_bias(false)
            .init(device);
        let bn = NormLayer::new(self.planes, self.batch_size, device);
        let relu = Relu::new();
        AsppDeformableBranch {
            atrous_conv,
            bn,
            relu,
        }
    }
}

#[derive(Module, Debug)]
struct AsppDeformableBranch {
    atrous_conv: DeformableConv2d,
    bn: NormLayer,
    relu: Relu,
}

impl AsppDeformableBranch {
    #[must_use]
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.atrous_conv.forward(x);
        let x = self.bn.forward(x);
        self.relu.forward(x)
    }
}

/// Configuration for [`ASPPDeformable`].
#[derive(Config, Debug)]
pub struct ASPPDeformableConfig {
    /// Number of input channels.
    #[config(default = 64)]
    in_channels: usize,
    /// Number of output channels; defaults to `in_channels`.
    #[config(default = "None")]
    out_channels: Option<usize>,
    /// Kernel sizes for the parallel deformable convolution branches.
    #[config(default = "[1, 3, 7]")]
    parallel_block_sizes: [usize; 3],
    /// Batch size for conditional `BatchNorm` usage.
    #[config(default = 4)]
    batch_size: usize,
    /// Interpolation strategy for tensor resizing operations.
    interpolation_strategy: InterpolationStrategy,
}

impl ASPPDeformableConfig {
    /// Initializes a new [`ASPPDeformable`] module.
    #[must_use]
    pub fn init(&self, device: &Device) -> ASPPDeformable {
        let out_channels = self.out_channels.unwrap_or(self.in_channels);

        let aspp1 = AsppDeformableBranchConfig::new(self.in_channels, BRANCH_CHANNELS, 1, 0)
            .with_batch_size(self.batch_size)
            .init(device);
        let aspp_deforms = self
            .parallel_block_sizes
            .iter()
            .map(|&conv_size| {
                AsppDeformableBranchConfig::new(
                    self.in_channels,
                    BRANCH_CHANNELS,
                    conv_size,
                    conv_size / 2,
                )
                .with_batch_size(self.batch_size)
                .init(device)
            })
            .collect();

        let global_avg_pool =
            GlobalAvgPool::new(self.in_channels, BRANCH_CHANNELS, self.batch_size, device);
        let conv1 = Conv2dConfig::new(
            [
                BRANCH_CHANNELS * (2 + self.parallel_block_sizes.len()),
                out_channels,
            ],
            [1, 1],
        )
        .with_bias(false)
        .init(device);
        let bn1 = NormLayer::new(out_channels, self.batch_size, device);
        let relu = Relu::new();
        let dropout = DropoutConfig::new(FUSION_DROPOUT).init();

        ASPPDeformable {
            aspp1,
            aspp_deforms,
            global_avg_pool,
            conv1,
            bn1,
            relu,
            dropout,
            interpolation_strategy: self.interpolation_strategy,
        }
    }
}

/// Atrous Spatial Pyramid Pooling (`ASPP`) module with deformable convolutions.
///
/// # Shapes
/// - input: `[batch_size, in_channels, height, width]`
/// - output: `[batch_size, out_channels, height, width]`
#[derive(Module, Debug)]
pub struct ASPPDeformable {
    aspp1: AsppDeformableBranch,
    aspp_deforms: Vec<AsppDeformableBranch>,
    global_avg_pool: GlobalAvgPool,
    conv1: Conv2d,
    bn1: NormLayer,
    relu: Relu,
    dropout: Dropout,
    #[module(skip)]
    interpolation_strategy: InterpolationStrategy,
}

impl ASPPDeformable {
    /// Applies multi-scale deformable `ASPP` branches and fuses them via `1x1` convolution.
    #[must_use]
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [_, _, height, width] = x.dims();
        let mut branches = Vec::with_capacity(2 + self.aspp_deforms.len());
        branches.push(self.aspp1.forward(x.clone()));
        branches.extend(self.aspp_deforms.iter().map(|aspp| aspp.forward(x.clone())));
        let pooled = self.global_avg_pool.forward(x);
        branches.push(intelligent_interpolate(
            pooled,
            [height, width],
            self.interpolation_strategy,
        ));
        let x = Tensor::cat(branches, 1);

        let x = self.conv1.forward(x);
        let x = self.bn1.forward(x);
        let x = self.relu.forward(x);

        self.dropout.forward(x)
    }
}
