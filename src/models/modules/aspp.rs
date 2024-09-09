use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Relu,
    },
    prelude::*,
    tensor::{
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};

use super::DeformableConv2d;
use crate::models::DeformableConv2dConfig;

#[derive(Config)]
pub struct _ASPPModuleConfig {
    in_channels: usize,
    planes: usize,
    kernel_size: usize,
    padding: usize,
    dilation: usize,
}

impl _ASPPModuleConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> _ASPPModule<B> {
        todo!()
    }
}

#[derive(Module, Debug)]
pub struct _ASPPModule<B: Backend> {
    atrous_conv: Conv2d<B>,
    bn: Option<BatchNorm<B, 2>>,
    relu: Relu,
}

impl<B: Backend> _ASPPModule<B> {
    pub fn forward(&self, x: Tensor<B, 4, Float>) -> Tensor<B, 4, Float> {
        todo!()
    }
}

#[derive(Config)]
pub struct ASPPConfig {
    #[config(default = "64")]
    in_channels: usize,
    #[config(default = "None")]
    out_channels: Option<usize>,
    #[config(default = "16")]
    output_stride: usize,
}

impl ASPPConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> ASPP<B> {
        todo!()
    }
}

#[derive(Module, Debug)]
pub struct ASPP<B: Backend> {
    aspp1: _ASPPModule<B>,
    aspp2: _ASPPModule<B>,
    aspp3: _ASPPModule<B>,
    aspp4: _ASPPModule<B>,
    // nn.Sequential
    global_avg_pool_1: AdaptiveAvgPool2d,
    global_avg_pool_2: Conv2d<B>,
    global_avg_pool_3: Option<BatchNorm<B, 2>>,
    global_avg_pool_4: Relu,
    conv1: Conv2d<B>,
    bn1: Option<BatchNorm<B, 2>>,
    relu: Relu,
    dropout: Dropout,
}

impl<B: Backend> ASPP<B> {
    pub fn forward(&self, x: Tensor<B, 4, Float>) -> Tensor<B, 4, Float> {
        todo!()
    }
}

#[derive(Config)]
pub struct _ASPPModuleDeformableConfig {
    in_channels: usize,
    planes: usize,
    kernel_size: usize,
    padding: usize,
}

impl _ASPPModuleDeformableConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> _ASPPModuleDeformable<B> {
        let atrous_conv = DeformableConv2dConfig::new(
            self.in_channels,
            self.planes,
            self.kernel_size,
            1,
            self.padding,
            false,
        )
        .init(device);
        let bn = BatchNormConfig::new(self.planes).init(device);
        let relu = Relu::new();
        _ASPPModuleDeformable {
            atrous_conv,
            bn: Some(bn),
            relu,
        }
    }
}

#[derive(Module, Debug)]
pub struct _ASPPModuleDeformable<B: Backend> {
    atrous_conv: DeformableConv2d<B>,
    bn: Option<BatchNorm<B, 2>>,
    relu: Relu,
}

impl<B: Backend> _ASPPModuleDeformable<B> {
    pub fn forward(&self, x: Tensor<B, 4, Float>) -> Tensor<B, 4, Float> {
        let x = self.atrous_conv.forward(x);
        let x = if let Some(bn) = self.bn.as_ref() {
            bn.forward(x)
        } else {
            x
        };
        let x = self.relu.forward(x);
        x
    }
}

#[derive(Config)]
pub struct ASPPDeformableConfig {
    #[config(default = "64")]
    in_channels: usize,
    #[config(default = "None")]
    out_channels: Option<usize>,
    #[config(default = "[1, 3, 7]")]
    parallel_block_sizes: [usize; 3],
}

impl ASPPDeformableConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> ASPPDeformable<B> {
        let out_channels = self.out_channels.unwrap_or(self.in_channels);
        let in_channelster = 256 / 1;

        let aspp1 =
            _ASPPModuleDeformableConfig::new(self.in_channels, in_channelster, 1, 0).init(device);
        let aspp_deforms = self
            .parallel_block_sizes
            .iter()
            .map(|conv_size| {
                _ASPPModuleDeformableConfig::new(
                    self.in_channels,
                    in_channelster,
                    *conv_size,
                    conv_size / 2,
                )
                .init(device)
            })
            .collect::<Vec<_>>();

        let global_avg_pool_1 = AdaptiveAvgPool2dConfig::new([1, 1]).init();
        let global_avg_pool_2 = Conv2dConfig::new([self.in_channels, in_channelster], [1, 1])
            .with_stride([1, 1])
            .with_bias(false)
            .init(device);
        let global_avg_pool_3 = BatchNormConfig::new(in_channelster).init(device);
        let global_avg_pool_4 = Relu::new();
        let conv1 = Conv2dConfig::new(
            [
                self.in_channels * (2 + self.parallel_block_sizes.len()),
                out_channels,
            ],
            [1, 1],
        )
        .with_bias(false)
        .init(device);
        let bn1 = BatchNormConfig::new(out_channels).init(device);
        let relu = Relu::new();
        let dropout = DropoutConfig::new(0.5).init();

        ASPPDeformable {
            aspp1,
            aspp_deforms,
            global_avg_pool_1,
            global_avg_pool_2,
            global_avg_pool_3: Some(global_avg_pool_3),
            global_avg_pool_4,
            conv1,
            bn1: Some(bn1),
            relu,
            dropout,
        }
    }
}

#[derive(Module, Debug)]
pub struct ASPPDeformable<B: Backend> {
    aspp1: _ASPPModuleDeformable<B>,
    aspp_deforms: Vec<_ASPPModuleDeformable<B>>,
    global_avg_pool_1: AdaptiveAvgPool2d,
    global_avg_pool_2: Conv2d<B>,
    global_avg_pool_3: Option<BatchNorm<B, 2>>,
    global_avg_pool_4: Relu,
    conv1: Conv2d<B>,
    bn1: Option<BatchNorm<B, 2>>,
    relu: Relu,
    dropout: Dropout,
}

impl<B: Backend> ASPPDeformable<B> {
    pub fn forward(&self, x: Tensor<B, 4, Float>) -> Tensor<B, 4, Float> {
        let x1 = self.aspp1.forward(x.clone());
        let x_aspp_deforms = self
            .aspp_deforms
            .iter()
            .map(|aspp| aspp.forward(x.clone()))
            .collect::<Vec<_>>();
        let x5 = self.global_avg_pool_1.forward(x.clone());
        let x5 = self.global_avg_pool_2.forward(x5);
        let x5 = if let Some(bn) = &self.global_avg_pool_3 {
            bn.forward(x5)
        } else {
            x5
        };
        let x5 = self.global_avg_pool_4.forward(x5);
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
        let x = if let Some(bn) = &self.bn1 {
            bn.forward(x)
        } else {
            x
        };
        let x = self.relu.forward(x);
        let x = self.dropout.forward(x);
        x
    }
}
