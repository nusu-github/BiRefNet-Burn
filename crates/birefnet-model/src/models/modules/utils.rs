use burn::{
    nn::{BatchNorm, BatchNormConfig, LayerNorm, LayerNormConfig},
    prelude::*,
    tensor::{
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};

use crate::{
    config::InterpolationStrategy,
    error::{BiRefNetError, BiRefNetResult},
};

#[derive(Module, Debug)]
pub enum NormLayerEnum<B: Backend> {
    ChannelsFirst(ChannelsFirst),
    ChannelsLast(ChannelsLast),
    BatchNorm2d(BatchNorm<B, 2>),
    BasicDecBlk(LayerNorm<B>),
}

#[derive(Module, Debug, Clone)]
pub struct ChannelsFirst;

impl Default for ChannelsFirst {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelsFirst {
    pub const fn new() -> Self {
        Self {}
    }

    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        input.permute([0, 3, 1, 2])
    }
}

#[derive(Module, Debug, Clone)]
pub struct ChannelsLast;

impl Default for ChannelsLast {
    fn default() -> Self {
        Self::new()
    }
}

impl ChannelsLast {
    pub const fn new() -> Self {
        Self {}
    }

    pub fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        input.permute([0, 2, 3, 1])
    }
}

pub fn build_norm_layer<B: Backend>(
    dim: usize,
    norm_layer: &str,
    in_format: bool,
    out_format: bool,
    eps: f64,
    device: &Device<B>,
) -> BiRefNetResult<Vec<NormLayerEnum<B>>> {
    match norm_layer {
        "BN" => {
            let mut layer = Vec::with_capacity(3);
            if in_format {
                layer.push(NormLayerEnum::ChannelsFirst(ChannelsFirst::new()));
            }
            layer.push(NormLayerEnum::BatchNorm2d(
                BatchNormConfig::new(dim).init(device),
            ));
            if out_format {
                layer.push(NormLayerEnum::ChannelsLast(ChannelsLast::new()));
            }
            Ok(layer)
        }
        "LN" => {
            let mut layer = Vec::with_capacity(3);
            if in_format {
                layer.push(NormLayerEnum::ChannelsFirst(ChannelsFirst::new()));
            }
            layer.push(NormLayerEnum::BasicDecBlk(
                LayerNormConfig::new(dim).with_epsilon(eps).init(device),
            ));
            if out_format {
                layer.push(NormLayerEnum::ChannelsLast(ChannelsLast::new()));
            }
            Ok(layer)
        }
        _ => {
            Err(BiRefNetError::UnsupportedSqueezeBlock {
                block_type: format!("Unsupported norm layer: {norm_layer}"),
            })
        }
    }
}

#[derive(Debug, Clone)]
pub enum ActLayerEnum {
    ReLU,
    SiLU,
    GELU,
}

/// Intelligent interpolation helper that automatically chooses the appropriate interpolation mode
/// based on the interpolation strategy and whether the model is in training mode.
///
/// # Arguments
/// * `tensor` - Input tensor to interpolate
/// * `size` - Target size [height, width]
/// * `strategy` - Interpolation strategy configuration
/// * `is_training` - Whether the model is currently in training mode
///
/// # Returns
/// Interpolated tensor with the specified size
///
/// # Notes
/// - During training (backward pass), Burn only supports Nearest interpolation
/// - During inference, both Bilinear and Nearest are supported
/// - The 'Auto' strategy automatically selects the appropriate mode
pub fn intelligent_interpolate<B: Backend>(
    tensor: Tensor<B, 4>,
    size: [usize; 2],
    strategy: &InterpolationStrategy,
) -> Tensor<B, 4> {
    let interpolate_mode = match strategy {
        InterpolationStrategy::Bilinear => InterpolateMode::Bilinear,
        InterpolationStrategy::Nearest => InterpolateMode::Nearest,
    };

    interpolate(tensor, size, InterpolateOptions::new(interpolate_mode))
}

/// Convenience function for interpolating tensors to match another tensor's spatial dimensions.
///
/// # Arguments
/// * `tensor` - Input tensor to interpolate
/// * `target` - Target tensor whose spatial dimensions to match
/// * `strategy` - Interpolation strategy configuration
/// * `is_training` - Whether the model is currently in training mode
///
/// # Returns
/// Interpolated tensor with spatial dimensions matching the target tensor
pub fn intelligent_interpolate_like<B: Backend>(
    tensor: Tensor<B, 4>,
    target: &Tensor<B, 4>,
    strategy: &InterpolationStrategy,
) -> Tensor<B, 4> {
    let [_, _, h, w] = target.dims();
    intelligent_interpolate(tensor, [h, w], strategy)
}
