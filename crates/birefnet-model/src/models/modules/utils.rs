//! Internal building helpers for normalization, activation, and spatial resizing.

use burn::{
    nn::{BatchNorm, BatchNormConfig, LayerNorm, LayerNormConfig},
    prelude::*,
    tensor::{
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};

use crate::config::InterpolationStrategy;

/// Supported normalization layer kinds for refinement stem layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NormKind {
    /// Two-dimensional batch normalization over `[B, C, H, W]`.
    BatchNorm,
    /// Layer normalization over channel vectors.
    #[expect(dead_code, reason = "supported stem normalization variant")]
    LayerNorm,
}

/// Wrapper enum storing layout-permutation and normalization layers.
#[derive(Module, Debug)]
pub enum NormLayerEnum {
    /// Permutes `[B, H, W, C]` inputs to `[B, C, H, W]`.
    ChannelsFirst(ChannelsFirst),
    /// Permutes `[B, C, H, W]` inputs to `[B, H, W, C]`.
    ChannelsLast(ChannelsLast),
    /// Batch normalization over channel dimension `C`.
    BatchNorm2d(BatchNorm),
    /// Layer normalization over the trailing dimension.
    LayerNorm(LayerNorm),
}

impl NormLayerEnum {
    /// Applies this layer to an NCHW tensor (`LayerNorm` normalizes over channels).
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        match self {
            Self::ChannelsFirst(layer) => layer.forward(x),
            Self::ChannelsLast(layer) => layer.forward(x),
            Self::BatchNorm2d(layer) => layer.forward(x),
            Self::LayerNorm(layer) => layer.forward(x.permute([0, 2, 3, 1])).permute([0, 3, 1, 2]),
        }
    }
}

/// Converts NHWC tensors into NCHW layout.
#[derive(Module, Debug, Default)]
pub struct ChannelsFirst;

impl ChannelsFirst {
    /// Creates a new [`ChannelsFirst`] permutation layer.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Permutes `[B, H, W, C]` -> `[B, C, H, W]`.
    #[must_use]
    #[expect(
        clippy::unused_self,
        reason = "matches Burn Module::forward(&self, ...) signature"
    )]
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        input.permute([0, 3, 1, 2])
    }
}

/// Converts NCHW tensors into NHWC layout.
#[derive(Module, Debug, Default)]
pub struct ChannelsLast;

impl ChannelsLast {
    /// Creates a new [`ChannelsLast`] permutation layer.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Permutes `[B, C, H, W]` -> `[B, H, W, C]`.
    #[must_use]
    #[expect(
        clippy::unused_self,
        reason = "matches Burn Module::forward(&self, ...) signature"
    )]
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        input.permute([0, 2, 3, 1])
    }
}

/// Builds a normalization layer sequence with optional channel-layout adapters.
#[must_use]
pub fn build_norm_layer(
    dim: usize,
    norm_kind: NormKind,
    in_format: bool,
    out_format: bool,
    eps: f64,
    device: &Device,
) -> Vec<NormLayerEnum> {
    let mut layer = Vec::with_capacity(3);
    if in_format {
        layer.push(NormLayerEnum::ChannelsFirst(ChannelsFirst::new()));
    }
    match norm_kind {
        NormKind::BatchNorm => {
            layer.push(NormLayerEnum::BatchNorm2d(
                BatchNormConfig::new(dim).init(device),
            ));
        }
        NormKind::LayerNorm => {
            layer.push(NormLayerEnum::LayerNorm(
                LayerNormConfig::new(dim).with_epsilon(eps).init(device),
            ));
        }
    }
    if out_format {
        layer.push(NormLayerEnum::ChannelsLast(ChannelsLast::new()));
    }
    layer
}

/// Activation kind stored in refinement stems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActLayerEnum {
    /// Gaussian Error Linear Unit (`GeLU`).
    Gelu,
}

/// Resizes `tensor` to `[height, width]` using the configured [`InterpolationStrategy`].
#[must_use]
pub fn intelligent_interpolate(
    tensor: Tensor<4>,
    size: [usize; 2],
    strategy: InterpolationStrategy,
) -> Tensor<4> {
    let interpolate_mode = match strategy {
        InterpolationStrategy::Bilinear => InterpolateMode::Bilinear,
        InterpolationStrategy::Nearest => InterpolateMode::Nearest,
    };

    interpolate(
        tensor,
        InterpolateOptions::new(interpolate_mode).with_output_size(size),
    )
}
