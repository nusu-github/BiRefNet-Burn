//! Transformer block and convolutional MLP for `PVTv2`.

use birefnet_extra_ops::{DropPath, DropPathConfig};
use burn::{
    nn::{
        Dropout, DropoutConfig, Gelu, Initializer, LayerNorm, LayerNormConfig, Linear,
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
};

use super::{
    attention::{Attention, AttentionConfig},
    trunc_normal_linear,
};
use crate::backbones::{scaled_hidden_dim, usize_to_f64};

/// Depth-wise convolution used in the `PVTv2` MLP block.
#[derive(Module, Debug)]
pub(crate) struct DWConv {
    dwconv: Conv2d,
}

impl DWConv {
    /// Initializes a 3x3 depth-wise convolution for `dim` channels.
    fn new(dim: usize, device: &Device) -> Self {
        // Depth-wise conv: fan_out = k * k * out_channels / groups = 3 * 3.
        let std = (2.0 / usize_to_f64(3 * 3)).sqrt();
        let conv = Conv2dConfig::new([dim, dim], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Same)
            .with_groups(dim)
            .with_bias(true)
            .with_initializer(Initializer::Normal { mean: 0.0, std })
            .init(device);
        Self { dwconv: conv }
    }

    /// Applies depth-wise spatial mixing to sequence tensor `[B, H * W, C]`.
    fn forward(&self, x: Tensor<3>, height: usize, width: usize) -> Tensor<3> {
        let [batch, _, channels] = x.dims();
        let spatial = x.transpose().reshape([batch, channels, height, width]);
        self.dwconv.forward(spatial).flatten(2, 3).transpose()
    }
}

/// Multi-layer perceptron block with embedded depth-wise convolution.
#[derive(Module, Debug)]
pub(crate) struct Mlp {
    fc1: Linear,
    dwconv: DWConv,
    act: Gelu,
    fc2: Linear,
    drop: Dropout,
}

impl Mlp {
    fn new(dim: usize, hidden_features: usize, drop: f64, device: &Device) -> Self {
        Self {
            fc1: trunc_normal_linear(dim, hidden_features, true, device),
            dwconv: DWConv::new(hidden_features, device),
            act: Gelu::new(),
            fc2: trunc_normal_linear(hidden_features, dim, true, device),
            drop: DropoutConfig::new(drop).init(),
        }
    }

    /// Applies `fc1 -> dwconv -> gelu -> drop -> fc2 -> drop`.
    fn forward(&self, x: Tensor<3>, h: usize, w: usize) -> Tensor<3> {
        let x = self.drop.forward(
            self.act
                .forward(self.dwconv.forward(self.fc1.forward(x), h, w)),
        );
        self.drop.forward(self.fc2.forward(x))
    }
}

/// Hyperparameters for [`Block`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct BlockConfig {
    pub(crate) attention: AttentionConfig,
    pub(crate) mlp_ratio: f64,
    pub(crate) drop: f64,
    pub(crate) drop_path: f64,
}

impl BlockConfig {
    pub(crate) fn init(&self, device: &Device) -> Block {
        let dim = self.attention.dim;
        let epsilon = self.attention.epsilon;
        Block {
            norm1: LayerNormConfig::new(dim).with_epsilon(epsilon).init(device),
            attn: self.attention.init(device),
            drop_path: DropPathConfig::new().with_drop_prob(self.drop_path).init(),
            norm2: LayerNormConfig::new(dim).with_epsilon(epsilon).init(device),
            mlp: Mlp::new(
                dim,
                scaled_hidden_dim(dim, self.mlp_ratio),
                self.drop,
                device,
            ),
        }
    }
}

/// Transformer block for `PVTv2`.
#[derive(Module, Debug)]
pub(crate) struct Block {
    norm1: LayerNorm,
    attn: Attention,
    drop_path: DropPath,
    norm2: LayerNorm,
    mlp: Mlp,
}

impl Block {
    /// Runs attention and MLP residual branches for spatial dimensions `(h, w)`.
    pub(crate) fn forward(&self, x: Tensor<3>, h: usize, w: usize) -> Tensor<3> {
        let x = x.clone()
            + self
                .drop_path
                .forward(self.attn.forward(self.norm1.forward(x), h, w));
        x.clone()
            + self
                .drop_path
                .forward(self.mlp.forward(self.norm2.forward(x), h, w))
    }
}
