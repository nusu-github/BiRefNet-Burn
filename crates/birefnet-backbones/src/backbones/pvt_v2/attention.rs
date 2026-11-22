//! Spatial-reduction multi-head attention for `PVTv2`.

use burn::{
    nn::{
        Dropout, DropoutConfig, Initializer, LayerNorm, LayerNormConfig, Linear,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::activation::softmax,
};

use super::trunc_normal_linear;
use crate::backbones::usize_to_f64;

/// Hyperparameters for [`Attention`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct AttentionConfig {
    pub(crate) dim: usize,
    pub(crate) num_heads: usize,
    pub(crate) qkv_bias: bool,
    pub(crate) qk_scale: Option<f64>,
    pub(crate) attn_drop: f64,
    pub(crate) proj_drop: f64,
    pub(crate) sr_ratio: usize,
    pub(crate) epsilon: f64,
}

impl AttentionConfig {
    pub(crate) fn init(&self, device: &Device) -> Attention {
        let Self {
            dim,
            num_heads,
            qkv_bias,
            qk_scale,
            attn_drop,
            proj_drop,
            sr_ratio,
            epsilon,
        } = *self;
        let head_dim = dim / num_heads;
        // He-normal init with fan_out = k * k * out_channels, as in the upstream `_init_weights`.
        let sr = (sr_ratio > 1).then(|| {
            let fan_out = usize_to_f64(sr_ratio * sr_ratio * dim);
            let std = (2.0 / fan_out).sqrt();
            Conv2dConfig::new([dim, dim], [sr_ratio, sr_ratio])
                .with_stride([sr_ratio, sr_ratio])
                .with_initializer(Initializer::Normal { mean: 0.0, std })
                .init(device)
        });
        let norm =
            (sr_ratio > 1).then(|| LayerNormConfig::new(dim).with_epsilon(epsilon).init(device));

        Attention {
            dim,
            num_heads,
            scale: qk_scale.unwrap_or_else(|| usize_to_f64(head_dim).powf(-0.5)),
            q: trunc_normal_linear(dim, dim, qkv_bias, device),
            kv: trunc_normal_linear(dim, dim * 2, qkv_bias, device),
            attn_drop: DropoutConfig::new(attn_drop).init(),
            proj: trunc_normal_linear(dim, dim, true, device),
            proj_drop: DropoutConfig::new(proj_drop).init(),
            sr_ratio,
            sr,
            norm,
        }
    }
}

/// Spatially-reductive multi-head attention module.
#[derive(Module, Debug)]
pub(crate) struct Attention {
    dim: usize,
    num_heads: usize,
    scale: f64,
    q: Linear,
    kv: Linear,
    attn_drop: Dropout,
    proj: Linear,
    proj_drop: Dropout,
    sr_ratio: usize,
    sr: Option<Conv2d>,
    norm: Option<LayerNorm>,
}

impl Attention {
    /// Computes spatially-reduced multi-head self-attention on `[B, H * W, C]`.
    pub(crate) fn forward(&self, x: Tensor<3>, height: usize, width: usize) -> Tensor<3> {
        let [batch, seq_len, channels] = x.dims();
        let head_dim = channels / self.num_heads;
        let query = self
            .q
            .forward(x.clone())
            .reshape([batch, seq_len, self.num_heads, head_dim])
            .permute([0, 2, 1, 3]);

        let (kv_input, kv_len) = match (&self.sr, &self.norm) {
            (Some(sr), Some(norm)) => {
                let reduced = sr.forward(
                    x.permute([0, 2, 1])
                        .reshape([batch, channels, height, width]),
                );
                let [_, _, new_h, new_w] = reduced.dims();
                let tokens = norm.forward(reduced.flatten(2, 3).permute([0, 2, 1]));
                (tokens, new_h * new_w)
            }
            _ => (x, seq_len),
        };

        // [B, N, 2, heads, head_dim] -> [2, B, heads, N, head_dim], then split K / V.
        let kv = self
            .kv
            .forward(kv_input)
            .reshape([batch, kv_len, 2, self.num_heads, head_dim])
            .permute([2, 0, 3, 1, 4]);
        let key = kv.clone().narrow(0, 0, 1).squeeze_dim::<4>(0);
        let value = kv.narrow(0, 1, 1).squeeze_dim::<4>(0);

        let attn = self
            .attn_drop
            .forward(softmax(query.matmul(key.transpose()) * self.scale, 3));
        let attended = attn
            .matmul(value)
            .permute([0, 2, 1, 3])
            .reshape([batch, seq_len, channels]);
        self.proj_drop.forward(self.proj.forward(attended))
    }
}
