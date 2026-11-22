//! Overlapping patch embedding for `PVTv2`.

use burn::{
    nn::{
        Initializer, LayerNorm, LayerNormConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
};

use crate::backbones::usize_to_f64;

/// Overlapping patch embedding layer for `PVTv2`.
#[derive(Module, Debug)]
pub(crate) struct OverlapPatchEmbed {
    proj: Conv2d,
    norm: LayerNorm,
}

impl OverlapPatchEmbed {
    /// Initializes a `patch_size x patch_size` strided embedding (`padding = patch_size / 2`).
    pub(crate) fn new(
        patch_size: usize,
        stride: usize,
        channels: [usize; 2],
        epsilon: f64,
        device: &Device,
    ) -> Self {
        let [in_channels, embed_dim] = channels;
        let fan_out = usize_to_f64(patch_size * patch_size * embed_dim);
        let std = (2.0 / fan_out).sqrt();
        let pad = patch_size / 2;

        Self {
            proj: Conv2dConfig::new([in_channels, embed_dim], [patch_size, patch_size])
                .with_stride([stride, stride])
                .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
                .with_initializer(Initializer::Normal { mean: 0.0, std })
                .init(device),
            norm: LayerNormConfig::new(embed_dim)
                .with_epsilon(epsilon)
                .init(device),
        }
    }

    /// Embeds `[B, C, H, W]` into `(tokens [B, H' * W', embed_dim], H', W')`.
    pub(crate) fn forward(&self, x: Tensor<4>) -> (Tensor<3>, usize, usize) {
        let x = self.proj.forward(x);
        let [_, _, h, w] = x.dims();
        let x = self.norm.forward(x.flatten(2, 3).transpose());
        (x, h, w)
    }
}
