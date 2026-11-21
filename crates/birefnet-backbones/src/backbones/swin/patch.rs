//! Patch embedding and patch merging layers for Swin Transformer.

use burn::{
    nn::{
        LayerNorm, LayerNormConfig, Linear, LinearConfig,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
};

use crate::backbones::usize_to_i64;

/// Configuration for [`PatchMerging`].
#[derive(Config, Debug, PartialEq, Eq)]
pub struct PatchMergingConfig {
    dim: usize,
}

impl PatchMergingConfig {
    /// Initializes a [`PatchMerging`] layer.
    #[must_use]
    pub fn init(&self, device: &Device) -> PatchMerging {
        PatchMerging {
            norm: LayerNormConfig::new(4 * self.dim).init(device),
            reduction: LinearConfig::new(4 * self.dim, 2 * self.dim)
                .with_bias(false)
                .init(device),
        }
    }
}

/// Downsampling layer that concatenates `2x2` neighboring patch tokens and projects `4C -> 2C`.
#[derive(Module, Debug)]
pub struct PatchMerging {
    norm: LayerNorm,
    reduction: Linear,
}

impl PatchMerging {
    /// Downsamples `[B, H * W, C]` to `[B, ceil(H / 2) * ceil(W / 2), 2C]`.
    #[must_use]
    pub fn forward(&self, x: Tensor<3>, height: usize, width: usize) -> Tensor<3> {
        let device = x.device();
        let [batch, _tokens, channels] = x.dims();
        let spatial = x.reshape([batch, height, width, channels]);

        let extra_rows = height % 2;
        let extra_cols = width % 2;
        let pad_h = height + extra_rows;
        let pad_w = width + extra_cols;

        let padded = if extra_rows != 0 || extra_cols != 0 {
            spatial
                .permute([0, 3, 1, 2])
                .pad((0, extra_cols, 0, extra_rows), 0.0)
                .permute([0, 2, 3, 1])
        } else {
            spatial
        };

        let top_idx = Tensor::arange_step(0..usize_to_i64(pad_h), 2, &device);
        let bottom_idx = Tensor::arange_step(1..usize_to_i64(pad_h), 2, &device);
        let left_idx = Tensor::arange_step(0..usize_to_i64(pad_w), 2, &device);
        let right_idx = Tensor::arange_step(1..usize_to_i64(pad_w), 2, &device);

        let top_left = padded
            .clone()
            .select(1, top_idx.clone())
            .select(2, left_idx.clone());
        let bottom_left = padded
            .clone()
            .select(1, bottom_idx.clone())
            .select(2, left_idx);
        let top_right = padded
            .clone()
            .select(1, top_idx)
            .select(2, right_idx.clone());
        let bottom_right = padded.select(1, bottom_idx).select(2, right_idx);

        let merged = Tensor::cat(vec![top_left, bottom_left, top_right, bottom_right], 3);
        let flat = merged.reshape([batch, height.div_ceil(2) * width.div_ceil(2), 4 * channels]);
        self.reduction.forward(self.norm.forward(flat))
    }
}

/// Configuration for [`PatchEmbed`].
#[derive(Config, Debug, PartialEq, Eq)]
pub struct PatchEmbedConfig {
    #[config(default = 4)]
    patch_size: usize,
    #[config(default = 3)]
    in_channels: usize,
    #[config(default = 96)]
    embed_dim: usize,
    #[config(default = false)]
    norm_layer: bool,
}

impl PatchEmbedConfig {
    /// Initializes a [`PatchEmbed`] layer.
    #[must_use]
    pub fn init(&self, device: &Device) -> PatchEmbed {
        let proj = Conv2dConfig::new(
            [self.in_channels, self.embed_dim],
            [self.patch_size, self.patch_size],
        )
        .with_stride([self.patch_size, self.patch_size])
        .init(device);
        let norm = self
            .norm_layer
            .then(|| LayerNormConfig::new(self.embed_dim).init(device));

        PatchEmbed {
            embed_dim: self.embed_dim,
            patch_size: self.patch_size,
            proj,
            norm,
        }
    }
}

/// Image-to-patch convolutional embedding layer.
#[derive(Module, Debug)]
pub struct PatchEmbed {
    embed_dim: usize,
    patch_size: usize,
    proj: Conv2d,
    norm: Option<LayerNorm>,
}

impl PatchEmbed {
    /// Projects `[B, in_channels, H, W]` into `[B, embed_dim, ceil(H / P), ceil(W / P)]`.
    #[must_use]
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let [_, _, h, w] = x.dims();
        let pad_r = (self.patch_size - (w % self.patch_size)) % self.patch_size;
        let pad_b = (self.patch_size - (h % self.patch_size)) % self.patch_size;
        let x = if pad_r > 0 || pad_b > 0 {
            x.pad((0, pad_r, 0, pad_b), 0.0)
        } else {
            x
        };
        let x = self.proj.forward(x);

        match &self.norm {
            Some(norm) => {
                let [batch_size, _, wh, ww] = x.dims();
                let x: Tensor<3> = x.flatten(2, 3).swap_dims(1, 2);
                let x = norm.forward(x);
                x.swap_dims(1, 2)
                    .reshape([batch_size, self.embed_dim, wh, ww])
            }
            None => x,
        }
    }
}
