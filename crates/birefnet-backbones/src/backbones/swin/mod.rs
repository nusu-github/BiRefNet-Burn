//! Swin Transformer v1 backbone implementation.

mod attention;
mod block;
mod layer;
mod patch;

use birefnet_extra_ops::trunc_normal;
use burn::{
    module::Param,
    nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig},
    prelude::*,
    tensor::{
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};

use self::{
    layer::{BasicLayer, BasicLayerConfig},
    patch::{PatchEmbed, PatchEmbedConfig},
};
use crate::backbones::usize_to_f64;
use crate::error::BackboneError;

/// Configuration for [`SwinTransformer`].
#[derive(Config, Debug, PartialEq)]
pub struct SwinTransformerConfig {
    #[config(default = 224)]
    pretrain_img_size: usize,
    #[config(default = 4)]
    patch_size: usize,
    #[config(default = 3)]
    in_channels: usize,
    #[config(default = 96)]
    embed_dim: usize,
    #[config(default = "[2, 2, 6, 2]")]
    depths: [usize; 4],
    #[config(default = "[3, 6, 12, 24]")]
    num_heads: [usize; 4],
    #[config(default = 7)]
    window_size: usize,
    #[config(default = 4.0)]
    mlp_ratio: f64,
    #[config(default = true)]
    qkv_bias: bool,
    #[config(default = "None")]
    qk_scale: Option<f64>,
    #[config(default = 0.0)]
    drop_rate: f64,
    #[config(default = 0.0)]
    attn_drop_rate: f64,
    #[config(default = 0.2)]
    drop_path_rate: f64,
    #[config(default = false)]
    ape: bool,
    #[config(default = true)]
    patch_norm: bool,
    #[config(default = "[0, 1, 2, 3]")]
    out_indices: [usize; 4],
}

impl SwinTransformerConfig {
    /// Swin-T configuration (`embed_dim = 96`, `depths = [2, 2, 6, 2]`, `window_size = 7`).
    #[must_use]
    pub fn swin_t() -> Self {
        Self::new()
    }

    /// Swin-S configuration (`embed_dim = 96`, `depths = [2, 2, 18, 2]`, `window_size = 7`).
    #[must_use]
    pub fn swin_s() -> Self {
        Self::new()
            .with_depths([2, 2, 18, 2])
            .with_num_heads([3, 6, 12, 24])
            .with_window_size(7)
    }

    /// Swin-B configuration (`embed_dim = 128`, `depths = [2, 2, 18, 2]`, `window_size = 12`).
    #[must_use]
    pub fn swin_b() -> Self {
        Self::new()
            .with_embed_dim(128)
            .with_depths([2, 2, 18, 2])
            .with_num_heads([4, 8, 16, 32])
            .with_window_size(12)
    }

    /// Swin-L configuration (`embed_dim = 192`, `depths = [2, 2, 18, 2]`, `window_size = 12`).
    #[must_use]
    pub fn swin_l() -> Self {
        Self::new()
            .with_embed_dim(192)
            .with_depths([2, 2, 18, 2])
            .with_num_heads([6, 12, 24, 48])
            .with_window_size(12)
    }

    /// Validates configuration invariants and initializes a [`SwinTransformer`].
    ///
    /// # Errors
    ///
    /// Returns [`BackboneError`] if channel dimensions, head counts, patch/window sizes, or
    /// output stage indices are invalid.
    pub fn init(&self, device: &Device) -> Result<SwinTransformer, BackboneError> {
        let num_layers = self.depths.len();
        if self.patch_size == 0
            || self.window_size == 0
            || self.embed_dim == 0
            || self.in_channels == 0
        {
            return Err(BackboneError::invalid_config(
                "patch_size, window_size, embed_dim, and in_channels must be positive",
            ));
        }

        let num_features = [0, 1, 2, 3].map(|i| self.embed_dim << i);
        for i in 0..num_layers {
            let heads = self.num_heads[i];
            if heads == 0 || !num_features[i].is_multiple_of(heads) {
                return Err(BackboneError::invalid_config(format!(
                    "stage {i} dim ({}) must be divisible by non-zero num_heads ({heads})",
                    num_features[i]
                )));
            }
        }
        if let Some(&bad_idx) = self.out_indices.iter().find(|&&idx| idx >= num_layers) {
            return Err(BackboneError::invalid_config(format!(
                "out_indices element {bad_idx} is out of bounds for {num_layers} stages"
            )));
        }

        let patch_embed = PatchEmbedConfig::new()
            .with_patch_size(self.patch_size)
            .with_in_channels(self.in_channels)
            .with_embed_dim(self.embed_dim)
            .with_norm_layer(self.patch_norm)
            .init(device);

        let absolute_pos_embed = self.ape.then(|| {
            let ph = self.pretrain_img_size / self.patch_size;
            let pw = self.pretrain_img_size / self.patch_size;
            Param::from_tensor(trunc_normal(
                [1, self.embed_dim, ph.max(1), pw.max(1)],
                0.0,
                0.02,
                -2.0,
                2.0,
                device,
            ))
        });

        let dpr = linspace(0.0, self.drop_path_rate, self.depths.iter().sum());

        let mut layers = Vec::with_capacity(num_layers);
        let mut cursor = 0_usize;
        for (i_layer, &depth) in self.depths.iter().enumerate() {
            let next_cursor = cursor + depth;
            layers.push(
                BasicLayerConfig::new(num_features[i_layer], depth, self.num_heads[i_layer])
                    .with_window_size(self.window_size)
                    .with_mlp_ratio(self.mlp_ratio)
                    .with_qkv_bias(self.qkv_bias)
                    .with_qk_scale(self.qk_scale)
                    .with_drop(self.drop_rate)
                    .with_attn_drop(self.attn_drop_rate)
                    .with_drop_path(dpr[cursor..next_cursor].to_vec())
                    .with_downsample(i_layer + 1 < num_layers)
                    .init(device),
            );
            cursor = next_cursor;
        }

        let norm_layers = self
            .out_indices
            .iter()
            .map(|&i_layer| LayerNormConfig::new(num_features[i_layer]).init(device))
            .collect();

        Ok(SwinTransformer {
            patch_embed,
            pos_drop: DropoutConfig::new(self.drop_rate).init(),
            num_layers,
            layers,
            num_features,
            norm_layers,
            out_indices: self.out_indices,
            absolute_pos_embed,
        })
    }
}

/// Swin Transformer hierarchical vision backbone outputting 4 multi-scale feature maps.
#[derive(Module, Debug)]
pub struct SwinTransformer {
    patch_embed: PatchEmbed,
    absolute_pos_embed: Option<Param<Tensor<4>>>,
    pos_drop: Dropout,
    num_layers: usize,
    layers: Vec<BasicLayer>,
    norm_layers: Vec<LayerNorm>,
    out_indices: [usize; 4],
    num_features: [usize; 4],
}

impl SwinTransformer {
    /// Runs the 4 Swin stages on `[B, C, H, W]` and returns 4 multi-scale `[B, C_i, H_i, W_i]` maps.
    ///
    /// # Errors
    ///
    /// Returns [`BackboneError`] if the stage outputs do not match the expected 4 feature levels.
    pub fn forward(&self, x: Tensor<4>) -> Result<[Tensor<4>; 4], BackboneError> {
        let [_, _, in_h, in_w] = x.dims();
        if in_h == 0 || in_w == 0 {
            return Err(BackboneError::invalid_input_shape(format!(
                "expected non-zero spatial dimensions, got [{in_h}, {in_w}]"
            )));
        }

        let x = self.patch_embed.forward(x);
        let [_, _, wh, ww] = x.dims();

        let x = match &self.absolute_pos_embed {
            Some(abs_pos) => {
                let interpolated = interpolate(
                    abs_pos.val(),
                    InterpolateOptions::new(InterpolateMode::Bicubic).with_output_size([wh, ww]),
                );
                x + interpolated
            }
            None => x,
        };

        let mut outs = Vec::with_capacity(self.out_indices.len());
        let mut x = self.pos_drop.forward(x.flatten(2, 3).swap_dims(1, 2));
        let mut wh = wh;
        let mut ww = ww;

        for i in 0..self.num_layers {
            let (x_out, out_h, out_w, x_next, next_h, next_w) = self.layers[i].forward(x, wh, ww);
            x = x_next;
            wh = next_h;
            ww = next_w;
            if let Some(norm_pos) = self.out_indices.iter().position(|&idx| idx == i) {
                let x_out = self.norm_layers[norm_pos].forward(x_out);
                let batch_size = x_out.dims()[0];
                let out = x_out
                    .reshape([batch_size, out_h, out_w, self.num_features[i]])
                    .permute([0, 3, 1, 2]);
                outs.push(out);
            }
        }

        outs.try_into().map_err(|_remaining| {
            BackboneError::invalid_input_shape(
                "expected 4 output feature levels from SwinTransformer",
            )
        })
    }

    /// Returns the output channel counts `[C1, C2, C3, C4]` for this Swin configuration.
    #[must_use]
    pub const fn output_channels(&self) -> [usize; 4] {
        self.num_features
    }
}

fn linspace(start: f64, end: f64, steps: usize) -> Vec<f64> {
    match steps {
        0 => Vec::new(),
        1 => vec![start],
        _ => {
            let step_size = (end - start) / usize_to_f64(steps - 1);
            (0..steps)
                .map(|i| usize_to_f64(i).mul_add(step_size, start))
                .collect()
        }
    }
}
