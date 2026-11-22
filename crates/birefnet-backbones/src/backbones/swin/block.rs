//! Swin Transformer block, MLP, and window partition helpers.

use birefnet_extra_ops::{DropPath, DropPathConfig};
use burn::{
    nn::{Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::*,
};

use crate::backbones::{
    scaled_hidden_dim,
    swin::attention::{WindowAttention, WindowAttentionConfig},
    usize_to_i64,
};

/// Configuration for the two-layer MLP in a Swin Transformer block.
#[derive(Config, Debug, PartialEq)]
pub struct MlpConfig {
    in_features: usize,
    #[config(default = "None")]
    hidden_features: Option<usize>,
    #[config(default = "None")]
    out_features: Option<usize>,
    #[config(default = 0.0)]
    drop: f64,
}

impl MlpConfig {
    /// Initializes an [`Mlp`] module.
    #[must_use]
    pub fn init(&self, device: &Device) -> Mlp {
        let out_features = self.out_features.unwrap_or(self.in_features);
        let hidden_features = self.hidden_features.unwrap_or(self.in_features);

        Mlp {
            fc1: LinearConfig::new(self.in_features, hidden_features).init(device),
            act: Gelu::new(),
            fc2: LinearConfig::new(hidden_features, out_features).init(device),
            drop: DropoutConfig::new(self.drop).init(),
        }
    }
}

/// Feed-forward network used in each Swin Transformer block.
#[derive(Module, Debug)]
pub struct Mlp {
    fc1: Linear,
    act: Gelu,
    fc2: Linear,
    drop: Dropout,
}

impl Mlp {
    /// Applies `Linear -> Gelu -> Dropout -> Linear -> Dropout`.
    #[must_use]
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let x = self.drop.forward(self.act.forward(self.fc1.forward(x)));
        self.drop.forward(self.fc2.forward(x))
    }
}

/// Partitions `[B, H, W, C]` into non-overlapping `[num_windows * B, ws, ws, C]` windows.
pub(crate) fn window_partition(x: Tensor<4>, window_size: usize) -> Tensor<4> {
    let [batch, height, width, channels] = x.dims();
    let num_win_h = height / window_size;
    let num_win_w = width / window_size;
    x.reshape([
        batch,
        num_win_h,
        window_size,
        num_win_w,
        window_size,
        channels,
    ])
    .permute([0, 1, 3, 2, 4, 5])
    .reshape([
        batch * num_win_h * num_win_w,
        window_size,
        window_size,
        channels,
    ])
}

/// Merges `[num_windows * B, ws, ws, C]` windows back to `[B, H, W, C]`.
pub(crate) fn window_reverse(
    windows: Tensor<4>,
    window_size: usize,
    h: usize,
    w: usize,
) -> Tensor<4> {
    let [total_windows, _, _, channels] = windows.dims();
    let nh = h / window_size;
    let nw = w / window_size;
    let b = total_windows / (nh * nw);
    windows
        .reshape([b, nh, nw, window_size, window_size, channels])
        .permute([0, 1, 3, 2, 4, 5])
        .reshape([b, h, w, channels])
}

/// Configuration for [`SwinTransformerBlock`].
#[derive(Config, Debug, PartialEq)]
pub struct SwinTransformerBlockConfig {
    dim: usize,
    num_heads: usize,
    #[config(default = 7)]
    window_size: usize,
    #[config(default = 0)]
    shift_size: usize,
    #[config(default = 4.0)]
    mlp_ratio: f64,
    #[config(default = true)]
    qkv_bias: bool,
    #[config(default = "None")]
    qk_scale: Option<f64>,
    #[config(default = 0.0)]
    drop: f64,
    #[config(default = 0.0)]
    attn_drop: f64,
    #[config(default = 0.0)]
    drop_path: f64,
}

impl SwinTransformerBlockConfig {
    /// Initializes a [`SwinTransformerBlock`].
    #[must_use]
    pub fn init(&self, device: &Device) -> SwinTransformerBlock {
        let norm1 = LayerNormConfig::new(self.dim).init(device);
        let attn = WindowAttentionConfig::new(
            self.dim,
            [self.window_size, self.window_size],
            self.num_heads,
        )
        .with_qkv_bias(self.qkv_bias)
        .with_qk_scale(self.qk_scale)
        .with_attn_drop(self.attn_drop)
        .with_proj_drop(self.drop)
        .init(device);

        let drop_path = DropPathConfig::new().with_drop_prob(self.drop_path).init();
        let norm2 = LayerNormConfig::new(self.dim).init(device);
        let mlp_hidden_dim = scaled_hidden_dim(self.dim, self.mlp_ratio);
        let mlp = MlpConfig::new(self.dim)
            .with_hidden_features(Some(mlp_hidden_dim))
            .with_drop(self.drop)
            .init(device);

        SwinTransformerBlock {
            window_size: self.window_size,
            shift_size: self.shift_size,
            norm1,
            attn,
            drop_path,
            norm2,
            mlp,
        }
    }
}

/// Core Swin Transformer block combining regular or shifted window attention with an MLP.
#[derive(Module, Debug)]
pub struct SwinTransformerBlock {
    window_size: usize,
    shift_size: usize,
    norm1: LayerNorm,
    attn: WindowAttention,
    norm2: LayerNorm,
    mlp: Mlp,
    drop_path: DropPath,
}

impl SwinTransformerBlock {
    /// Returns `true` if this block uses shifted windows (and therefore an attention mask).
    #[must_use]
    pub const fn is_shifted(&self) -> bool {
        self.shift_size > 0
    }

    /// Applies attention and MLP residual branches to `[B, H * W, C]`.
    ///
    /// # Panics
    ///
    /// Panics if this block uses shifted windows and `mask_matrix` is `None`.
    #[must_use]
    pub fn forward(
        &self,
        x: Tensor<3>,
        height: usize,
        width: usize,
        mask_matrix: Option<Tensor<3>>,
    ) -> Tensor<3> {
        let [batch, _tokens, channels] = x.dims();

        let shortcut = x.clone();
        let normalized = self
            .norm1
            .forward(x)
            .reshape([batch, height, width, channels]);

        let pad_r = (self.window_size - width % self.window_size) % self.window_size;
        let pad_b = (self.window_size - height % self.window_size) % self.window_size;
        let padded = normalized
            .permute([0, 3, 1, 2])
            .pad((0, pad_r, 0, pad_b), 0.0)
            .permute([0, 2, 3, 1]);
        let [_, padded_h, padded_w, _] = padded.dims();

        let shift = usize_to_i64(self.shift_size);
        let (shifted_x, attn_mask) = if self.shift_size > 0 {
            let mask = mask_matrix.unwrap_or_else(|| {
                panic!(
                    "shifted Swin block (shift_size = {}) requires an attention mask",
                    self.shift_size
                )
            });
            (padded.roll(&[-shift, -shift], &[1, 2]), Some(mask))
        } else {
            (padded, None)
        };

        let x_window = window_partition(shifted_x, self.window_size);
        let num_windows = x_window.dims()[0];
        let x_window =
            x_window.reshape([num_windows, self.window_size * self.window_size, channels]);

        let attn_window = self.attn.forward(x_window, attn_mask);
        let attn_num_windows = attn_window.dims()[0];
        let attn_window = attn_window.reshape([
            attn_num_windows,
            self.window_size,
            self.window_size,
            channels,
        ]);
        let unshifted = window_reverse(attn_window, self.window_size, padded_h, padded_w);

        let restored = if self.shift_size > 0 {
            unshifted.roll(&[shift, shift], &[1, 2])
        } else {
            unshifted
        };

        let cropped = if pad_r > 0 || pad_b > 0 {
            restored.slice(s![.., 0..height, 0..width, ..])
        } else {
            restored
        };

        let flat = cropped.reshape([batch, height * width, channels]);
        let residual = shortcut + self.drop_path.forward(flat);
        residual.clone()
            + self
                .drop_path
                .forward(self.mlp.forward(self.norm2.forward(residual)))
    }
}
