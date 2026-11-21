//! Pyramid Vision Transformer v2 (`PVTv2`) backbone implementation.

use birefnet_extra_ops::{DropPath, DropPathConfig, trunc_normal};
use burn::{
    module::Param,
    nn::{
        Dropout, DropoutConfig, Gelu, Initializer, LayerNorm, LayerNormConfig, Linear,
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
    tensor::activation::softmax,
};

use crate::backbones::usize_to_f64;
use crate::error::BackboneError;

fn trunc_normal_linear(
    in_features: usize,
    out_features: usize,
    bias: bool,
    device: &Device,
) -> Linear {
    let std = 0.02;
    let weight = trunc_normal(
        [in_features, out_features],
        0.0,
        std,
        -2.0 * std,
        2.0 * std,
        device,
    );
    let bias = bias.then(|| Param::from_tensor(Tensor::zeros([out_features], device)));
    Linear {
        weight: Param::from_tensor(weight),
        bias,
    }
}

/// Depth-wise convolution used in the `PVTv2` MLP block.
#[derive(Module, Debug)]
pub struct DWConv {
    dwconv: Conv2d,
}

impl DWConv {
    /// Initializes a 3x3 depth-wise convolution for `dim` channels.
    #[must_use]
    pub fn new(dim: usize, device: &Device) -> Self {
        let fan_out = usize_to_f64(3 * 3);
        let std = (2.0 / fan_out).sqrt();
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
    #[must_use]
    pub fn forward(&self, x: Tensor<3>, height: usize, width: usize) -> Tensor<3> {
        let [batch, _, channels] = x.dims();
        let spatial = x.transpose().reshape([batch, channels, height, width]);
        self.dwconv.forward(spatial).flatten(2, 3).transpose()
    }
}

/// Multi-layer perceptron block with embedded depth-wise convolution.
#[derive(Module, Debug)]
pub struct Mlp {
    fc1: Linear,
    dwconv: DWConv,
    act: Gelu,
    fc2: Linear,
    drop: Dropout,
}

impl Mlp {
    /// Initializes a `PVTv2` [`Mlp`] block.
    #[must_use]
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
        drop: f64,
        device: &Device,
    ) -> Self {
        Self {
            fc1: trunc_normal_linear(in_features, hidden_features, true, device),
            dwconv: DWConv::new(hidden_features, device),
            act: Gelu::new(),
            fc2: trunc_normal_linear(hidden_features, out_features, true, device),
            drop: DropoutConfig::new(drop).init(),
        }
    }

    /// Applies `fc1 -> dwconv -> gelu -> drop -> fc2 -> drop`.
    #[must_use]
    pub fn forward(&self, x: Tensor<3>, h: usize, w: usize) -> Tensor<3> {
        let x = self.drop.forward(
            self.act
                .forward(self.dwconv.forward(self.fc1.forward(x), h, w)),
        );
        self.drop.forward(self.fc2.forward(x))
    }
}

/// Spatially-reductive multi-head attention module.
#[derive(Module, Debug)]
pub struct Attention {
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
    /// Initializes a `PVTv2` [`Attention`] module.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the upstream PVTv2 attention signature directly"
    )]
    pub fn new(
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        qk_scale: Option<f64>,
        attn_drop: f64,
        proj_drop: f64,
        sr_ratio: usize,
        epsilon: f64,
        device: &Device,
    ) -> Self {
        let head_dim = dim / num_heads;
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

        Self {
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

    /// Computes spatially-reduced multi-head self-attention on `[B, H * W, C]`.
    #[must_use]
    pub fn forward(&self, x: Tensor<3>, height: usize, width: usize) -> Tensor<3> {
        let [batch, seq_len, channels] = x.dims();
        let head_dim = channels / self.num_heads;
        let query = self
            .q
            .forward(x.clone())
            .reshape([batch, seq_len, self.num_heads, head_dim])
            .permute([0, 2, 1, 3]);

        let (key, value) = if let (Some(sr), Some(norm)) = (&self.sr, &self.norm) {
            let reduced = sr.forward(
                x.permute([0, 2, 1])
                    .reshape([batch, channels, height, width]),
            );
            let [_, _, new_h, new_w] = reduced.dims();
            let reduced = norm.forward(reduced.flatten(2, 3).permute([0, 2, 1]));
            let kv = self
                .kv
                .forward(reduced)
                .reshape([batch, new_h * new_w, 2, self.num_heads, head_dim])
                .permute([2, 0, 3, 1, 4]);
            (
                kv.clone().slice(s![0..1]).squeeze_dim::<4>(0),
                kv.slice(s![1..2]).squeeze_dim::<4>(0),
            )
        } else {
            let kv = self
                .kv
                .forward(x)
                .reshape([batch, seq_len, 2, self.num_heads, head_dim])
                .permute([2, 0, 3, 1, 4]);
            (
                kv.clone().slice(s![0..1]).squeeze_dim::<4>(0),
                kv.slice(s![1..2]).squeeze_dim::<4>(0),
            )
        };

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

/// Transformer block for `PVTv2`.
#[derive(Module, Debug)]
pub struct Block {
    norm1: LayerNorm,
    attn: Attention,
    drop_path: DropPath,
    norm2: LayerNorm,
    mlp: Mlp,
}

impl Block {
    /// Initializes a `PVTv2` [`Block`].
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "mirrors the upstream PVTv2 transformer block hyperparameters"
    )]
    pub fn new(
        dim: usize,
        num_heads: usize,
        mlp_ratio: f64,
        qkv_bias: bool,
        qk_scale: Option<f64>,
        drop: f64,
        attn_drop: f64,
        drop_path: f64,
        sr_ratio: usize,
        epsilon: f64,
        device: &Device,
    ) -> Self {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "MLP ratio is a small non-negative scalar"
        )]
        let mlp_hidden_dim = (usize_to_f64(dim) * mlp_ratio) as usize;
        Self {
            norm1: LayerNormConfig::new(dim).with_epsilon(epsilon).init(device),
            attn: Attention::new(
                dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio, epsilon, device,
            ),
            drop_path: DropPathConfig::new().with_drop_prob(drop_path).init(),
            norm2: LayerNormConfig::new(dim).with_epsilon(epsilon).init(device),
            mlp: Mlp::new(dim, mlp_hidden_dim, dim, drop, device),
        }
    }

    /// Runs attention and MLP residual branches for spatial dimensions `(h, w)`.
    #[must_use]
    pub fn forward(&self, x: Tensor<3>, h: usize, w: usize) -> Tensor<3> {
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

/// Overlapping patch embedding layer for `PVTv2`.
#[derive(Module, Debug)]
pub struct OverlapPatchEmbed {
    proj: Conv2d,
    norm: LayerNorm,
}

impl OverlapPatchEmbed {
    /// Initializes an [`OverlapPatchEmbed`] layer.
    #[must_use]
    pub fn new(
        patch_size: usize,
        stride: usize,
        in_channels: usize,
        embed_dim: usize,
        epsilon: f64,
        device: &Device,
    ) -> Self {
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
    #[must_use]
    pub fn forward(&self, x: Tensor<4>) -> (Tensor<3>, usize, usize) {
        let x = self.proj.forward(x);
        let [_, _, h, w] = x.dims();
        let x = self.norm.forward(x.flatten(2, 3).transpose());
        (x, h, w)
    }
}

/// Pyramid Vision Transformer v2 (`PVTv2`) hierarchical backbone.
#[derive(Module, Debug)]
pub struct PyramidVisionTransformerImpr {
    patch_embed1: OverlapPatchEmbed,
    patch_embed2: OverlapPatchEmbed,
    patch_embed3: OverlapPatchEmbed,
    patch_embed4: OverlapPatchEmbed,
    block1: Vec<Block>,
    norm1: LayerNorm,
    block2: Vec<Block>,
    norm2: LayerNorm,
    block3: Vec<Block>,
    norm3: LayerNorm,
    block4: Vec<Block>,
    norm4: LayerNorm,
    embed_dims: [usize; 4],
}

impl PyramidVisionTransformerImpr {
    /// Runs the 4 `PVTv2` stages on `[B, C, H, W]` and returns 4 multi-scale `[B, C_i, H_i, W_i]` maps.
    #[must_use]
    pub fn forward(&self, x: Tensor<4>) -> [Tensor<4>; 4] {
        let b = x.dims()[0];
        let out1 = run_stage(
            &self.patch_embed1,
            &self.block1,
            &self.norm1,
            x,
            b,
            self.embed_dims[0],
        );
        let out2 = run_stage(
            &self.patch_embed2,
            &self.block2,
            &self.norm2,
            out1.clone(),
            b,
            self.embed_dims[1],
        );
        let out3 = run_stage(
            &self.patch_embed3,
            &self.block3,
            &self.norm3,
            out2.clone(),
            b,
            self.embed_dims[2],
        );
        let out4 = run_stage(
            &self.patch_embed4,
            &self.block4,
            &self.norm4,
            out3.clone(),
            b,
            self.embed_dims[3],
        );
        [out1, out2, out3, out4]
    }

    /// Returns the output channel counts `[C1, C2, C3, C4]` for each stage.
    #[must_use]
    pub const fn output_channels(&self) -> [usize; 4] {
        self.embed_dims
    }
}

fn run_stage(
    patch_embed: &OverlapPatchEmbed,
    blocks: &[Block],
    norm: &LayerNorm,
    input: Tensor<4>,
    batch_size: usize,
    embed_dim: usize,
) -> Tensor<4> {
    let (mut x, h, w) = patch_embed.forward(input);
    for blk in blocks {
        x = blk.forward(x, h, w);
    }
    norm.forward(x)
        .reshape([batch_size, h, w, embed_dim])
        .permute([0, 3, 1, 2])
}

/// Configuration for [`PyramidVisionTransformerImpr`].
#[derive(Config, Debug, PartialEq)]
pub struct PvtV2Config {
    in_channels: usize,
    embed_dims: [usize; 4],
    num_heads: [usize; 4],
    mlp_ratios: [f64; 4],
    qkv_bias: bool,
    depths: [usize; 4],
    sr_ratios: [usize; 4],
    drop_rate: f64,
    drop_path_rate: f64,
    #[config(default = 1e-6)]
    epsilon: f64,
}

impl PvtV2Config {
    /// Validates hyperparameters and initializes a [`PyramidVisionTransformerImpr`].
    ///
    /// # Errors
    ///
    /// Returns [`BackboneError`] if channel dims, head counts, or spatial reduction ratios are invalid.
    pub fn init(&self, device: &Device) -> Result<PyramidVisionTransformerImpr, BackboneError> {
        if self.in_channels == 0 {
            return Err(BackboneError::invalid_config(
                "in_channels must be positive",
            ));
        }
        for i in 0..4 {
            if self.embed_dims[i] == 0
                || self.num_heads[i] == 0
                || !self.embed_dims[i].is_multiple_of(self.num_heads[i])
            {
                return Err(BackboneError::invalid_config(format!(
                    "stage {i} embed_dim ({}) must be positive and divisible by num_heads ({})",
                    self.embed_dims[i], self.num_heads[i]
                )));
            }
            if self.sr_ratios[i] == 0 {
                return Err(BackboneError::invalid_config(format!(
                    "stage {i} sr_ratio must be positive"
                )));
            }
        }

        let total_depth: usize = self.depths.iter().sum();
        let dpr: Vec<f64> = if total_depth <= 1 {
            vec![0.0; total_depth]
        } else {
            let denom = usize_to_f64(total_depth - 1);
            (0..total_depth)
                .map(|i| self.drop_path_rate * usize_to_f64(i) / denom)
                .collect()
        };

        let patch_embed1 = OverlapPatchEmbed::new(
            7,
            4,
            self.in_channels,
            self.embed_dims[0],
            self.epsilon,
            device,
        );
        let patch_embed2 = OverlapPatchEmbed::new(
            3,
            2,
            self.embed_dims[0],
            self.embed_dims[1],
            self.epsilon,
            device,
        );
        let patch_embed3 = OverlapPatchEmbed::new(
            3,
            2,
            self.embed_dims[1],
            self.embed_dims[2],
            self.epsilon,
            device,
        );
        let patch_embed4 = OverlapPatchEmbed::new(
            3,
            2,
            self.embed_dims[2],
            self.embed_dims[3],
            self.epsilon,
            device,
        );

        let mut cursor = 0_usize;
        let (block1, norm1) = self.build_stage(0, &dpr[cursor..cursor + self.depths[0]], device);
        cursor += self.depths[0];
        let (block2, norm2) = self.build_stage(1, &dpr[cursor..cursor + self.depths[1]], device);
        cursor += self.depths[1];
        let (block3, norm3) = self.build_stage(2, &dpr[cursor..cursor + self.depths[2]], device);
        cursor += self.depths[2];
        let (block4, norm4) = self.build_stage(3, &dpr[cursor..cursor + self.depths[3]], device);

        Ok(PyramidVisionTransformerImpr {
            patch_embed1,
            patch_embed2,
            patch_embed3,
            patch_embed4,
            block1,
            norm1,
            block2,
            norm2,
            block3,
            norm3,
            block4,
            norm4,
            embed_dims: self.embed_dims,
        })
    }

    fn build_stage(
        &self,
        stage: usize,
        stage_dpr: &[f64],
        device: &Device,
    ) -> (Vec<Block>, LayerNorm) {
        let blocks = stage_dpr
            .iter()
            .map(|&dp| {
                Block::new(
                    self.embed_dims[stage],
                    self.num_heads[stage],
                    self.mlp_ratios[stage],
                    self.qkv_bias,
                    None,
                    self.drop_rate,
                    0.0,
                    dp,
                    self.sr_ratios[stage],
                    self.epsilon,
                    device,
                )
            })
            .collect();
        let norm = LayerNormConfig::new(self.embed_dims[stage])
            .with_epsilon(self.epsilon)
            .init(device);
        (blocks, norm)
    }

    /// PVTv2-B0 configuration.
    #[must_use]
    pub const fn b0(in_channels: usize) -> Self {
        Self {
            in_channels,
            embed_dims: [32, 64, 160, 256],
            num_heads: [1, 2, 5, 8],
            mlp_ratios: [8.0, 8.0, 4.0, 4.0],
            qkv_bias: true,
            depths: [2, 2, 2, 2],
            sr_ratios: [8, 4, 2, 1],
            drop_rate: 0.0,
            drop_path_rate: 0.1,
            epsilon: 1e-6,
        }
    }

    /// PVTv2-B1 configuration.
    #[must_use]
    pub const fn b1(in_channels: usize) -> Self {
        Self {
            in_channels,
            embed_dims: [64, 128, 320, 512],
            num_heads: [1, 2, 5, 8],
            mlp_ratios: [8.0, 8.0, 4.0, 4.0],
            qkv_bias: true,
            depths: [2, 2, 2, 2],
            sr_ratios: [8, 4, 2, 1],
            drop_rate: 0.0,
            drop_path_rate: 0.1,
            epsilon: 1e-6,
        }
    }

    /// PVTv2-B2 configuration.
    #[must_use]
    pub const fn b2(in_channels: usize) -> Self {
        Self {
            in_channels,
            embed_dims: [64, 128, 320, 512],
            num_heads: [1, 2, 5, 8],
            mlp_ratios: [8.0, 8.0, 4.0, 4.0],
            qkv_bias: true,
            depths: [3, 4, 6, 3],
            sr_ratios: [8, 4, 2, 1],
            drop_rate: 0.0,
            drop_path_rate: 0.1,
            epsilon: 1e-6,
        }
    }

    /// PVTv2-B3 configuration.
    #[must_use]
    pub const fn b3(in_channels: usize) -> Self {
        Self {
            in_channels,
            embed_dims: [64, 128, 320, 512],
            num_heads: [1, 2, 5, 8],
            mlp_ratios: [8.0, 8.0, 4.0, 4.0],
            qkv_bias: true,
            depths: [3, 4, 18, 3],
            sr_ratios: [8, 4, 2, 1],
            drop_rate: 0.0,
            drop_path_rate: 0.1,
            epsilon: 1e-6,
        }
    }

    /// PVTv2-B4 configuration.
    #[must_use]
    pub const fn b4(in_channels: usize) -> Self {
        Self {
            in_channels,
            embed_dims: [64, 128, 320, 512],
            num_heads: [1, 2, 5, 8],
            mlp_ratios: [8.0, 8.0, 4.0, 4.0],
            qkv_bias: true,
            depths: [3, 8, 27, 3],
            sr_ratios: [8, 4, 2, 1],
            drop_rate: 0.0,
            drop_path_rate: 0.1,
            epsilon: 1e-6,
        }
    }

    /// PVTv2-B5 configuration.
    #[must_use]
    pub const fn b5(in_channels: usize) -> Self {
        Self {
            in_channels,
            embed_dims: [64, 128, 320, 512],
            num_heads: [1, 2, 5, 8],
            mlp_ratios: [4.0, 4.0, 4.0, 4.0],
            qkv_bias: true,
            depths: [3, 6, 40, 3],
            sr_ratios: [8, 4, 2, 1],
            drop_rate: 0.0,
            drop_path_rate: 0.1,
            epsilon: 1e-6,
        }
    }
}
