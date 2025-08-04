//! # Pyramid Vision Transformer v2 (PVTv2) Implementation
//!
//! This module provides a Rust implementation of the PVTv2 backbone using the Burn framework.
//! PVTv2 is a hierarchical vision transformer that uses overlapping patch embeddings and
//! spatial reduction attention to achieve a balance between performance and computational cost.
//!
//! ## Architecture Overview
//!
//! The PVTv2 consists of several key components:
//! - **OverlapPatchEmbed**: Converts input images into overlapping patch embeddings.
//! - **Attention**: Spatially reductive attention mechanism.
//! - **Mlp**: Feed-forward network with a depth-wise convolution.
//! - **Block**: The main transformer block combining attention and MLP.
//! - **PyramidVisionTransformerImpr**: The full model with 4 stages.
//!
//! ## Reference
//! Based on "PVTv2: Improved Baselines with Pyramid Vision Transformer"
//! - Paper: https://arxiv.org/abs/2106.13797
//! - Original PyTorch implementation: https://github.com/whai362/PVT

use burn::{
    module::Param,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Dropout, DropoutConfig, Gelu, Initializer, LayerNorm, LayerNormConfig, Linear,
        PaddingConfig2d,
    },
    prelude::*,
    tensor::activation::softmax,
};
use burn_extra_ops::{trunc_normal, DropPath, DropPathConfig};

/// Depth-wise convolution used in the MLP block.
#[derive(Module, Debug)]
pub struct DWConv<B: Backend> {
    dwconv: Conv2d<B>,
}

impl<B: Backend> DWConv<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        let fan_out = (3 * 3 * dim) / dim;
        let std = (2.0 / fan_out as f64).sqrt();
        let conv_initializer = Initializer::Normal { mean: 0.0, std };

        let conv = Conv2dConfig::new([dim, dim], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Same)
            .with_groups(dim)
            .with_bias(true)
            .with_initializer(conv_initializer)
            .init(device);
        Self { dwconv: conv }
    }

    pub fn forward(&self, x: Tensor<B, 3>, h: usize, w: usize) -> Tensor<B, 3> {
        let [b, _, c] = x.dims();
        let x = x.transpose().reshape([b, c, h, w]);
        let x = self.dwconv.forward(x);
        x.flatten(2, 3).transpose()
    }
}

/// Multi-Layer Perceptron (MLP) block for PVTv2.
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1: Linear<B>,
    dwconv: DWConv<B>,
    act: Gelu,
    fc2: Linear<B>,
    drop: Dropout,
}

impl<B: Backend> Mlp<B> {
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        out_features: usize,
        drop: f64,
        device: &B::Device,
    ) -> Self {
        let std = 0.02;
        let fc1 = {
            let weight = trunc_normal(
                Tensor::zeros([in_features, hidden_features], device),
                0.0,
                std,
                -2.0 * std,
                2.0 * std,
            );
            let bias = Tensor::zeros([hidden_features], device);
            Linear {
                weight: Param::from_tensor(weight),
                bias: Some(Param::from_tensor(bias)),
            }
        };
        let fc2 = {
            let weight = trunc_normal(
                Tensor::zeros([hidden_features, out_features], device),
                0.0,
                std,
                -2.0 * std,
                2.0 * std,
            );
            let bias = Tensor::zeros([out_features], device);
            Linear {
                weight: Param::from_tensor(weight),
                bias: Some(Param::from_tensor(bias)),
            }
        };

        Self {
            fc1,
            dwconv: DWConv::new(hidden_features, device),
            act: Gelu::new(),
            fc2,
            drop: DropoutConfig::new(drop).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, h: usize, w: usize) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = self.dwconv.forward(x, h, w);
        let x = self.act.forward(x);
        let x = self.drop.forward(x);
        let x = self.fc2.forward(x);
        self.drop.forward(x)
    }
}

/// Spatially-reductive attention mechanism for PVTv2.
#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    dim: usize,
    num_heads: usize,
    scale: f64,
    q: Linear<B>,
    kv: Linear<B>,
    attn_drop: Dropout,
    proj: Linear<B>,
    proj_drop: Dropout,
    sr_ratio: usize,
    sr: Option<Conv2d<B>>,
    norm: Option<LayerNorm<B>>,
}

impl<B: Backend> Attention<B> {
    pub fn new(
        dim: usize,
        num_heads: usize,
        qkv_bias: bool,
        qk_scale: Option<f64>,
        attn_drop: f64,
        proj_drop: f64,
        sr_ratio: usize,
        epsilon: f64,
        device: &B::Device,
    ) -> Self {
        let head_dim = dim / num_heads;
        let std = 0.02;

        let q = {
            let weight = trunc_normal(
                Tensor::zeros([dim, dim], device),
                0.0,
                std,
                -2.0 * std,
                2.0 * std,
            );
            let bias = if qkv_bias {
                Some(Param::from_tensor(Tensor::zeros([dim], device)))
            } else {
                None
            };
            Linear {
                weight: Param::from_tensor(weight),
                bias,
            }
        };

        let kv = {
            let weight = trunc_normal(
                Tensor::zeros([dim, dim * 2], device),
                0.0,
                std,
                -2.0 * std,
                2.0 * std,
            );
            let bias = if qkv_bias {
                Some(Param::from_tensor(Tensor::zeros([dim * 2], device)))
            } else {
                None
            };
            Linear {
                weight: Param::from_tensor(weight),
                bias,
            }
        };

        let proj = {
            let weight = trunc_normal(
                Tensor::zeros([dim, dim], device),
                0.0,
                std,
                -2.0 * std,
                2.0 * std,
            );
            let bias = Some(Param::from_tensor(Tensor::zeros([dim], device)));
            Linear {
                weight: Param::from_tensor(weight),
                bias,
            }
        };

        let sr = if sr_ratio > 1 {
            let fan_out = (sr_ratio * sr_ratio * dim) / dim;
            let std = (2.0 / fan_out as f64).sqrt();
            let conv_initializer = Initializer::Normal { mean: 0.0, std };
            Some(
                Conv2dConfig::new([dim, dim], [sr_ratio, sr_ratio])
                    .with_stride([sr_ratio, sr_ratio])
                    .with_initializer(conv_initializer)
                    .init(device),
            )
        } else {
            None
        };
        let norm = if sr_ratio > 1 {
            Some(LayerNormConfig::new(dim).with_epsilon(epsilon).init(device))
        } else {
            None
        };

        Self {
            dim,
            num_heads,
            scale: qk_scale.unwrap_or_else(|| (head_dim as f64).powf(-0.5)),
            q,
            kv,
            attn_drop: DropoutConfig::new(attn_drop).init(),
            proj,
            proj_drop: DropoutConfig::new(proj_drop).init(),
            sr_ratio,
            sr,
            norm,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, h: usize, w: usize) -> Tensor<B, 3> {
        let [b, n, c] = x.dims();
        let q = self
            .q
            .forward(x.clone())
            .reshape([b, n, self.num_heads, c / self.num_heads])
            .permute([0, 2, 1, 3]);

        let (k, v) = if let (Some(sr), Some(norm)) = (&self.sr, &self.norm) {
            let x_ = x.permute([0, 2, 1]).reshape([b, c, h, w]);
            let x_ = sr.forward(x_);
            let [_, _, new_h, new_w] = x_.dims();
            let x_ = x_.flatten(2, 3).permute([0, 2, 1]);
            let x_ = norm.forward(x_);
            let kv = self
                .kv
                .forward(x_)
                .reshape([b, new_h * new_w, 2, self.num_heads, c / self.num_heads])
                .permute([2, 0, 3, 1, 4]);
            (
                kv.clone().slice([0; 1]).squeeze(0),
                kv.slice([1; 2]).squeeze(0),
            )
        } else {
            let kv = self
                .kv
                .forward(x)
                .reshape([b, n, 2, self.num_heads, c / self.num_heads])
                .permute([2, 0, 3, 1, 4]);
            (
                kv.clone().slice([0; 1]).squeeze(0),
                kv.slice([1; 2]).squeeze(0),
            )
        };

        let attn = q.matmul(k.transpose()) * self.scale;
        let attn = softmax(attn, 3);
        let attn = self.attn_drop.forward(attn);

        let x = attn.matmul(v).permute([0, 2, 1, 3]).reshape([b, n, c]);
        let x = self.proj.forward(x);
        self.proj_drop.forward(x)
    }
}

/// Transformer block for PVTv2.
#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    norm1: LayerNorm<B>,
    attn: Attention<B>,
    drop_path: DropPath<B>,
    norm2: LayerNorm<B>,
    mlp: Mlp<B>,
}

impl<B: Backend> Block<B> {
    pub fn new(
        dim: usize,
        num_heads: usize,
        mlp_ratio: f64,
        qkv_bias: bool,
        qk_scale: Option<f64>,
        drop: f64,
        attn_drop: f64,
        _drop_path: f64,
        sr_ratio: usize,
        epsilon: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            norm1: LayerNormConfig::new(dim).with_epsilon(epsilon).init(device),
            attn: Attention::new(
                dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio, epsilon, device,
            ),
            drop_path: DropPathConfig::new().init(device),
            norm2: LayerNormConfig::new(dim).with_epsilon(epsilon).init(device),
            mlp: Mlp::new(dim, (dim as f64 * mlp_ratio) as usize, dim, drop, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, h: usize, w: usize) -> Tensor<B, 3> {
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

/// Overlapping patch embedding for PVTv2.
#[derive(Module, Debug)]
pub struct OverlapPatchEmbed<B: Backend> {
    proj: Conv2d<B>,
    norm: LayerNorm<B>,
}

impl<B: Backend> OverlapPatchEmbed<B> {
    pub fn new(
        patch_size: usize,
        stride: usize,
        in_channels: usize,
        embed_dim: usize,
        epsilon: f64,
        device: &B::Device,
    ) -> Self {
        let fan_out = (patch_size * patch_size * embed_dim) as f64;
        let std = (2.0 / fan_out).sqrt();
        let conv_initializer = Initializer::Normal { mean: 0.0, std };

        Self {
            proj: Conv2dConfig::new([in_channels, embed_dim], [patch_size, patch_size])
                .with_stride([stride, stride])
                .with_padding(PaddingConfig2d::Explicit(patch_size / 2, patch_size / 2))
                .with_initializer(conv_initializer)
                .init(device),
            norm: LayerNormConfig::new(embed_dim)
                .with_epsilon(epsilon)
                .init(device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 3>, usize, usize) {
        let x = self.proj.forward(x);
        let [_, _, h, w] = x.dims();
        let x = x.flatten(2, 3).transpose();
        let x = self.norm.forward(x);
        (x, h, w)
    }
}

/// Pyramid Vision Transformer v2 (PVTv2) model.
#[derive(Module, Debug)]
pub struct PyramidVisionTransformerImpr<B: Backend> {
    patch_embed1: OverlapPatchEmbed<B>,
    patch_embed2: OverlapPatchEmbed<B>,
    patch_embed3: OverlapPatchEmbed<B>,
    patch_embed4: OverlapPatchEmbed<B>,
    block1: Vec<Block<B>>,
    norm1: LayerNorm<B>,
    block2: Vec<Block<B>>,
    norm2: LayerNorm<B>,
    block3: Vec<Block<B>>,
    norm3: LayerNorm<B>,
    block4: Vec<Block<B>>,
    norm4: LayerNorm<B>,
    embed_dims: [usize; 4],
}

impl<B: Backend> PyramidVisionTransformerImpr<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> [Tensor<B, 4>; 4] {
        let b = x.dims()[0];

        // Stage 1
        let (mut x, h, w) = self.patch_embed1.forward(x);
        for blk in &self.block1 {
            x = blk.forward(x.clone(), h, w);
        }
        x = self.norm1.forward(x);
        let out1 = x
            .reshape([b, h, w, self.embed_dims[0]])
            .permute([0, 3, 1, 2]);

        // Stage 2
        let (x, h, w) = self.patch_embed2.forward(out1.clone());
        let mut x = x;
        for blk in &self.block2 {
            x = blk.forward(x.clone(), h, w);
        }
        x = self.norm2.forward(x);
        let out2 = x
            .reshape([b, h, w, self.embed_dims[1]])
            .permute([0, 3, 1, 2]);

        // Stage 3
        let (x, h, w) = self.patch_embed3.forward(out2.clone());
        let mut x = x;
        for blk in &self.block3 {
            x = blk.forward(x.clone(), h, w);
        }
        x = self.norm3.forward(x);
        let out3 = x
            .reshape([b, h, w, self.embed_dims[2]])
            .permute([0, 3, 1, 2]);

        // Stage 4
        let (x, h, w) = self.patch_embed4.forward(out3.clone());
        let mut x = x;
        for blk in &self.block4 {
            x = blk.forward(x.clone(), h, w);
        }
        x = self.norm4.forward(x);
        let out4 = x
            .reshape([b, h, w, self.embed_dims[3]])
            .permute([0, 3, 1, 2]);

        [out1, out2, out3, out4]
    }

    pub const fn output_channels(&self) -> [usize; 4] {
        self.embed_dims
    }
}

#[derive(Config, Debug)]
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
    pub fn init<B: Backend>(&self, device: &B::Device) -> PyramidVisionTransformerImpr<B> {
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

        let dpr: Vec<f64> = (0..self.depths.iter().sum())
            .map(|i| {
                self.drop_path_rate * (i as f64) / ((self.depths.iter().sum::<usize>() - 1) as f64)
            })
            .collect();

        let mut cur = 0;
        let block1 = (0..self.depths[0])
            .map(|i| {
                Block::new(
                    self.embed_dims[0],
                    self.num_heads[0],
                    self.mlp_ratios[0],
                    self.qkv_bias,
                    None,
                    self.drop_rate,
                    0.0,
                    dpr[cur + i],
                    self.sr_ratios[0],
                    self.epsilon,
                    device,
                )
            })
            .collect();
        let norm1 = LayerNormConfig::new(self.embed_dims[0])
            .with_epsilon(self.epsilon)
            .init(device);

        cur += self.depths[0];
        let block2 = (0..self.depths[1])
            .map(|i| {
                Block::new(
                    self.embed_dims[1],
                    self.num_heads[1],
                    self.mlp_ratios[1],
                    self.qkv_bias,
                    None,
                    self.drop_rate,
                    0.0,
                    dpr[cur + i],
                    self.sr_ratios[1],
                    self.epsilon,
                    device,
                )
            })
            .collect();
        let norm2 = LayerNormConfig::new(self.embed_dims[1])
            .with_epsilon(self.epsilon)
            .init(device);

        cur += self.depths[1];
        let block3 = (0..self.depths[2])
            .map(|i| {
                Block::new(
                    self.embed_dims[2],
                    self.num_heads[2],
                    self.mlp_ratios[2],
                    self.qkv_bias,
                    None,
                    self.drop_rate,
                    0.0,
                    dpr[cur + i],
                    self.sr_ratios[2],
                    self.epsilon,
                    device,
                )
            })
            .collect();
        let norm3 = LayerNormConfig::new(self.embed_dims[2])
            .with_epsilon(self.epsilon)
            .init(device);

        cur += self.depths[2];
        let block4 = (0..self.depths[3])
            .map(|i| {
                Block::new(
                    self.embed_dims[3],
                    self.num_heads[3],
                    self.mlp_ratios[3],
                    self.qkv_bias,
                    None,
                    self.drop_rate,
                    0.0,
                    dpr[cur + i],
                    self.sr_ratios[3],
                    self.epsilon,
                    device,
                )
            })
            .collect();
        let norm4 = LayerNormConfig::new(self.embed_dims[3])
            .with_epsilon(self.epsilon)
            .init(device);

        PyramidVisionTransformerImpr {
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
        }
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_pvt_v2_b2_forward() {
        let device = Default::default();
        let config = PvtV2Config::b2(3);
        let model = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 4>::random(
            [1, 3, 224, 224],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let output = model.forward(input);

        // Check output shapes for PVTv2-B2
        assert_eq!(output[0].dims(), [1, 64, 56, 56]);
        assert_eq!(output[1].dims(), [1, 128, 28, 28]);
        assert_eq!(output[2].dims(), [1, 320, 14, 14]);
        assert_eq!(output[3].dims(), [1, 512, 7, 7]);
    }
}
