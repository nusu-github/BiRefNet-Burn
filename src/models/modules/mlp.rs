use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig,
        PaddingConfig2d,
    },
    prelude::*,
};

use crate::special::{DropPath, DropPathConfig};

#[derive(Config, Debug)]
pub struct MLPLayerConfig {
    in_features: usize,
    hidden_features: Option<usize>,
    out_features: Option<usize>,
    // act_layer: nn.GELU,
    #[config(default = "0.0")]
    drop: f64,
}

impl MLPLayerConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> MLPLayer<B> {
        let out_features = self.out_features.unwrap_or(self.in_features);
        let hidden_features = self.hidden_features.unwrap_or(self.in_features);
        let fc1 = LinearConfig::new(self.in_features, out_features).init(device);
        let act = Gelu::new();
        let drop = DropoutConfig::new(self.drop).init();
        let fc2 = LinearConfig::new(out_features, hidden_features).init(device);
        MLPLayer {
            fc1,
            act,
            drop,
            fc2,
        }
    }
}

#[derive(Module, Debug)]
pub struct MLPLayer<B: Backend> {
    fc1: Linear<B>,
    act: Gelu,
    fc2: Linear<B>,
    drop: Dropout,
}

impl<B: Backend> MLPLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = self.act.forward(x);
        let x = self.drop.forward(x);
        let x = self.fc2.forward(x);

        self.drop.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct AttentionConfig {
    dim: usize,
    #[config(default = "8")]
    num_heads: usize,
    #[config(default = "false")]
    qkv_bias: bool,
    #[config(default = "None")]
    qk_scale: Option<f64>,
    #[config(default = "0.0")]
    attn_drop: f64,
    #[config(default = "0.0")]
    proj_drop: f64,
    #[config(default = "1")]
    sr_ratio: usize,
}

impl AttentionConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Attention<B> {
        let head_dim = self.dim / self.num_heads;
        let scale = self.qk_scale.unwrap_or(head_dim as f64).powf(-0.5);

        let q = LinearConfig::new(self.dim, self.dim).init(device);
        let kv = LinearConfig::new(self.dim, self.dim * 2).init(device);
        let proj = LinearConfig::new(self.dim, self.dim).init(device);

        let sr = if self.sr_ratio > 1 {
            Some(
                Conv2dConfig::new([self.dim, self.dim], [self.sr_ratio, self.sr_ratio])
                    .init(device),
            )
        } else {
            None
        };

        let norm = if self.sr_ratio > 1 {
            Some(LayerNormConfig::new(self.dim).init(device))
        } else {
            None
        };

        Attention {
            num_heads: self.num_heads,
            scale,
            sr_ratio: self.sr_ratio,
            q,
            kv,
            proj,
            attn_drop: DropoutConfig::new(self.attn_drop).init(),
            proj_drop: DropoutConfig::new(self.proj_drop).init(),
            sr,
            norm,
        }
    }
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    num_heads: usize,
    scale: f64,
    sr_ratio: usize,
    q: Linear<B>,
    kv: Linear<B>,
    proj: Linear<B>,
    attn_drop: Dropout,
    proj_drop: Dropout,
    sr: Option<Conv2d<B>>,
    norm: Option<LayerNorm<B>>,
}

impl<B: Backend> Attention<B> {
    pub fn forward(&self, x: Tensor<B, 3>, h: usize, w: usize) -> Tensor<B, 3> {
        todo!()
    }
}

#[derive(Config, Debug)]
pub struct BlockConfig {
    dim: usize,
    #[config(default = "8")]
    num_heads: usize,
    #[config(default = "4.0")]
    mlp_ratio: f64,
    #[config(default = "false")]
    qkv_bias: bool,
    #[config(default = "None")]
    qk_scale: Option<f64>,
    #[config(default = "0.0")]
    drop: f64,
    #[config(default = "0.0")]
    attn_drop: f64,
    #[config(default = "0.0")]
    drop_path: f64,
    // act_layer: nn.GELU,
    // norm_layer: nn.LayerNorm,
    #[config(default = "1")]
    sr_ratio: usize,
}

impl BlockConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Block<B> {
        let norm1 = LayerNormConfig::new(self.dim).init(device);
        let attn = AttentionConfig::new(self.dim)
            .with_num_heads(self.num_heads)
            .with_qkv_bias(self.qkv_bias)
            .with_qk_scale(self.qk_scale)
            .with_attn_drop(self.attn_drop)
            .with_proj_drop(self.drop)
            .with_sr_ratio(self.sr_ratio)
            .init(device);
        let norm2 = LayerNormConfig::new(self.dim).init(device);
        let mlp_hidden_dim = (self.dim as f64 * self.mlp_ratio) as usize;
        let mlp = MLPLayerConfig::new(self.dim)
            .with_hidden_features(Some(mlp_hidden_dim))
            .with_drop(self.drop)
            .init(device);
        let drop_path = if self.drop_path > 0.0 {
            Some(DropPathConfig::new().with_drop_prob(self.drop_path).init())
        } else {
            None
        };

        Block {
            norm1,
            attn,
            norm2,
            mlp,
            drop_path,
        }
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    norm1: LayerNorm<B>,
    attn: Attention<B>,
    norm2: LayerNorm<B>,
    mlp: MLPLayer<B>,
    drop_path: Option<DropPath>,
}

impl<B: Backend> Block<B> {
    pub fn forward(&self, x: Tensor<B, 3>, h: usize, w: usize) -> Tensor<B, 3> {
        let x = x.clone();
        let shortcut = x.clone();

        let x = self.norm1.forward(x);
        let x = self.attn.forward(x, h, w);
        let x = if let Some(drop_path) = &self.drop_path {
            drop_path.forward(x)
        } else {
            x
        };
        let x = x + shortcut;

        let shortcut = x.clone();
        let x = self.norm2.forward(x);
        let x = self.mlp.forward(x);
        let x = if let Some(drop_path) = &self.drop_path {
            drop_path.forward(x)
        } else {
            x
        };
        x + shortcut
    }
}

#[derive(Config, Debug)]
pub struct OverlapPatchEmbedConfig {
    #[config(default = "224")]
    img_size: usize,
    #[config(default = "7")]
    patch_size: usize,
    #[config(default = "4")]
    stride: usize,
    #[config(default = "3")]
    in_channels: usize,
    #[config(default = "768")]
    embed_dim: usize,
}

impl OverlapPatchEmbedConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> OverlapPatchEmbed<B> {
        let proj = Conv2dConfig::new(
            [self.in_channels, self.embed_dim],
            [self.patch_size, self.patch_size],
        )
        .with_stride([self.stride, self.stride])
        .with_padding(PaddingConfig2d::Explicit(
            self.patch_size / 2,
            self.patch_size / 2,
        ))
        .init(device);
        let norm = LayerNormConfig::new(self.embed_dim).init(device);

        OverlapPatchEmbed { proj, norm }
    }
}

#[derive(Module, Debug)]
pub struct OverlapPatchEmbed<B: Backend> {
    proj: Conv2d<B>,
    norm: LayerNorm<B>,
}

impl<B: Backend> OverlapPatchEmbed<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> (Tensor<B, 3>, usize, usize) {
        let x = self.proj.forward(x);
        let [_, _, h, w] = x.dims();
        let x = x.flatten(0, 2).transpose();
        let x = self.norm.forward(x);
        (x, h, w)
    }
}
