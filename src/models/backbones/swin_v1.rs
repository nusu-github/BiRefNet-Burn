use burn::{module::Param, nn, prelude::*};

use crate::special::drop_path::DropPath;

#[derive(Config)]
pub struct MlpConfig {
    in_features: usize,
    hidden_features: usize,
    out_features: usize,
    // act_layer:Module,
    drop: f64,
}

impl MlpConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Mlp<B> {
        let fc1 = nn::LinearConfig::new(self.in_features, self.hidden_features).init(device);
        let act = nn::Gelu::new();
        let fc2 = nn::LinearConfig::new(self.hidden_features, self.out_features).init(device);
        let drop = nn::DropoutConfig::new(self.drop).init();

        Mlp {
            fc1,
            act,
            fc2,
            drop,
        }
    }
}

#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1: nn::Linear<B>,
    act: nn::Gelu,
    fc2: nn::Linear<B>,
    drop: nn::Dropout,
}

impl<B: Backend> Mlp<B> {
    pub fn forward(&self, x: Tensor<B, 3, Float>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = self.act.forward(x);
        let x = self.drop.forward(x);
        let x = self.fc2.forward(x);
        let x = self.drop.forward(x);
        x
    }
}

fn window_partition<B: Backend>(x: Tensor<B, 4, Float>, window_size: usize) -> Tensor<B, 4, Float> {
    let [b, h, w, c] = x.dims();
    let x = x.reshape([
        b,
        h / window_size,
        window_size,
        w / window_size,
        window_size,
        c,
    ]);
    let x = x.permute([0, 1, 3, 2, 4, 5]).reshape([
        -1,
        window_size as i32,
        window_size as i32,
        c as i32,
    ]);
    x
}

fn window_reverse<B: Backend>(
    windows: Tensor<B, 4, Float>,
    window_size: usize,
    h: usize,
    w: usize,
) -> Tensor<B, 4, Float> {
    let b = windows.shape().dims[0] / (h * w / window_size / window_size);
    let x = windows.reshape([
        b as i32,
        (h / window_size) as i32,
        (w / window_size) as i32,
        window_size as i32,
        window_size as i32,
        -1,
    ]);
    let x = x
        .permute([0, 1, 3, 2, 4, 5])
        .reshape([b as i32, h as i32, w as i32, -1]);
    x
}

#[derive(Config)]
pub struct WindowAttentionConfig {
    dim: usize,
    window_size: [usize; 2],
    num_heads: usize,
    #[config(default = "true")]
    qkv_bias: bool,
    #[config(default = "None")]
    qk_scale: Option<f64>,
    #[config(default = "0.0")]
    attn_drop: f64,
    #[config(default = "0.0")]
    proj_drop: f64,
}

impl WindowAttentionConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> WindowAttention<B> {
        let head_dim = self.dim / self.num_heads;
        let relative_position_bias_table = Param::from_tensor(Tensor::zeros(
            [
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                self.num_heads,
            ],
            device,
        ));

        let coords: Tensor<B, 3, _> =
            Tensor::<B, 2, Int>::cartesian_grid([self.window_size[0], self.window_size[1]], device);
        let coords_flatten: Tensor<B, 2, _> = coords.flatten(1, 2);
        let relative_coords: Tensor<B, 3, _> =
            coords_flatten.clone().unsqueeze_dim(2) - coords_flatten.unsqueeze_dim(1);
        let relative_coords = relative_coords.permute([1, 2, 0]);
        let [b, d1, d2] = relative_coords.dims();
        let relative_coords_: Tensor<B, 3, _> = relative_coords
            .clone()
            .slice([0..b, 0..d1, 0..1])
            .squeeze_dims::<1>(&[0, 1])
            .add_scalar((self.window_size[0] - 1) as u32)
            .unsqueeze_dims(&[0, 1]);
        let relative_coords = relative_coords.slice_assign([0..b, 0..d1, 0..1], relative_coords_);
        let relative_coords_: Tensor<B, 3, _> = relative_coords
            .clone()
            .slice([0..b, 0..d1, 1..2])
            .squeeze_dims::<1>(&[0, 1])
            .add_scalar((self.window_size[1] - 1) as u32)
            .unsqueeze_dims(&[0, 1]);
        let relative_coords = relative_coords.slice_assign([0..b, 0..d1, 1..2], relative_coords_);
        let relative_coords_: Tensor<B, 3, _> = relative_coords
            .clone()
            .slice([0..b, 0..d1, 0..1])
            .squeeze_dims::<1>(&[0, 1])
            .mul_scalar((2 * self.window_size[1] - 1) as u32)
            .unsqueeze_dims(&[0, 1]);
        let relative_coords = relative_coords.slice_assign([0..b, 0..d1, 0..1], relative_coords_);
        let relative_position_index = relative_coords.sum();
        let relative_position_index = Param::from_tensor(relative_position_index.float());

        let qkv = nn::LinearConfig::new(self.dim, self.dim * 3)
            .with_bias(self.qkv_bias)
            .init(device);
        let attn_drop_prob = self.attn_drop;
        let attn_drop = nn::DropoutConfig::new(attn_drop_prob).init();
        let proj = nn::LinearConfig::new(self.dim, self.dim).init(device);
        let proj_drop = nn::DropoutConfig::new(self.proj_drop).init();

        WindowAttention {
            dim: self.dim,
            window_size: self.window_size,
            num_heads: self.num_heads,
            scale: self.qk_scale.unwrap_or((head_dim as f64).powf(-0.5)),
            relative_position_bias_table,
            relative_position_index,
            qkv,
            attn_drop,
            proj,
            proj_drop,
        }
    }
}

#[derive(Module, Debug)]
pub struct WindowAttention<B: Backend> {
    dim: usize,
    window_size: [usize; 2],
    num_heads: usize,
    scale: f64,
    relative_position_bias_table: Param<Tensor<B, 2, Float>>,
    relative_position_index: Param<Tensor<B, 1, Float>>,
    qkv: nn::Linear<B>,
    attn_drop: nn::Dropout,
    proj: nn::Linear<B>,
    proj_drop: nn::Dropout,
}

impl<B: Backend> WindowAttention<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3, Float>,
        mask: Option<Tensor<B, 3, Float>>,
    ) -> Tensor<B, 4> {
        todo!();

        let [b, n, c] = x.dims();
        let qkv = self
            .qkv
            .forward(x)
            .reshape([b, n, 3, self.num_heads, c / self.num_heads]);
        let qkv = qkv.permute([2, 0, 3, 1, 4]);
        let [d1, d2, d3, d4m, d5] = qkv.dims();
        let q = qkv.clone().slice([0..1, 0..d2, 0..1, 0..d4m, 0..d5]);
        let k = qkv.clone().slice([1..2, 0..d2, 1..2, 0..d4m, 0..d5]);
        let v = qkv.clone().slice([2..3, 0..d2, 2..3, 0..d4m, 0..d5]);

        let q = q * self.scale;

        let mut attn = q.matmul(k.swap_dims(4, 5));
    }
}

#[derive(Config)]
pub struct SwinTransformerBlockConfig {
    dim: usize,
    num_heads: usize,
    #[config(default = "7")]
    window_size: usize,
    #[config(default = "0")]
    shift_size: usize,
    #[config(default = "4.0")]
    mlp_ratio: f64,
    #[config(default = "true")]
    qkv_bias: bool,
    #[config(default = "None")]
    proj_drop: Option<f64>,
    #[config(default = "0.0")]
    drop: f64,
    #[config(default = "0.0")]
    attn_drop: f64,
    #[config(default = "0.0")]
    drop_path: f64,
    // act_layer
    // norm_layer
}

impl SwinTransformerBlockConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> SwinTransformerBlock<B> {
        todo!()
    }
}

#[derive(Module, Debug)]
pub struct SwinTransformerBlock<B: Backend> {
    norm1: nn::LayerNorm<B>,
    window_size: usize,
    shift_size: usize,
    attn: WindowAttention<B>,
    norm2: nn::LayerNorm<B>,
    mlp: Mlp<B>,
    drop_path: DropPath<B>,
}

impl<B: Backend> SwinTransformerBlock<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3, Float>,
        h: usize,
        w: usize,
        mask_matrix: Tensor<B, 3, Float>,
    ) -> Tensor<B, 3> {
        todo!()
    }
}

#[derive(Config)]
pub struct PatchMergingConfig {
    dim: usize,
    // norm: nn::LayerNormConfig,
}

impl PatchMergingConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> PatchMerging<B> {
        PatchMerging {
            norm: nn::LayerNormConfig::new(4 * self.dim).init(device),
            reduction: nn::LinearConfig::new(4 * self.dim, 2 * self.dim)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct PatchMerging<B: Backend> {
    norm: nn::LayerNorm<B>,
    reduction: nn::Linear<B>,
}

impl<B: Backend> PatchMerging<B> {
    pub fn forward(&self, x: Tensor<B, 3, Float>, h: usize, w: usize) -> Tensor<B, 3> {
        todo!()
    }
}

#[derive(Config)]
pub struct BasicLayerConfig {
    dim: usize,
    depth: usize,
    num_heads: usize,
    #[config(default = "7")]
    window_size: usize,
    #[config(default = "4.0")]
    mlp_ratio: f64,
    #[config(default = "true")]
    qkv_bias: bool,
    #[config(default = "None")]
    qk_scale: Option<f64>,
    #[config(default = "0.0")]
    drop: f64,
    #[config(default = "0.0")]
    attn_drop: f64,
    #[config(default = "0.0")]
    drop_path: f64,
    norm_layer: nn::LayerNormConfig,
    #[config(default = "None")]
    downsample: Option<PatchMergingConfig>,
    // use_checkpoint: bool,
}

impl BasicLayerConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> BasicLayer<B> {
        todo!()
    }
}

#[derive(Module, Debug)]
pub struct BasicLayer<B: Backend> {
    window_size: usize,
    shift_size: usize,
    blocks: Vec<SwinTransformerBlock<B>>,
    // TODO: Checkpoint
    // #[config(default = "false")]
    // use_checkpoint: bool,
    downsample: Option<PatchMerging<B>>,
}

impl<B: Backend> BasicLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3, Float>) -> Tensor<B, 3> {
        todo!()
    }
}

#[derive(Config)]
pub struct PatchEmbedConfig {
    patch_size: [usize; 2],
    in_channels: usize,
    embed_dim: usize,
    // norm_layer
}

impl PatchEmbedConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> PatchEmbed<B> {
        todo!()
    }
}

#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    patch_size: [usize; 2],
    proj: nn::conv::Conv2d<B>,
    norm: Option<nn::LayerNorm<B>>,
    embed_dim: usize,
}

impl<B: Backend> PatchEmbed<B> {
    pub fn forward(&self, x: Tensor<B, 4, Float>) -> Tensor<B, 4, Float> {
        todo!()
    }
}

#[derive(Config)]
pub struct SwinTransformerConfig {
    #[config(default = "224")]
    pretrain_img_size: usize,
    #[config(default = "[4, 4]")]
    patch_size: [usize; 2],
    #[config(default = "3")]
    in_channels: usize,
    #[config(default = "96")]
    embed_dim: usize,
    #[config(default = "[2, 2, 6, 2]")]
    depths: [usize; 4],
    #[config(default = "[3, 6, 12, 24]")]
    num_heads: [usize; 4],
    #[config(default = "7")]
    window_size: usize,
    #[config(default = "4.0")]
    mlp_ratio: f64,
    #[config(default = "true")]
    qkv_bias: bool,
    #[config(default = "None")]
    qk_scale: Option<f64>,
    #[config(default = "0.0")]
    drop_rate: f64,
    #[config(default = "0.0")]
    attn_drop_rate: f64,
    #[config(default = "0.2")]
    drop_path_rate: f64,
    // norm_layer: f64,
    #[config(default = "false")]
    ape: bool,
    #[config(default = "false")]
    patch_norm: bool,
    #[config(default = "[0, 1, 2, 3]")]
    out_indices: [usize; 4],
    #[config(default = "-1")]
    frozen_stages: i32,
    // TODO: Checkpoint
    // #[config(default = "false")]
    // use_checkpoint: bool,
}

impl SwinTransformerConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> SwinTransformer<B> {
        todo!()
    }
}

#[derive(Module, Debug)]
pub struct SwinTransformer<B: Backend> {
    patch_embed: PatchEmbed<B>,
    ape: bool,
    absolute_pos_embed: Param<Tensor<B, 4, Float>>,
    pos_drop: nn::Dropout,
    num_layers: usize,
    layers: Vec<BasicLayer<B>>,
    out_indices: Vec<nn::LayerNorm<B>>,
}

impl<B: Backend> SwinTransformer<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 4, Float>,
    ) -> (
        Tensor<B, 4, Float>,
        Tensor<B, 4, Float>,
        Tensor<B, 4, Float>,
        Tensor<B, 4, Float>,
    ) {
        todo!()
    }
}
