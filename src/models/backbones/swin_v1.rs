use std::ops::Deref;

use burn::{
    module::Param,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig,
    },
    prelude::*,
    tensor::{
        activation::softmax,
        module::interpolate,
        ops::{InterpolateMode, InterpolateOptions},
    },
};

use crate::special::{roll, trunc_normal, DropPath, DropPathConfig};

#[derive(Config, Debug)]
pub struct MlpConfig {
    in_features: usize,
    #[config(default = "None")]
    hidden_features: Option<usize>,
    #[config(default = "None")]
    out_features: Option<usize>,
    // act_layer:Module,
    #[config(default = "0.0")]
    drop: f64,
}

impl MlpConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> Mlp<B> {
        let out_features = self.out_features.unwrap_or(self.in_features);
        let hidden_features = self.hidden_features.unwrap_or(self.in_features);
        let fc1 = LinearConfig::new(self.in_features, hidden_features).init(device);
        let act = Gelu::new();
        let fc2 = LinearConfig::new(hidden_features, out_features).init(device);
        let drop = DropoutConfig::new(self.drop).init();

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
    fc1: Linear<B>,
    act: Gelu,
    fc2: Linear<B>,
    drop: Dropout,
}

impl<B: Backend> Mlp<B> {
    pub fn forward(&self, x: Tensor<B, 3, Float>) -> Tensor<B, 3, Float> {
        let x = self.fc1.forward(x);
        let x = self.act.forward(x);
        let x = self.drop.forward(x);
        let x = self.fc2.forward(x);

        self.drop.forward(x)
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

    x.permute([0, 1, 3, 2, 4, 5])
        .reshape([-1, window_size as i32, window_size as i32, c as i32])
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

    x.permute([0, 1, 3, 2, 4, 5])
        .reshape([b as i32, h as i32, w as i32, -1])
}

#[derive(Config, Debug)]
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
        let relative_position_bias_table = Tensor::zeros(
            [
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1),
                self.num_heads,
            ],
            device,
        );

        let coords: Tensor<B, 3, _> =
            Tensor::<B, 2, Int>::cartesian_grid([self.window_size[0], self.window_size[1]], device)
                .permute([2, 0, 1])
                .float();
        let coords_flatten: Tensor<B, 2> = coords.flatten(1, 2);
        let relative_coords =
            coords_flatten.clone().unsqueeze_dim(2) - coords_flatten.unsqueeze_dim(1);
        let relative_coords = relative_coords.permute([1, 2, 0]);
        let [b, d1, _] = relative_coords.dims();
        let relative_coords = relative_coords.clone().slice_assign(
            [0..b, 0..d1, 0..1],
            relative_coords.slice([0..b, 0..d1, 0..1]) + (self.window_size[1] - 1) as f64,
        );
        let relative_coords = relative_coords.clone().slice_assign(
            [0..b, 0..d1, 1..2],
            relative_coords.slice([0..b, 0..d1, 1..2]) + (self.window_size[1] - 2) as f64,
        );
        let relative_coords = relative_coords.clone().slice_assign(
            [0..b, 0..d1, 0..1],
            relative_coords.slice([0..b, 0..d1, 0..1]) * (2.0 * (self.window_size[1] as f64) - 1.),
        );
        let relative_position_index = relative_coords.sum_dim(2).squeeze_dims(&[-1]);
        let relative_position_index = Param::from_tensor(relative_position_index);

        let qkv = LinearConfig::new(self.dim, self.dim * 3)
            .with_bias(self.qkv_bias)
            .init(device);
        let attn_drop_prob = self.attn_drop;
        let attn_drop = DropoutConfig::new(attn_drop_prob).init();
        let proj = LinearConfig::new(self.dim, self.dim).init(device);
        let proj_drop = DropoutConfig::new(self.proj_drop).init();

        let relative_position_bias_table = Param::from_tensor(trunc_normal(
            relative_position_bias_table,
            0.02,
            1.0,
            -2.0,
            2.0,
        ));

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
    relative_position_bias_table: Param<Tensor<B, 2>>,
    relative_position_index: Param<Tensor<B, 2>>,
    qkv: Linear<B>,
    attn_drop: Dropout,
    proj: Linear<B>,
    proj_drop: Dropout,
}

impl<B: Backend> WindowAttention<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3, Float>,
        mask: Option<Tensor<B, 3, Float>>,
    ) -> Tensor<B, 3, Float> {
        let [b, n, c] = x.dims();
        let qkv = self
            .qkv
            .forward(x)
            .reshape([b, n, 3, self.num_heads, c / self.num_heads])
            .permute([2, 0, 3, 1, 4]);
        let [_, d2, d3, d4m, d5] = qkv.dims();
        let q: Tensor<B, 4, Float> = qkv
            .clone()
            .slice([0..1, 0..d2, 0..d3, 0..d4m, 0..d5])
            .squeeze(0);
        let k: Tensor<B, 4, Float> = qkv
            .clone()
            .slice([1..2, 0..d2, 0..d3, 0..d4m, 0..d5])
            .squeeze(0);
        let v: Tensor<B, 4, Float> = qkv
            .clone()
            .slice([2..3, 0..d2, 0..d3, 0..d4m, 0..d5])
            .squeeze(0);

        let q = q * self.scale;

        let attn = q.matmul(k.swap_dims(2, 3));
        let relative_position_bias = self
            .relative_position_bias_table
            .deref()
            .clone()
            .select(0, self.relative_position_index.val().reshape([-1]).int())
            .reshape([
                (self.window_size[0] * self.window_size[1]) as i32,
                (self.window_size[0] * self.window_size[1]) as i32,
                -1,
            ])
            .permute([2, 0, 1]);
        let attn = attn + relative_position_bias.unsqueeze();

        let attn = match mask {
            Some(mask) => {
                let [nw, _, _] = mask.dims();
                let attn = attn.reshape([b / nw, nw, self.num_heads, n, n])
                    + mask.unsqueeze_dim::<4>(1).unsqueeze();
                attn.reshape([-1, self.num_heads as i32, n as i32, n as i32])
            }
            None => attn,
        };

        let attn = softmax(attn, 3);

        let attn = self.attn_drop.forward(attn);
        let x = attn.matmul(v).swap_dims(1, 2).reshape([b, n, c]);

        let x = self.proj.forward(x);

        self.proj_drop.forward(x)
    }
}

#[derive(Config, Debug)]
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
    qk_scale: Option<f64>,
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
        let mlp_hidden_dim = ((self.dim as f64) * self.mlp_ratio) as usize;
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

#[derive(Module, Debug)]
pub struct SwinTransformerBlock<B: Backend> {
    window_size: usize,
    shift_size: usize,
    norm1: LayerNorm<B>,
    attn: WindowAttention<B>,
    norm2: LayerNorm<B>,
    mlp: Mlp<B>,
    drop_path: DropPath,
}

impl<B: Backend> SwinTransformerBlock<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3, Float>,
        h: usize,
        w: usize,
        mask_matrix: Tensor<B, 3, Float>,
    ) -> Tensor<B, 3, Float> {
        let [b, l, c] = x.dims();

        let shortcut = x.clone();
        let x = self.norm1.forward(x);
        let x = x.reshape([b, h, w, c]);

        let pad_l = 0;
        let pad_t = 0;
        let pad_r = (self.window_size - w % self.window_size) % self.window_size;
        let pad_b = (self.window_size - h % self.window_size) % self.window_size;
        let x = x
            .permute([0, 3, 1, 2])
            .pad((pad_l, pad_r, pad_t, pad_b), B::FloatElem::from_elem(0.0))
            .permute([0, 2, 3, 1]);
        let [_, hp, wp, _] = x.dims();

        let (shifted_x, attn_mask) = if self.shift_size > 0 {
            let shifted_x = roll(
                x,
                vec![-(self.shift_size as i64), -(self.shift_size as i64)],
                vec![1, 2],
            );
            let attn_mask = mask_matrix;
            (shifted_x, Some(attn_mask))
        } else {
            (x, None)
        };

        let x_window = window_partition(shifted_x, self.window_size);
        let x_window =
            x_window.reshape([-1, (self.window_size * self.window_size) as i32, c as i32]);

        let attn_window = self.attn.forward(x_window, attn_mask);

        let attn_window = attn_window.reshape([
            -1,
            self.window_size as i32,
            self.window_size as i32,
            c as i32,
        ]);
        let shifted_x = window_reverse(attn_window, self.window_size, hp, wp);

        let x = if self.shift_size > 0 {
            roll(
                shifted_x,
                vec![self.shift_size as i64, self.shift_size as i64],
                vec![1, 2],
            )
        } else {
            shifted_x
        };

        let x = if pad_r > 0 || pad_b > 0 {
            let [d1, _, _, d4] = x.dims();
            x.slice([0..d1, 0..h, 0..w, 0..d4])
        } else {
            x
        };

        let x = x.reshape([b, h * w, c]);

        let x = shortcut + self.drop_path.forward(x);

        self.drop_path
            .forward(self.mlp.forward(self.norm2.forward(x)))
    }
}

#[derive(Config, Debug)]
pub struct PatchMergingConfig {
    dim: usize,
    // norm: nn::LayerNormConfig,
}

impl PatchMergingConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> PatchMerging<B> {
        PatchMerging {
            norm: LayerNormConfig::new(4 * self.dim).init(device),
            reduction: LinearConfig::new(4 * self.dim, 2 * self.dim)
                .with_bias(false)
                .init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct PatchMerging<B: Backend> {
    norm: LayerNorm<B>,
    reduction: Linear<B>,
}

impl<B: Backend> PatchMerging<B> {
    pub fn forward(&self, x: Tensor<B, 3, Float>, h: usize, w: usize) -> Tensor<B, 3, Float> {
        let [b, l, c] = x.dims();

        let x = x.reshape([b, h, w, c]);

        let pad_input = h % 2 == 1 || w % 2 == 1;

        let x = if pad_input {
            x.pad((0, h % 2, 0, w % 2), B::FloatElem::from_elem(0.0))
        } else {
            x
        };

        let device = x.device();
        let top_idx = Tensor::arange_step(0..h as i64, 2, &device);
        let bottom_idx = Tensor::arange_step(1..h as i64, 2, &device);
        let left_idx = Tensor::arange_step(0..w as i64, 2, &device);
        let right_idx = Tensor::arange_step(1..w as i64, 2, &device);

        let x0 = x
            .clone()
            .select(1, top_idx.clone())
            .select(2, left_idx.clone());
        let x1 = x.clone().select(1, top_idx).select(2, right_idx.clone());
        let x2 = x.clone().select(1, bottom_idx.clone()).select(2, left_idx);
        let x3 = x.select(1, bottom_idx).select(2, right_idx);

        let x = Tensor::cat(vec![x0, x1, x2, x3], 3);
        let x = x.reshape([b as i32, -1, (4 * c) as i32]);

        let x = self.norm.forward(x);

        self.reduction.forward(x)
    }
}

#[derive(Config, Debug)]
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
    #[config(default = "Vec::new()")]
    drop_path: Vec<f64>,
    #[config(default = "false")]
    downsample: bool,
    // use_checkpoint: bool,
}

impl BasicLayerConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> BasicLayer<B> {
        let blocks = (0..self.depth)
            .map(|i| {
                SwinTransformerBlockConfig::new(self.dim, self.num_heads)
                    .with_window_size(self.window_size)
                    .with_shift_size(if i % 2 == 0 { 0 } else { self.window_size / 2 })
                    .with_mlp_ratio(self.mlp_ratio)
                    .with_qkv_bias(self.qkv_bias)
                    .with_qk_scale(self.qk_scale)
                    .with_drop(self.drop)
                    .with_attn_drop(self.attn_drop)
                    .with_drop_path(self.drop_path[i])
                    .init(device)
            })
            .collect::<Vec<_>>();
        let downsample = if self.downsample {
            Some(PatchMergingConfig::new(self.dim).init(device))
        } else {
            None
        };

        BasicLayer {
            window_size: self.window_size,
            shift_size: self.window_size / 2,
            blocks,
            downsample,
        }
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
    pub fn forward(
        &self,
        x: Tensor<B, 3, Float>,
        h: usize,
        w: usize,
    ) -> (
        Tensor<B, 3, Float>,
        usize,
        usize,
        Tensor<B, 3, Float>,
        usize,
        usize,
    ) {
        let hp = ((h as f64) / self.window_size as f64).ceil() as usize * self.window_size;
        let wp = ((w as f64) / self.window_size as f64).ceil() as usize * self.window_size;
        let mut img_mask: Tensor<B, 4, Float> = Tensor::zeros([1, hp, wp, 1], &x.device());
        let h_slices = [
            (0, -(self.window_size as isize)),
            (-(self.window_size as isize), -(self.shift_size as isize)),
            (-(self.shift_size as isize), 0),
        ];
        let w_slices = [
            (0, -(self.window_size as isize)),
            (-(self.window_size as isize), -(self.shift_size as isize)),
            (-(self.shift_size as isize), 0),
        ];
        let mut cnt = 0;
        for (hs, he) in h_slices {
            for (ws, we) in w_slices {
                let [d1, d2, d3, d4] = img_mask.dims();
                img_mask = img_mask.slice_assign(
                    [
                        0..d1,
                        if hs > 0 {
                            (d2 as isize + hs) as usize
                        } else {
                            0
                        }..((d2 as isize + he) as usize),
                        if ws > 0 {
                            (d3 as isize + ws) as usize
                        } else {
                            0
                        }..((d3 as isize + we) as usize),
                        0..d4,
                    ],
                    Tensor::zeros(
                        [
                            d1,
                            (if hs > 0 {
                                (d2 as isize + hs) as usize
                            } else {
                                0
                            }..((d2 as isize + he) as usize))
                                .end,
                            (if ws > 0 {
                                (d3 as isize + ws) as usize
                            } else {
                                0
                            }..((d3 as isize + we) as usize))
                                .end,
                            d4,
                        ],
                        &x.device(),
                    )
                    .add_scalar(cnt),
                );
                cnt += 1;
            }
        }

        let mask_windows = window_partition(img_mask, self.window_size);
        let mask_windows = mask_windows.reshape([-1, (self.window_size * self.window_size) as i32]);
        let attn_mask: Tensor<B, 3, Float> =
            mask_windows.clone().unsqueeze_dim(1) - mask_windows.unsqueeze_dim(2);
        let attn_mask = attn_mask
            .clone()
            .mask_fill(attn_mask.clone().not_equal_elem(0.0), -100.0)
            .mask_fill(attn_mask.equal_elem(0.0), 0.0);

        let mut x = x;
        for blk in &self.blocks {
            x = blk.forward(x, h, w, attn_mask.clone());
        }
        match &self.downsample {
            Some(downsample) => {
                let x_down = downsample.forward(x.clone(), h, w);
                let wh = (h + 1) / 2;
                let ww = (w + 1) / 2;
                (x, h, w, x_down, wh, ww)
            }
            None => (x.clone(), h, w, x, h, w),
        }
    }
}

#[derive(Config, Debug)]
pub struct PatchEmbedConfig {
    #[config(default = "4")]
    patch_size: usize,
    #[config(default = "3")]
    in_channels: usize,
    #[config(default = "96")]
    embed_dim: usize,
    #[config(default = "false")]
    norm_layer: bool,
}

impl PatchEmbedConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> PatchEmbed<B> {
        let proj = Conv2dConfig::new(
            [self.in_channels, self.embed_dim],
            [self.patch_size, self.patch_size],
        )
        .with_stride([self.patch_size, self.patch_size])
        .init(device);
        let norm = if self.norm_layer {
            Some(LayerNormConfig::new(self.embed_dim).init(device))
        } else {
            None
        };

        PatchEmbed {
            embed_dim: self.embed_dim,
            patch_size: self.patch_size,
            proj,
            norm,
        }
    }
}

#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    embed_dim: usize,
    patch_size: usize,
    proj: Conv2d<B>,
    norm: Option<LayerNorm<B>>,
}

impl<B: Backend> PatchEmbed<B> {
    pub fn forward(&self, x: Tensor<B, 4, Float>) -> Tensor<B, 4, Float> {
        let [_, _, h, w] = x.dims();
        let mut x = x;
        if w % self.patch_size != 0 {
            x = x.pad(
                (0, self.patch_size - (w % self.patch_size), 0, 0),
                B::FloatElem::from_elem(0.0),
            );
        };
        if h % self.patch_size != 0 {
            x = x.pad(
                (0, 0, 0, self.patch_size - (h % self.patch_size)),
                B::FloatElem::from_elem(0.0),
            );
        };
        x = self.proj.forward(x);
        if let Some(norm) = &self.norm {
            let [_, _, wh, ww] = x.dims();
            x = x.flatten(2, 3).swap_dims(1, 2);
            x = norm.forward(x);
            x = x
                .swap_dims(1, 2)
                .reshape([-1, self.embed_dim as i32, wh as i32, ww as i32]);
        };

        x
    }
}

#[derive(Config, Debug)]
pub struct SwinTransformerConfig {
    #[config(default = "224")]
    pretrain_img_size: usize,
    #[config(default = "4")]
    patch_size: usize,
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

fn linspace(start: f64, end: f64, steps: usize) -> Vec<f64> {
    if steps == 0 {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(steps);
    let step_size = (end - start) / (steps as f64 - 1.0);

    for i in 0..steps {
        result.push(start + i as f64 * step_size);
    }

    result
}

impl SwinTransformerConfig {
    pub fn init<B: Backend>(&self, device: &Device<B>) -> SwinTransformer<B> {
        let num_layers = self.depths.len();

        let patch_embed = PatchEmbedConfig::new()
            .with_patch_size(self.patch_size)
            .with_in_channels(self.in_channels)
            .with_embed_dim(self.embed_dim)
            .with_norm_layer(self.patch_norm)
            .init(device);

        let absolute_pos_embed = if self.ape {
            let patches_resolution = [
                self.pretrain_img_size / self.patch_size,
                self.pretrain_img_size / self.patch_size,
            ];
            let absolute_pos_embed = Tensor::<B, 4>::zeros(
                [
                    1,
                    self.embed_dim,
                    patches_resolution[0],
                    patches_resolution[1],
                ],
                device,
            );
            Some(Param::from_tensor(trunc_normal(
                absolute_pos_embed,
                0.0,
                0.02,
                -2.0,
                2.0,
            )))
        } else {
            None
        };

        let dpr = linspace(0.0, self.drop_path_rate, self.depths.iter().sum());

        let mut layers = Vec::new();
        for i_layer in 0..num_layers {
            let start: usize = self.depths[..i_layer].iter().sum();
            let end: usize = self.depths[..i_layer + 1].iter().sum();
            let layer = BasicLayerConfig::new(
                ((self.embed_dim as i32) * 2_i32.pow(i_layer as u32)) as usize,
                self.depths[i_layer],
                self.num_heads[i_layer],
            )
            .with_window_size(self.window_size)
            .with_mlp_ratio(self.mlp_ratio)
            .with_qkv_bias(self.qkv_bias)
            .with_qk_scale(self.qk_scale)
            .with_drop(self.drop_rate)
            .with_attn_drop(self.attn_drop_rate)
            .with_drop_path(dpr[start..end].to_owned())
            .with_downsample(i_layer < num_layers - 1)
            .init(device);
            layers.push(layer);
        }

        let num_features = (0..num_layers)
            .map(|i| ((self.embed_dim as i32) * 2_i32.pow(i as u32)) as usize)
            .collect::<Vec<_>>();

        let mut norm_layers = Vec::new();
        for i_layer in self.out_indices {
            let layer: LayerNorm<B> = LayerNormConfig::new(num_features[i_layer]).init(device);
            norm_layers.push(layer);
        }

        SwinTransformer {
            patch_embed,
            pos_drop: DropoutConfig::new(self.drop_rate).init(),
            num_layers,
            layers,
            num_features,
            norm_layers,
            out_indices: self.out_indices,
            absolute_pos_embed,
        }
    }
}

#[derive(Module, Debug)]
pub struct SwinTransformer<B: Backend> {
    patch_embed: PatchEmbed<B>,
    absolute_pos_embed: Option<Param<Tensor<B, 4, Float>>>,
    pos_drop: Dropout,
    num_layers: usize,
    layers: Vec<BasicLayer<B>>,
    norm_layers: Vec<LayerNorm<B>>,
    out_indices: [usize; 4],
    num_features: Vec<usize>,
}

impl<B: Backend> SwinTransformer<B> {
    pub fn forward(&self, x: Tensor<B, 4, Float>) -> [Tensor<B, 4, Float>; 4] {
        let mut x = self.patch_embed.forward(x);

        let [_, _, wh, ww] = x.dims();
        if let Some(absolute_pos_embed) = self.absolute_pos_embed.clone() {
            let absolute_pos_embed = interpolate(
                absolute_pos_embed.deref().to_owned(),
                [wh, ww],
                InterpolateOptions::new(InterpolateMode::Bicubic),
            );

            x = x + absolute_pos_embed;
        };

        let mut outs = Vec::new();
        let [_, _, h, w] = x.dims();
        let x: Tensor<B, 3, Float> = x.flatten(2, 3).swap_dims(1, 2);
        let x = self.pos_drop.forward(x);
        let mut x = x;
        let mut wh = wh;
        let mut ww = ww;
        for i in 0..self.num_layers {
            let (x_out, h, w, x_, wh_, ww_) = self.layers[i].forward(x.clone(), wh, ww);
            x = x_;
            wh = wh_;
            ww = ww_;
            if self.out_indices.contains(&i) {
                let x_out = self.norm_layers[i].forward(x_out.clone());
                let out = x_out
                    .reshape([-1, h as i32, w as i32, self.num_features[i] as i32])
                    .permute([0, 3, 1, 2]);
                outs.push(out);
            }
        }
        outs.try_into().unwrap()
    }
}

pub fn swin_v1_t<B: Backend>(device: &Device<B>) -> SwinTransformer<B> {
    SwinTransformerConfig::new()
        .with_embed_dim(96)
        .with_depths([2, 2, 6, 2])
        .with_num_heads([3, 6, 12, 24])
        .with_window_size(7)
        .init(device)
}

pub fn swin_v1_s<B: Backend>(device: &Device<B>) -> SwinTransformer<B> {
    SwinTransformerConfig::new()
        .with_embed_dim(96)
        .with_depths([2, 2, 18, 2])
        .with_num_heads([3, 6, 12, 24])
        .with_window_size(7)
        .init(device)
}

pub fn swin_v1_b<B: Backend>(device: &Device<B>) -> SwinTransformer<B> {
    SwinTransformerConfig::new()
        .with_embed_dim(128)
        .with_depths([2, 2, 18, 2])
        .with_num_heads([4, 8, 16, 32])
        .with_window_size(12)
        .init(device)
}

pub fn swin_v1_l<B: Backend>(device: &Device<B>) -> SwinTransformer<B> {
    SwinTransformerConfig::new()
        .with_embed_dim(192)
        .with_depths([2, 2, 18, 2])
        .with_num_heads([6, 12, 24, 48])
        .with_window_size(12)
        .init(device)
}
