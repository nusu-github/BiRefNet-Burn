//! Stage (`BasicLayer`) module for Swin Transformer.

use std::ops::Range;

use burn::prelude::*;

use crate::backbones::{
    swin::{
        block::{SwinTransformerBlock, SwinTransformerBlockConfig, window_partition},
        patch::{PatchMerging, PatchMergingConfig},
    },
    usize_to_f64,
};

/// Configuration for [`BasicLayer`].
#[derive(Config, Debug, PartialEq)]
pub struct BasicLayerConfig {
    dim: usize,
    depth: usize,
    num_heads: usize,
    #[config(default = 7)]
    window_size: usize,
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
    #[config(default = "Vec::new()")]
    drop_path: Vec<f64>,
    #[config(default = false)]
    downsample: bool,
}

impl BasicLayerConfig {
    /// Initializes a [`BasicLayer`] stage.
    #[must_use]
    pub fn init(&self, device: &Device) -> BasicLayer {
        let blocks = (0..self.depth)
            .map(|i| {
                let drop_path = self.drop_path.get(i).copied().unwrap_or(0.0);
                SwinTransformerBlockConfig::new(self.dim, self.num_heads)
                    .with_window_size(self.window_size)
                    .with_shift_size(if i % 2 == 0 { 0 } else { self.window_size / 2 })
                    .with_mlp_ratio(self.mlp_ratio)
                    .with_qkv_bias(self.qkv_bias)
                    .with_qk_scale(self.qk_scale)
                    .with_drop(self.drop)
                    .with_attn_drop(self.attn_drop)
                    .with_drop_path(drop_path)
                    .init(device)
            })
            .collect();
        let downsample = self
            .downsample
            .then(|| PatchMergingConfig::new(self.dim).init(device));

        BasicLayer {
            window_size: self.window_size,
            shift_size: self.window_size / 2,
            blocks,
            downsample,
        }
    }
}

/// One Swin Transformer stage with alternating regular/shifted window attention and optional downsampling.
#[derive(Module, Debug)]
pub struct BasicLayer {
    window_size: usize,
    shift_size: usize,
    blocks: Vec<SwinTransformerBlock>,
    downsample: Option<PatchMerging>,
}

impl BasicLayer {
    /// Runs all blocks in this stage and optional downsampling.
    #[must_use]
    pub fn forward(
        &self,
        x: Tensor<3>,
        h: usize,
        w: usize,
    ) -> (Tensor<3>, usize, usize, Tensor<3>, usize, usize) {
        let device = x.device();
        let attn_mask = self.compute_attn_mask(h, w, &device);

        let mut x = x;
        for blk in &self.blocks {
            x = blk.forward(x, h, w, attn_mask.clone());
        }

        match &self.downsample {
            Some(downsample) => {
                let x_down = downsample.forward(x.clone(), h, w);
                (x, h, w, x_down, h.div_ceil(2), w.div_ceil(2))
            }
            None => (x.clone(), h, w, x, h, w),
        }
    }

    fn compute_attn_mask(&self, h: usize, w: usize, device: &Device) -> Tensor<3> {
        let hp = h.div_ceil(self.window_size) * self.window_size;
        let wp = w.div_ceil(self.window_size) * self.window_size;
        let mut img_mask: Tensor<4> = Tensor::zeros([1, hp, wp, 1], device);

        let h_ranges = mask_ranges(hp, self.window_size, self.shift_size);
        let w_ranges = mask_ranges(wp, self.window_size, self.shift_size);

        let mut cnt = 0_usize;
        for h_range in &h_ranges {
            for w_range in &w_ranges {
                img_mask = img_mask.slice_fill(
                    s![.., h_range.clone(), w_range.clone(), ..],
                    usize_to_f64(cnt),
                );
                cnt += 1;
            }
        }

        let mask_windows = window_partition(img_mask, self.window_size);
        let mask_num_windows = mask_windows.dims()[0];
        let mask_windows =
            mask_windows.reshape([mask_num_windows, self.window_size * self.window_size]);

        let attn_mask: Tensor<3> =
            mask_windows.clone().unsqueeze_dim(1) - mask_windows.unsqueeze_dim(2);
        let non_zero = attn_mask.not_equal_elem(0.0);
        non_zero.zeros_like().float().mask_fill(non_zero, -100.0)
    }
}

fn mask_ranges(size: usize, window_size: usize, shift_size: usize) -> Vec<Range<usize>> {
    let split1 = size.saturating_sub(window_size);
    let split2 = size.saturating_sub(shift_size);
    [(0, split1), (split1, split2), (split2, size)]
        .into_iter()
        .filter_map(|(start, end)| (start < end).then_some(start..end))
        .collect()
}
