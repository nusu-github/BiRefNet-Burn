//! Pyramid Vision Transformer v2 (`PVTv2`) backbone implementation.

mod attention;
mod block;
mod patch;

use birefnet_extra_ops::trunc_normal;
use burn::{
    module::Param,
    nn::{LayerNorm, LayerNormConfig, Linear},
    prelude::*,
};

use self::{
    attention::AttentionConfig,
    block::{Block, BlockConfig},
    patch::OverlapPatchEmbed,
};
use crate::{
    PvtV2Variant,
    backbones::{linspace, validate_mlp_ratio, validate_rate},
    error::BackboneError,
};

/// Linear layer initialized with `trunc_normal(std = 0.02)` weights and zero bias.
fn trunc_normal_linear(
    in_features: usize,
    out_features: usize,
    bias: bool,
    device: &Device,
) -> Linear {
    const STD: f64 = 0.02;
    let weight = trunc_normal(
        [in_features, out_features],
        0.0,
        STD,
        -2.0 * STD,
        2.0 * STD,
        device,
    );
    let bias = bias.then(|| Param::from_tensor(Tensor::zeros([out_features], device)));
    Linear {
        weight: Param::from_tensor(weight),
        bias,
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
        let out1 = run_stage(&self.patch_embed1, &self.block1, &self.norm1, x);
        let out2 = run_stage(&self.patch_embed2, &self.block2, &self.norm2, out1.clone());
        let out3 = run_stage(&self.patch_embed3, &self.block3, &self.norm3, out2.clone());
        let out4 = run_stage(&self.patch_embed4, &self.block4, &self.norm4, out3.clone());
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
) -> Tensor<4> {
    let (mut x, h, w) = patch_embed.forward(input);
    for blk in blocks {
        x = blk.forward(x, h, w);
    }
    let [batch, _, channels] = x.dims();
    norm.forward(x)
        .reshape([batch, h, w, channels])
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

/// Stage-0 embedding kernel and stride; later stages use `3x3` with stride 2.
const STEM_PATCH: (usize, usize) = (7, 4);
const STAGE_PATCH: (usize, usize) = (3, 2);

impl PvtV2Config {
    /// Returns the number of input image channels.
    #[must_use]
    pub const fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Validates hyperparameters and initializes a [`PyramidVisionTransformerImpr`].
    ///
    /// # Errors
    ///
    /// Returns [`BackboneError`] if channel dims, head counts, spatial reduction ratios, MLP
    /// ratios, or dropout rates are invalid.
    pub fn init(&self, device: &Device) -> Result<PyramidVisionTransformerImpr, BackboneError> {
        self.validate()?;

        let total_depth: usize = self.depths.iter().sum();
        let dpr = linspace(0.0, self.drop_path_rate, total_depth);
        let mut remaining = dpr.as_slice();
        let mut next_stage = |stage: usize| {
            let (stage_dpr, rest) = remaining.split_at(self.depths[stage]);
            remaining = rest;
            self.build_stage(stage, stage_dpr, device)
        };
        let (patch_embed1, block1, norm1) = next_stage(0);
        let (patch_embed2, block2, norm2) = next_stage(1);
        let (patch_embed3, block3, norm3) = next_stage(2);
        let (patch_embed4, block4, norm4) = next_stage(3);

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

    fn validate(&self) -> Result<(), BackboneError> {
        if self.in_channels == 0 {
            return Err(BackboneError::invalid_config(
                "in_channels must be positive",
            ));
        }
        for stage in 0..4 {
            let (dim, heads) = (self.embed_dims[stage], self.num_heads[stage]);
            if dim == 0 || heads == 0 || !dim.is_multiple_of(heads) {
                return Err(BackboneError::invalid_config(format!(
                    "stage {stage} embed_dim ({dim}) must be positive and divisible by num_heads ({heads})"
                )));
            }
            if self.sr_ratios[stage] == 0 {
                return Err(BackboneError::invalid_config(format!(
                    "stage {stage} sr_ratio must be positive"
                )));
            }
            validate_mlp_ratio(self.mlp_ratios[stage])?;
        }
        validate_rate("drop_rate", self.drop_rate)?;
        validate_rate("drop_path_rate", self.drop_path_rate)?;
        if !(self.epsilon.is_finite() && self.epsilon > 0.0) {
            return Err(BackboneError::invalid_config(format!(
                "epsilon must be finite and positive, got {}",
                self.epsilon
            )));
        }
        Ok(())
    }

    fn build_stage(
        &self,
        stage: usize,
        stage_dpr: &[f64],
        device: &Device,
    ) -> (OverlapPatchEmbed, Vec<Block>, LayerNorm) {
        let (patch, stride) = if stage == 0 { STEM_PATCH } else { STAGE_PATCH };
        let in_dim = if stage == 0 {
            self.in_channels
        } else {
            self.embed_dims[stage - 1]
        };
        let embed_dim = self.embed_dims[stage];
        let patch_embed =
            OverlapPatchEmbed::new(patch, stride, [in_dim, embed_dim], self.epsilon, device);
        let blocks = stage_dpr
            .iter()
            .map(|&drop_path| self.block_config(stage, drop_path).init(device))
            .collect();
        let norm = LayerNormConfig::new(embed_dim)
            .with_epsilon(self.epsilon)
            .init(device);
        (patch_embed, blocks, norm)
    }

    const fn block_config(&self, stage: usize, drop_path: f64) -> BlockConfig {
        BlockConfig {
            attention: AttentionConfig {
                dim: self.embed_dims[stage],
                num_heads: self.num_heads[stage],
                qkv_bias: self.qkv_bias,
                qk_scale: None,
                attn_drop: 0.0,
                proj_drop: self.drop_rate,
                sr_ratio: self.sr_ratios[stage],
                epsilon: self.epsilon,
            },
            mlp_ratio: self.mlp_ratios[stage],
            drop: self.drop_rate,
            drop_path,
        }
    }

    /// Returns the preset configuration for `variant` with `in_channels` input channels.
    #[must_use]
    pub const fn from_variant(variant: PvtV2Variant, in_channels: usize) -> Self {
        let (depths, mlp_ratios, embed_dims) = match variant {
            PvtV2Variant::B0 => ([2, 2, 2, 2], [8.0, 8.0, 4.0, 4.0], [32, 64, 160, 256]),
            PvtV2Variant::B1 => ([2, 2, 2, 2], [8.0, 8.0, 4.0, 4.0], [64, 128, 320, 512]),
            PvtV2Variant::B2 => ([3, 4, 6, 3], [8.0, 8.0, 4.0, 4.0], [64, 128, 320, 512]),
            PvtV2Variant::B3 => ([3, 4, 18, 3], [8.0, 8.0, 4.0, 4.0], [64, 128, 320, 512]),
            PvtV2Variant::B4 => ([3, 8, 27, 3], [8.0, 8.0, 4.0, 4.0], [64, 128, 320, 512]),
            PvtV2Variant::B5 => ([3, 6, 40, 3], [4.0; 4], [64, 128, 320, 512]),
        };
        Self {
            in_channels,
            embed_dims,
            num_heads: [1, 2, 5, 8],
            mlp_ratios,
            qkv_bias: true,
            depths,
            sr_ratios: [8, 4, 2, 1],
            drop_rate: 0.0,
            drop_path_rate: 0.1,
            epsilon: 1e-6,
        }
    }
}
