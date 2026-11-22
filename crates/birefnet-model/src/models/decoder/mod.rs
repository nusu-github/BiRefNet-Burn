//! Hierarchical multi-scale decoder for `BiRefNet`.

mod config;
mod patches;

use burn::{
    nn::{BatchNorm, BatchNormConfig, Relu, conv::Conv2d, conv::Conv2dConfig},
    prelude::*,
    tensor::activation::sigmoid,
};

pub use self::config::DecoderConfig;
use super::modules::{BasicDecBlk, BasicLatBlk, ResBlk, SimpleConvs};
use crate::{config::InterpolationStrategy, models::modules::utils::intelligent_interpolate};

/// Wrapper enum over supported decoder block implementations.
#[derive(Module, Debug)]
#[expect(
    clippy::large_enum_variant,
    reason = "Burn Module requires inline storage without Box indirection"
)]
pub enum DecoderBlockModule {
    /// Basic decoder block.
    BasicDecBlk(BasicDecBlk),
    /// Residual decoder block.
    ResBlk(ResBlk),
}

impl DecoderBlockModule {
    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        match self {
            Self::BasicDecBlk(blk) => blk.forward(input),
            Self::ResBlk(blk) => blk.forward(input),
        }
    }
}

/// Wrapper enum over supported lateral connection block implementations.
#[derive(Module, Debug)]
pub enum LateralBlockModule {
    /// Basic `1x1` lateral projection block.
    BasicLatBlk(BasicLatBlk),
}

impl LateralBlockModule {
    fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        match self {
            Self::BasicLatBlk(blk) => blk.forward(input),
        }
    }
}

/// `conv -> batch norm -> ReLU` feature extractor of a gradient-guided attention branch.
#[derive(Module, Debug)]
struct GdtConvs {
    conv: Conv2d,
    bn: BatchNorm,
    relu: Relu,
}

impl GdtConvs {
    fn init(
        conv2d_config: &Conv2dConfig,
        batch_norm_config: &BatchNormConfig,
        device: &Device,
    ) -> Self {
        Self {
            conv: conv2d_config.init(device),
            bn: batch_norm_config.init(device),
            relu: Relu::new(),
        }
    }

    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        self.relu.forward(self.bn.forward(self.conv.forward(x)))
    }
}

/// Multiplies `input` by `sigmoid(attn(gdt(input)))` when both GDT layers are present.
fn apply_gdt_attention(
    input: Tensor<4>,
    gdt_conv: Option<&GdtConvs>,
    gdt_attn: Option<&Conv2d>,
) -> Tensor<4> {
    match (gdt_conv, gdt_attn) {
        (Some(conv), Some(attn)) => {
            let attention = sigmoid(attn.forward(conv.forward(input.clone())));
            input * attention
        }
        _ => input,
    }
}

/// Multi-stage hierarchical decoder module for `BiRefNet`.
#[derive(Module, Debug)]
#[expect(
    clippy::struct_field_names,
    reason = "field names match BiRefNet PyTorch state_dict keys for checkpoint compatibility"
)]
pub struct Decoder {
    split: bool,
    ipt_blk5: Option<SimpleConvs>,
    ipt_blk4: Option<SimpleConvs>,
    ipt_blk3: Option<SimpleConvs>,
    ipt_blk2: Option<SimpleConvs>,
    ipt_blk1: Option<SimpleConvs>,
    decoder_block4: DecoderBlockModule,
    decoder_block3: DecoderBlockModule,
    decoder_block2: DecoderBlockModule,
    decoder_block1: DecoderBlockModule,
    lateral_block4: LateralBlockModule,
    lateral_block3: LateralBlockModule,
    lateral_block2: LateralBlockModule,
    conv_out1: Conv2d,
    conv_ms_spvn_4: Option<Conv2d>,
    conv_ms_spvn_3: Option<Conv2d>,
    conv_ms_spvn_2: Option<Conv2d>,
    gdt_convs_4: Option<GdtConvs>,
    gdt_convs_3: Option<GdtConvs>,
    gdt_convs_2: Option<GdtConvs>,
    gdt_convs_pred_4: Option<Conv2d>,
    gdt_convs_pred_3: Option<Conv2d>,
    gdt_convs_pred_2: Option<Conv2d>,
    gdt_convs_attn_4: Option<Conv2d>,
    gdt_convs_attn_3: Option<Conv2d>,
    gdt_convs_attn_2: Option<Conv2d>,
    #[module(skip)]
    interpolation_strategy: InterpolationStrategy,
}

impl Decoder {
    /// Concatenates `feature` with the image-pyramid branch `ipt_blk(x)` resized to it.
    fn process_ipt_block(
        &self,
        ipt_blk: Option<&SimpleConvs>,
        x: &Tensor<4>,
        feature: Tensor<4>,
    ) -> Tensor<4> {
        let Some(blk) = ipt_blk else {
            return feature;
        };

        let [_, _, h, w] = feature.dims();
        let pyramid_input = if self.split {
            patches::patches_batch(x.clone(), h, w)
        } else {
            x.clone()
        };
        let pyramid = blk.forward(intelligent_interpolate(
            pyramid_input,
            [h, w],
            self.interpolation_strategy,
        ));

        Tensor::cat(vec![feature, pyramid], 1)
    }

    fn resize_like(&self, tensor: Tensor<4>, like: &Tensor<4>) -> Tensor<4> {
        let [_, _, h, w] = like.dims();
        intelligent_interpolate(tensor, [h, w], self.interpolation_strategy)
    }

    fn forward_core(&self, features: [Tensor<4>; 5], collect_ms: bool) -> Vec<Tensor<4>> {
        let [x, x1, x2, x3, x4] = features;
        let mut outs = Vec::with_capacity(4);
        let mut side_output = |conv: Option<&Conv2d>, p: &Tensor<4>| {
            if collect_ms && let Some(conv) = conv {
                outs.push(conv.forward(p.clone()));
            }
        };

        // Stage 4
        let p4 = self.process_ipt_block(self.ipt_blk5.as_ref(), &x, x4);
        let p4 = apply_gdt_attention(
            self.decoder_block4.forward(p4),
            self.gdt_convs_4.as_ref(),
            self.gdt_convs_attn_4.as_ref(),
        );
        side_output(self.conv_ms_spvn_4.as_ref(), &p4);
        let p3 = self.resize_like(p4, &x3) + self.lateral_block4.forward(x3);

        // Stage 3
        let p3 = self.process_ipt_block(self.ipt_blk4.as_ref(), &x, p3);
        let p3 = apply_gdt_attention(
            self.decoder_block3.forward(p3),
            self.gdt_convs_3.as_ref(),
            self.gdt_convs_attn_3.as_ref(),
        );
        side_output(self.conv_ms_spvn_3.as_ref(), &p3);
        let p2 = self.resize_like(p3, &x2) + self.lateral_block3.forward(x2);

        // Stage 2
        let p2 = self.process_ipt_block(self.ipt_blk3.as_ref(), &x, p2);
        let p2 = apply_gdt_attention(
            self.decoder_block2.forward(p2),
            self.gdt_convs_2.as_ref(),
            self.gdt_convs_attn_2.as_ref(),
        );
        side_output(self.conv_ms_spvn_2.as_ref(), &p2);
        let p1 = self.resize_like(p2, &x1) + self.lateral_block2.forward(x1);

        // Stage 1
        let p1 = self.process_ipt_block(self.ipt_blk2.as_ref(), &x, p1);
        let p1 = self.resize_like(self.decoder_block1.forward(p1), &x);
        let p1 = self.process_ipt_block(self.ipt_blk1.as_ref(), &x, p1);
        outs.push(self.conv_out1.forward(p1));

        outs
    }

    /// Runs the decoder and returns only the final segmentation logit map `[B, 1, H, W]`.
    #[must_use]
    pub fn forward(&self, features: [Tensor<4>; 5]) -> Tensor<4> {
        let [x, x1, x2, x3, x4] = features;
        let mut outs = self.forward_core([x, x1, x2, x3, x4], false);
        outs.pop()
            .unwrap_or_else(|| panic!("decoder emitted {} outputs, expected 1", outs.len()))
    }

    /// Runs the decoder and returns multi-scale supervision logits followed by the final prediction.
    #[must_use]
    pub fn forward_multi_scale(&self, features: [Tensor<4>; 5]) -> Vec<Tensor<4>> {
        self.forward_core(features, true)
    }
}
