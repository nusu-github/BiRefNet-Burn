use birefnet_burn::{
    Backbone, BackboneConfig, BiRefNetConfig, BiRefNetLossConfig, DecAtt, DecBlk, DecChannelsInter,
    DecoderConfig, LatBlk, LossWeightsConfig, ModelConfig, MulSclIpt, PixLossConfig, Prompt4loc,
    Refine, RefineConfig, SqueezeBlock, Task, TaskConfig,
};
use hf_hub::api::sync;
use hf_hub::{Repo, RepoType};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::LazyLock;

/// モデルの仕様を定義
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub hf_model_id: &'static str, // HuggingFace model ID (e.g., "BiRefNet")
    pub default_resolution: (u32, u32),
    pub supports_dynamic_resolution: bool,
    pub config_builder: fn() -> BiRefNetConfig,
}

/// モデルカタログ - 元の実装と同じシンプルなマッピング方式
static MODEL_SPECS: LazyLock<HashMap<String, ModelSpec>> = LazyLock::new(|| {
    let mut specs = HashMap::new();

    // Define all models in one place
    let models = [
        (
            "General",
            ModelSpec {
                hf_model_id: "BiRefNet",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: false,
                config_builder: || {
                    let model_config = ModelConfig::new()
                        .with_task(
                            TaskConfig::new()
                                .with_task(Task::DIS5K)
                                .with_prompt4_loc(Prompt4loc::Dense)
                                .with_batch_size(4),
                        )
                        .with_backbone(BackboneConfig::new().with_backbone(Backbone::SwinV1L))
                        .with_decoder(
                            DecoderConfig::new()
                                .with_ms_supervision(true)
                                .with_out_ref(true)
                                .with_dec_ipt(true)
                                .with_dec_ipt_split(true)
                                .with_cxt_num(3)
                                .with_mul_scl_ipt(MulSclIpt::Cat)
                                .with_dec_att(DecAtt::ASPPDeformable)
                                .with_squeeze_block(SqueezeBlock::BasicDecBlk(1))
                                .with_dec_blk(DecBlk::BasicDecBlk)
                                .with_lat_blk(LatBlk::BasicLatBlk)
                                .with_dec_channels_inter(DecChannelsInter::Fixed),
                        )
                        .with_refine(RefineConfig::new().with_refine(Refine::None));

                    let loss_config = BiRefNetLossConfig::new(PixLossConfig::new(
                        LossWeightsConfig::new()
                            .with_bce(30.0 * 1.0)
                            .with_iou(0.5 * 1.0)
                            .with_iou_patch(0.5 * 0.0)
                            .with_mse(150.0 * 0.0)
                            .with_triplet(3.0 * 0.0)
                            .with_reg(100.0 * 0.0)
                            .with_ssim(10.0 * 1.0)
                            .with_cnt(5.0 * 0.0)
                            .with_structure(5.0 * 0.0),
                    ));
                    BiRefNetConfig::new(model_config, loss_config)
                },
            },
        ),
        (
            "General-HR",
            ModelSpec {
                hf_model_id: "BiRefNet_HR",
                default_resolution: (2048, 2048),
                supports_dynamic_resolution: false,
                config_builder: || {
                    // Same as General but with HR resolution
                    let model_config = ModelConfig::new()
                        .with_task(
                            TaskConfig::new()
                                .with_task(Task::DIS5K)
                                .with_prompt4_loc(Prompt4loc::Dense)
                                .with_batch_size(4),
                        )
                        .with_backbone(BackboneConfig::new().with_backbone(Backbone::SwinV1L))
                        .with_decoder(
                            DecoderConfig::new()
                                .with_ms_supervision(true)
                                .with_out_ref(true)
                                .with_dec_ipt(true)
                                .with_dec_ipt_split(true)
                                .with_cxt_num(3)
                                .with_mul_scl_ipt(MulSclIpt::Cat)
                                .with_dec_att(DecAtt::ASPPDeformable)
                                .with_squeeze_block(SqueezeBlock::BasicDecBlk(1))
                                .with_dec_blk(DecBlk::BasicDecBlk)
                                .with_lat_blk(LatBlk::BasicLatBlk)
                                .with_dec_channels_inter(DecChannelsInter::Fixed),
                        )
                        .with_refine(RefineConfig::new().with_refine(Refine::None));

                    let loss_config = BiRefNetLossConfig::new(PixLossConfig::new(
                        LossWeightsConfig::new()
                            .with_bce(30.0 * 1.0)
                            .with_iou(0.5 * 1.0)
                            .with_iou_patch(0.5 * 0.0)
                            .with_mse(150.0 * 0.0)
                            .with_triplet(3.0 * 0.0)
                            .with_reg(100.0 * 0.0)
                            .with_ssim(10.0 * 1.0)
                            .with_cnt(5.0 * 0.0)
                            .with_structure(5.0 * 0.0),
                    ));
                    BiRefNetConfig::new(model_config, loss_config)
                },
            },
        ),
        (
            "Matting-HR",
            ModelSpec {
                hf_model_id: "BiRefNet_HR-matting",
                default_resolution: (2048, 2048),
                supports_dynamic_resolution: false,
                config_builder: || {
                    // Matting task with HR resolution
                    let model_config = ModelConfig::new()
                        .with_task(
                            TaskConfig::new()
                                .with_task(Task::Matting) // Matting task
                                .with_prompt4_loc(Prompt4loc::Dense)
                                .with_batch_size(4),
                        )
                        .with_backbone(BackboneConfig::new().with_backbone(Backbone::SwinV1L))
                        .with_decoder(
                            DecoderConfig::new()
                                .with_ms_supervision(true)
                                .with_out_ref(true)
                                .with_dec_ipt(true)
                                .with_dec_ipt_split(true)
                                .with_cxt_num(3)
                                .with_mul_scl_ipt(MulSclIpt::Cat)
                                .with_dec_att(DecAtt::ASPPDeformable)
                                .with_squeeze_block(SqueezeBlock::BasicDecBlk(1))
                                .with_dec_blk(DecBlk::BasicDecBlk)
                                .with_lat_blk(LatBlk::BasicLatBlk)
                                .with_dec_channels_inter(DecChannelsInter::Fixed),
                        )
                        .with_refine(RefineConfig::new().with_refine(Refine::None));

                    let loss_config = BiRefNetLossConfig::new(PixLossConfig::new(
                        LossWeightsConfig::new()
                            .with_bce(30.0 * 1.0)
                            .with_iou(0.5 * 1.0)
                            .with_iou_patch(0.5 * 0.0)
                            .with_mse(150.0 * 0.0)
                            .with_triplet(3.0 * 0.0)
                            .with_reg(100.0 * 0.0)
                            .with_ssim(10.0 * 1.0)
                            .with_cnt(5.0 * 0.0)
                            .with_structure(5.0 * 0.0),
                    ));
                    BiRefNetConfig::new(model_config, loss_config)
                },
            },
        ),
        (
            "Matting",
            ModelSpec {
                hf_model_id: "BiRefNet-matting",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: false,
                config_builder: || {
                    let model_config = ModelConfig::new()
                        .with_task(
                            TaskConfig::new()
                                .with_task(Task::Matting) // Matting task
                                .with_prompt4_loc(Prompt4loc::Dense)
                                .with_batch_size(4),
                        )
                        .with_backbone(BackboneConfig::new().with_backbone(Backbone::SwinV1L))
                        .with_decoder(
                            DecoderConfig::new()
                                .with_ms_supervision(true)
                                .with_out_ref(true)
                                .with_dec_ipt(true)
                                .with_dec_ipt_split(true)
                                .with_cxt_num(3)
                                .with_mul_scl_ipt(MulSclIpt::Cat)
                                .with_dec_att(DecAtt::ASPPDeformable)
                                .with_squeeze_block(SqueezeBlock::BasicDecBlk(1))
                                .with_dec_blk(DecBlk::BasicDecBlk)
                                .with_lat_blk(LatBlk::BasicLatBlk)
                                .with_dec_channels_inter(DecChannelsInter::Fixed),
                        )
                        .with_refine(RefineConfig::new().with_refine(Refine::None));

                    let loss_config = BiRefNetLossConfig::new(PixLossConfig::new(
                        LossWeightsConfig::new()
                            .with_bce(30.0 * 1.0)
                            .with_iou(0.5 * 1.0)
                            .with_iou_patch(0.5 * 0.0)
                            .with_mse(150.0 * 0.0)
                            .with_triplet(3.0 * 0.0)
                            .with_reg(100.0 * 0.0)
                            .with_ssim(10.0 * 1.0)
                            .with_cnt(5.0 * 0.0)
                            .with_structure(5.0 * 0.0),
                    ));
                    BiRefNetConfig::new(model_config, loss_config)
                },
            },
        ),
        (
            "Portrait",
            ModelSpec {
                hf_model_id: "BiRefNet-portrait",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: false,
                config_builder: || {
                    let model_config = ModelConfig::new()
                        .with_task(
                            TaskConfig::new()
                                .with_task(Task::Matting) // Matting task
                                .with_prompt4_loc(Prompt4loc::Dense)
                                .with_batch_size(4),
                        )
                        .with_backbone(BackboneConfig::new().with_backbone(Backbone::SwinV1L))
                        .with_decoder(
                            DecoderConfig::new()
                                .with_ms_supervision(true)
                                .with_out_ref(true)
                                .with_dec_ipt(true)
                                .with_dec_ipt_split(true)
                                .with_cxt_num(3)
                                .with_mul_scl_ipt(MulSclIpt::Cat)
                                .with_dec_att(DecAtt::ASPPDeformable)
                                .with_squeeze_block(SqueezeBlock::BasicDecBlk(1))
                                .with_dec_blk(DecBlk::BasicDecBlk)
                                .with_lat_blk(LatBlk::BasicLatBlk)
                                .with_dec_channels_inter(DecChannelsInter::Fixed),
                        )
                        .with_refine(RefineConfig::new().with_refine(Refine::None));

                    let loss_config = BiRefNetLossConfig::new(PixLossConfig::new(
                        LossWeightsConfig::new()
                            .with_bce(30.0 * 1.0)
                            .with_iou(0.5 * 1.0)
                            .with_iou_patch(0.5 * 0.0)
                            .with_mse(150.0 * 0.0)
                            .with_triplet(3.0 * 0.0)
                            .with_reg(100.0 * 0.0)
                            .with_ssim(10.0 * 1.0)
                            .with_cnt(5.0 * 0.0)
                            .with_structure(5.0 * 0.0),
                    ));
                    BiRefNetConfig::new(model_config, loss_config)
                },
            },
        ),
        (
            "General-reso_512",
            ModelSpec {
                hf_model_id: "BiRefNet_512x512",
                default_resolution: (512, 512),
                supports_dynamic_resolution: false,
                config_builder: || {
                    let model_config = ModelConfig::new()
                        .with_task(
                            TaskConfig::new()
                                .with_task(Task::Matting) // Matting task
                                .with_prompt4_loc(Prompt4loc::Dense)
                                .with_batch_size(4),
                        )
                        .with_backbone(BackboneConfig::new().with_backbone(Backbone::SwinV1L))
                        .with_decoder(
                            DecoderConfig::new()
                                .with_ms_supervision(true)
                                .with_out_ref(true)
                                .with_dec_ipt(true)
                                .with_dec_ipt_split(true)
                                .with_cxt_num(3)
                                .with_mul_scl_ipt(MulSclIpt::Cat)
                                .with_dec_att(DecAtt::ASPPDeformable)
                                .with_squeeze_block(SqueezeBlock::BasicDecBlk(1))
                                .with_dec_blk(DecBlk::BasicDecBlk)
                                .with_lat_blk(LatBlk::BasicLatBlk)
                                .with_dec_channels_inter(DecChannelsInter::Fixed),
                        )
                        .with_refine(RefineConfig::new().with_refine(Refine::None));

                    let loss_config = BiRefNetLossConfig::new(PixLossConfig::new(
                        LossWeightsConfig::new()
                            .with_bce(30.0 * 1.0)
                            .with_iou(0.5 * 1.0)
                            .with_iou_patch(0.5 * 0.0)
                            .with_mse(150.0 * 0.0)
                            .with_triplet(3.0 * 0.0)
                            .with_reg(100.0 * 0.0)
                            .with_ssim(10.0 * 1.0)
                            .with_cnt(5.0 * 0.0)
                            .with_structure(5.0 * 0.0),
                    ));
                    BiRefNetConfig::new(model_config, loss_config)
                },
            },
        ),
        (
            "General-Lite",
            ModelSpec {
                hf_model_id: "BiRefNet_lite",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: false,
                config_builder: || {
                    let model_config = ModelConfig::new()
                        .with_task(
                            TaskConfig::new()
                                .with_task(Task::Matting) // Matting task
                                .with_prompt4_loc(Prompt4loc::Dense)
                                .with_batch_size(4),
                        )
                        .with_backbone(BackboneConfig::new().with_backbone(Backbone::SwinV1L))
                        .with_decoder(
                            DecoderConfig::new()
                                .with_ms_supervision(true)
                                .with_out_ref(true)
                                .with_dec_ipt(true)
                                .with_dec_ipt_split(true)
                                .with_cxt_num(3)
                                .with_mul_scl_ipt(MulSclIpt::Cat)
                                .with_dec_att(DecAtt::ASPPDeformable)
                                .with_squeeze_block(SqueezeBlock::BasicDecBlk(1))
                                .with_dec_blk(DecBlk::BasicDecBlk)
                                .with_lat_blk(LatBlk::BasicLatBlk)
                                .with_dec_channels_inter(DecChannelsInter::Fixed),
                        )
                        .with_refine(RefineConfig::new().with_refine(Refine::None));

                    let loss_config = BiRefNetLossConfig::new(PixLossConfig::new(
                        LossWeightsConfig::new()
                            .with_bce(30.0 * 1.0)
                            .with_iou(0.5 * 1.0)
                            .with_iou_patch(0.5 * 0.0)
                            .with_mse(150.0 * 0.0)
                            .with_triplet(3.0 * 0.0)
                            .with_reg(100.0 * 0.0)
                            .with_ssim(10.0 * 1.0)
                            .with_cnt(5.0 * 0.0)
                            .with_structure(5.0 * 0.0),
                    ));
                    BiRefNetConfig::new(model_config, loss_config)
                },
            },
        ),
        (
            "General-Lite-2K",
            ModelSpec {
                hf_model_id: "BiRefNet_lite-2K",
                default_resolution: (2560, 1440),
                supports_dynamic_resolution: false,
                config_builder: || {
                    let model_config = ModelConfig::new()
                        .with_task(
                            TaskConfig::new()
                                .with_task(Task::Matting) // Matting task
                                .with_prompt4_loc(Prompt4loc::Dense)
                                .with_batch_size(4),
                        )
                        .with_backbone(BackboneConfig::new().with_backbone(Backbone::SwinV1L))
                        .with_decoder(
                            DecoderConfig::new()
                                .with_ms_supervision(true)
                                .with_out_ref(true)
                                .with_dec_ipt(true)
                                .with_dec_ipt_split(true)
                                .with_cxt_num(3)
                                .with_mul_scl_ipt(MulSclIpt::Cat)
                                .with_dec_att(DecAtt::ASPPDeformable)
                                .with_squeeze_block(SqueezeBlock::BasicDecBlk(1))
                                .with_dec_blk(DecBlk::BasicDecBlk)
                                .with_lat_blk(LatBlk::BasicLatBlk)
                                .with_dec_channels_inter(DecChannelsInter::Fixed),
                        )
                        .with_refine(RefineConfig::new().with_refine(Refine::None));

                    let loss_config = BiRefNetLossConfig::new(PixLossConfig::new(
                        LossWeightsConfig::new()
                            .with_bce(30.0 * 1.0)
                            .with_iou(0.5 * 1.0)
                            .with_iou_patch(0.5 * 0.0)
                            .with_mse(150.0 * 0.0)
                            .with_triplet(3.0 * 0.0)
                            .with_reg(100.0 * 0.0)
                            .with_ssim(10.0 * 1.0)
                            .with_cnt(5.0 * 0.0)
                            .with_structure(5.0 * 0.0),
                    ));
                    BiRefNetConfig::new(model_config, loss_config)
                },
            },
        ),
        (
            "DIS",
            ModelSpec {
                hf_model_id: "BiRefNet-DIS5K",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: false,
                config_builder: || {
                    let model_config = ModelConfig::new()
                        .with_task(
                            TaskConfig::new()
                                .with_task(Task::Matting) // Matting task
                                .with_prompt4_loc(Prompt4loc::Dense)
                                .with_batch_size(4),
                        )
                        .with_backbone(BackboneConfig::new().with_backbone(Backbone::SwinV1L))
                        .with_decoder(
                            DecoderConfig::new()
                                .with_ms_supervision(true)
                                .with_out_ref(true)
                                .with_dec_ipt(true)
                                .with_dec_ipt_split(true)
                                .with_cxt_num(3)
                                .with_mul_scl_ipt(MulSclIpt::Cat)
                                .with_dec_att(DecAtt::ASPPDeformable)
                                .with_squeeze_block(SqueezeBlock::BasicDecBlk(1))
                                .with_dec_blk(DecBlk::BasicDecBlk)
                                .with_lat_blk(LatBlk::BasicLatBlk)
                                .with_dec_channels_inter(DecChannelsInter::Fixed),
                        )
                        .with_refine(RefineConfig::new().with_refine(Refine::None));

                    let loss_config = BiRefNetLossConfig::new(PixLossConfig::new(
                        LossWeightsConfig::new()
                            .with_bce(30.0 * 1.0)
                            .with_iou(0.5 * 1.0)
                            .with_iou_patch(0.5 * 0.0)
                            .with_mse(150.0 * 0.0)
                            .with_triplet(3.0 * 0.0)
                            .with_reg(100.0 * 0.0)
                            .with_ssim(10.0 * 1.0)
                            .with_cnt(5.0 * 0.0)
                            .with_structure(5.0 * 0.0),
                    ));
                    BiRefNetConfig::new(model_config, loss_config)
                },
            },
        ),
        (
            "HRSOD",
            ModelSpec {
                hf_model_id: "BiRefNet-HRSOD",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: false,
                config_builder: || {
                    let model_config = ModelConfig::new()
                        .with_task(
                            TaskConfig::new()
                                .with_task(Task::Matting) // Matting task
                                .with_prompt4_loc(Prompt4loc::Dense)
                                .with_batch_size(4),
                        )
                        .with_backbone(BackboneConfig::new().with_backbone(Backbone::SwinV1L))
                        .with_decoder(
                            DecoderConfig::new()
                                .with_ms_supervision(true)
                                .with_out_ref(true)
                                .with_dec_ipt(true)
                                .with_dec_ipt_split(true)
                                .with_cxt_num(3)
                                .with_mul_scl_ipt(MulSclIpt::Cat)
                                .with_dec_att(DecAtt::ASPPDeformable)
                                .with_squeeze_block(SqueezeBlock::BasicDecBlk(1))
                                .with_dec_blk(DecBlk::BasicDecBlk)
                                .with_lat_blk(LatBlk::BasicLatBlk)
                                .with_dec_channels_inter(DecChannelsInter::Fixed),
                        )
                        .with_refine(RefineConfig::new().with_refine(Refine::None));

                    let loss_config = BiRefNetLossConfig::new(PixLossConfig::new(
                        LossWeightsConfig::new()
                            .with_bce(30.0 * 1.0)
                            .with_iou(0.5 * 1.0)
                            .with_iou_patch(0.5 * 0.0)
                            .with_mse(150.0 * 0.0)
                            .with_triplet(3.0 * 0.0)
                            .with_reg(100.0 * 0.0)
                            .with_ssim(10.0 * 1.0)
                            .with_cnt(5.0 * 0.0)
                            .with_structure(5.0 * 0.0),
                    ));
                    BiRefNetConfig::new(model_config, loss_config)
                },
            },
        ),
        (
            "COD",
            ModelSpec {
                hf_model_id: "BiRefNet-COD",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: false,
                config_builder: || {
                    let model_config = ModelConfig::new()
                        .with_task(
                            TaskConfig::new()
                                .with_task(Task::Matting) // Matting task
                                .with_prompt4_loc(Prompt4loc::Dense)
                                .with_batch_size(4),
                        )
                        .with_backbone(BackboneConfig::new().with_backbone(Backbone::SwinV1L))
                        .with_decoder(
                            DecoderConfig::new()
                                .with_ms_supervision(true)
                                .with_out_ref(true)
                                .with_dec_ipt(true)
                                .with_dec_ipt_split(true)
                                .with_cxt_num(3)
                                .with_mul_scl_ipt(MulSclIpt::Cat)
                                .with_dec_att(DecAtt::ASPPDeformable)
                                .with_squeeze_block(SqueezeBlock::BasicDecBlk(1))
                                .with_dec_blk(DecBlk::BasicDecBlk)
                                .with_lat_blk(LatBlk::BasicLatBlk)
                                .with_dec_channels_inter(DecChannelsInter::Fixed),
                        )
                        .with_refine(RefineConfig::new().with_refine(Refine::None));

                    let loss_config = BiRefNetLossConfig::new(PixLossConfig::new(
                        LossWeightsConfig::new()
                            .with_bce(30.0 * 1.0)
                            .with_iou(0.5 * 1.0)
                            .with_iou_patch(0.5 * 0.0)
                            .with_mse(150.0 * 0.0)
                            .with_triplet(3.0 * 0.0)
                            .with_reg(100.0 * 0.0)
                            .with_ssim(10.0 * 1.0)
                            .with_cnt(5.0 * 0.0)
                            .with_structure(5.0 * 0.0),
                    ));
                    BiRefNetConfig::new(model_config, loss_config)
                },
            },
        ),
        (
            "DIS-TR_TEs",
            ModelSpec {
                hf_model_id: "BiRefNet-DIS5K-TR_TEs",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: false,
                config_builder: || {
                    let model_config = ModelConfig::new()
                        .with_task(
                            TaskConfig::new()
                                .with_task(Task::Matting) // Matting task
                                .with_prompt4_loc(Prompt4loc::Dense)
                                .with_batch_size(4),
                        )
                        .with_backbone(BackboneConfig::new().with_backbone(Backbone::SwinV1L))
                        .with_decoder(
                            DecoderConfig::new()
                                .with_ms_supervision(true)
                                .with_out_ref(true)
                                .with_dec_ipt(true)
                                .with_dec_ipt_split(true)
                                .with_cxt_num(3)
                                .with_mul_scl_ipt(MulSclIpt::Cat)
                                .with_dec_att(DecAtt::ASPPDeformable)
                                .with_squeeze_block(SqueezeBlock::BasicDecBlk(1))
                                .with_dec_blk(DecBlk::BasicDecBlk)
                                .with_lat_blk(LatBlk::BasicLatBlk)
                                .with_dec_channels_inter(DecChannelsInter::Fixed),
                        )
                        .with_refine(RefineConfig::new().with_refine(Refine::None));

                    let loss_config = BiRefNetLossConfig::new(PixLossConfig::new(
                        LossWeightsConfig::new()
                            .with_bce(30.0 * 1.0)
                            .with_iou(0.5 * 1.0)
                            .with_iou_patch(0.5 * 0.0)
                            .with_mse(150.0 * 0.0)
                            .with_triplet(3.0 * 0.0)
                            .with_reg(100.0 * 0.0)
                            .with_ssim(10.0 * 1.0)
                            .with_cnt(5.0 * 0.0)
                            .with_structure(5.0 * 0.0),
                    ));
                    BiRefNetConfig::new(model_config, loss_config)
                },
            },
        ),
        (
            "General-legacy",
            ModelSpec {
                hf_model_id: "BiRefNet-legacy",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: false,
                config_builder: || {
                    let model_config = ModelConfig::new()
                        .with_task(
                            TaskConfig::new()
                                .with_task(Task::Matting) // Matting task
                                .with_prompt4_loc(Prompt4loc::Dense)
                                .with_batch_size(4),
                        )
                        .with_backbone(BackboneConfig::new().with_backbone(Backbone::SwinV1L))
                        .with_decoder(
                            DecoderConfig::new()
                                .with_ms_supervision(true)
                                .with_out_ref(true)
                                .with_dec_ipt(true)
                                .with_dec_ipt_split(true)
                                .with_cxt_num(3)
                                .with_mul_scl_ipt(MulSclIpt::Cat)
                                .with_dec_att(DecAtt::ASPPDeformable)
                                .with_squeeze_block(SqueezeBlock::BasicDecBlk(1))
                                .with_dec_blk(DecBlk::BasicDecBlk)
                                .with_lat_blk(LatBlk::BasicLatBlk)
                                .with_dec_channels_inter(DecChannelsInter::Fixed),
                        )
                        .with_refine(RefineConfig::new().with_refine(Refine::None));

                    let loss_config = BiRefNetLossConfig::new(PixLossConfig::new(
                        LossWeightsConfig::new()
                            .with_bce(30.0 * 1.0)
                            .with_iou(0.5 * 1.0)
                            .with_iou_patch(0.5 * 0.0)
                            .with_mse(150.0 * 0.0)
                            .with_triplet(3.0 * 0.0)
                            .with_reg(100.0 * 0.0)
                            .with_ssim(10.0 * 1.0)
                            .with_cnt(5.0 * 0.0)
                            .with_structure(5.0 * 0.0),
                    ));
                    BiRefNetConfig::new(model_config, loss_config)
                },
            },
        ),
        (
            "General-dynamic",
            ModelSpec {
                hf_model_id: "BiRefNet_dynamic",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: true,
                config_builder: || {
                    let model_config = ModelConfig::new()
                        .with_task(
                            TaskConfig::new()
                                .with_task(Task::Matting) // Matting task
                                .with_prompt4_loc(Prompt4loc::Dense)
                                .with_batch_size(4),
                        )
                        .with_backbone(BackboneConfig::new().with_backbone(Backbone::SwinV1L))
                        .with_decoder(
                            DecoderConfig::new()
                                .with_ms_supervision(true)
                                .with_out_ref(true)
                                .with_dec_ipt(true)
                                .with_dec_ipt_split(true)
                                .with_cxt_num(3)
                                .with_mul_scl_ipt(MulSclIpt::Cat)
                                .with_dec_att(DecAtt::ASPPDeformable)
                                .with_squeeze_block(SqueezeBlock::BasicDecBlk(1))
                                .with_dec_blk(DecBlk::BasicDecBlk)
                                .with_lat_blk(LatBlk::BasicLatBlk)
                                .with_dec_channels_inter(DecChannelsInter::Fixed),
                        )
                        .with_refine(RefineConfig::new().with_refine(Refine::None));

                    let loss_config = BiRefNetLossConfig::new(PixLossConfig::new(
                        LossWeightsConfig::new()
                            .with_bce(30.0 * 1.0)
                            .with_iou(0.5 * 1.0)
                            .with_iou_patch(0.5 * 0.0)
                            .with_mse(150.0 * 0.0)
                            .with_triplet(3.0 * 0.0)
                            .with_reg(100.0 * 0.0)
                            .with_ssim(10.0 * 1.0)
                            .with_cnt(5.0 * 0.0)
                            .with_structure(5.0 * 0.0),
                    ));
                    BiRefNetConfig::new(model_config, loss_config)
                },
            },
        ),
        (
            "Matting-dynamic",
            ModelSpec {
                hf_model_id: "BiRefNet_dynamic-matting",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: true,
                config_builder: || {
                    let model_config = ModelConfig::new()
                        .with_task(
                            TaskConfig::new()
                                .with_task(Task::Matting) // Matting task
                                .with_prompt4_loc(Prompt4loc::Dense)
                                .with_batch_size(4),
                        )
                        .with_backbone(BackboneConfig::new().with_backbone(Backbone::SwinV1L))
                        .with_decoder(
                            DecoderConfig::new()
                                .with_ms_supervision(true)
                                .with_out_ref(true)
                                .with_dec_ipt(true)
                                .with_dec_ipt_split(true)
                                .with_cxt_num(3)
                                .with_mul_scl_ipt(MulSclIpt::Cat)
                                .with_dec_att(DecAtt::ASPPDeformable)
                                .with_squeeze_block(SqueezeBlock::BasicDecBlk(1))
                                .with_dec_blk(DecBlk::BasicDecBlk)
                                .with_lat_blk(LatBlk::BasicLatBlk)
                                .with_dec_channels_inter(DecChannelsInter::Fixed),
                        )
                        .with_refine(RefineConfig::new().with_refine(Refine::None));

                    let loss_config = BiRefNetLossConfig::new(PixLossConfig::new(
                        LossWeightsConfig::new()
                            .with_bce(30.0 * 1.0)
                            .with_iou(0.5 * 1.0)
                            .with_iou_patch(0.5 * 0.0)
                            .with_mse(150.0 * 0.0)
                            .with_triplet(3.0 * 0.0)
                            .with_reg(100.0 * 0.0)
                            .with_ssim(10.0 * 1.0)
                            .with_cnt(5.0 * 0.0)
                            .with_structure(5.0 * 0.0),
                    ));
                    BiRefNetConfig::new(model_config, loss_config)
                },
            },
        ),
    ];

    for (name, spec) in models {
        specs.insert(name.to_string(), spec);
    }

    specs
});

/// モデル識別子 - シンプルに文字列ベースで管理
/// HashMapのキーとして直接使用可能
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelName(pub String);

impl ModelName {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::str::FromStr for ModelName {
    type Err = std::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::new(s))
    }
}

impl From<&str> for ModelName {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for ModelName {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl std::fmt::Display for ModelName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// ─────────────────────────────────────────────
/// ② 重みの所在 (外部キー & バリアント) ───────────
/// ─────────────────────────────────────────────
#[derive(Debug, Clone)]
pub enum WeightSource {
    /// HuggingFace から取得
    Remote {
        repo_id: String,  // hub‑hf の Repo 型
        filename: String, // blobs/paths 含む
    },
    /// ローカルファイルから読み込み
    Local { path: PathBuf },
}

/// ─────────────────────────────────────────────
/// ③ “テーブル行” に相当する 1 レコード
///     BiRefNetConfig は既存構造体をそのまま保持
///     extra_params で「列後付け」を許容
/// ─────────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct ManagedModel {
    pub name: ModelName,                // モデル名
    pub config: Option<BiRefNetConfig>, // 既存の設定
    pub weights: WeightSource,          // 重み所在
}

/// ─────────────────────────────────────────────
/// ④ アクセス用トレイト
///     実際のダウンロード / ロードはここでは実装しない
/// ─────────────────────────────────────────────
pub trait ModelRecord {
    fn name(&self) -> &ModelName;
    fn config(&self) -> &Option<BiRefNetConfig>;
    fn weight_source(&self) -> &WeightSource;
}

impl ModelRecord for ManagedModel {
    fn name(&self) -> &ModelName {
        &self.name
    }
    fn config(&self) -> &Option<BiRefNetConfig> {
        &self.config
    }
    fn weight_source(&self) -> &WeightSource {
        &self.weights
    }
}

impl ManagedModel {
    /// Create a new model record with the given name, config, and weight source.
    pub const fn new(
        name: ModelName,
        config: Option<BiRefNetConfig>,
        weights: WeightSource,
    ) -> Self {
        Self {
            name,
            config,
            weights,
        }
    }

    /// List all available pretrained models
    pub fn list_available_models() -> Vec<&'static str> {
        MODEL_SPECS.keys().map(|k| k.as_str()).collect()
    }

    /// Get model spec by name
    pub fn get_model_spec(model_name: &str) -> Option<&'static ModelSpec> {
        MODEL_SPECS.get(model_name)
    }

    /// Create a model from a known model name with default settings
    pub fn from_pretrained(model_name: &str) -> Result<Self, String> {
        let spec = MODEL_SPECS
            .get(model_name)
            .ok_or_else(|| format!("Unknown model: {model_name}"))?;

        let weights = WeightSource::Remote {
            repo_id: format!("ZhengPeng7/{}", spec.hf_model_id),
            filename: format!("{}.pth", spec.hf_model_id),
        };

        Ok(Self::new(ModelName::new(model_name), None, weights))
    }

    pub fn get_weights_path(&self) -> Option<PathBuf> {
        match &self.weights {
            WeightSource::Remote { repo_id, filename } => sync::Api::new().map_or(None, |api| {
                api.repo(Repo::new(repo_id.clone(), RepoType::Model))
                    .get(filename)
                    .ok()
            }),
            WeightSource::Local { path } => Some(path.clone()),
        }
    }

    pub fn get_config(&self) -> BiRefNetConfig {
        if let Some(config) = &self.config {
            return config.clone();
        }

        // Look up in the catalog and use config builder
        MODEL_SPECS.get(self.name.as_str()).map_or_else(
            || {
                // Default config for unknown models
                let model_config = ModelConfig::new()
                    .with_task(
                        TaskConfig::new()
                            .with_task(Task::Matting) // Matting task
                            .with_prompt4_loc(Prompt4loc::Dense)
                            .with_batch_size(4),
                    )
                    .with_backbone(BackboneConfig::new().with_backbone(Backbone::SwinV1L))
                    .with_decoder(
                        DecoderConfig::new()
                            .with_ms_supervision(true)
                            .with_out_ref(true)
                            .with_dec_ipt(true)
                            .with_dec_ipt_split(true)
                            .with_cxt_num(3)
                            .with_mul_scl_ipt(MulSclIpt::Cat)
                            .with_dec_att(DecAtt::ASPPDeformable)
                            .with_squeeze_block(SqueezeBlock::BasicDecBlk(1))
                            .with_dec_blk(DecBlk::BasicDecBlk)
                            .with_lat_blk(LatBlk::BasicLatBlk)
                            .with_dec_channels_inter(DecChannelsInter::Fixed),
                    )
                    .with_refine(RefineConfig::new().with_refine(Refine::None));

                let loss_config = BiRefNetLossConfig::new(PixLossConfig::new(
                    LossWeightsConfig::new()
                        .with_bce(30.0 * 1.0)
                        .with_iou(0.5 * 1.0)
                        .with_iou_patch(0.5 * 0.0)
                        .with_mse(150.0 * 0.0)
                        .with_triplet(3.0 * 0.0)
                        .with_reg(100.0 * 0.0)
                        .with_ssim(10.0 * 1.0)
                        .with_cnt(5.0 * 0.0)
                        .with_structure(5.0 * 0.0),
                ));
                BiRefNetConfig::new(model_config, loss_config)
            },
            |spec| (spec.config_builder)(),
        )
    }

    /// Get model resolution
    pub fn get_resolution(&self) -> Option<(u32, u32)> {
        MODEL_SPECS
            .get(self.name.as_str())
            .map(|spec| spec.default_resolution)
    }

    /// Check if model supports dynamic resolution
    pub fn supports_dynamic_resolution(&self) -> bool {
        MODEL_SPECS
            .get(self.name.as_str())
            .map(|spec| spec.supports_dynamic_resolution)
            .unwrap_or(false)
    }
}
