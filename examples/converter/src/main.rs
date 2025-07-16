use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use birefnet_burn::{
    Backbone, BiRefNetRecord, DecAtt, DecBlk, DecChannelsInter, LatBlk, ModelConfig, MulSclIpt,
    Prompt4loc, SqueezeBlock,
};
use burn::{
    backend::NdArray,
    prelude::*,
    record::{CompactRecorder, DefaultRecorder, FullPrecisionSettings, Recorder},
};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use clap::{Parser, ValueEnum};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    model: PathBuf,
    name: BiRefNetModels,
    #[clap(long, default_value_t = false)]
    half: bool,
}

#[derive(ValueEnum, Debug, Clone)]
enum BiRefNetModels {
    BiRefNetPortrait,
    BiRefNet,
    BiRefNetCOD,
    BiRefNetDIS5k,
    BiRefNetHRSOD,
    BiRefNetDIS5KTRTEs,
    BiRefNetLite,
    BiRefNetLite2k,
    BiRefNetMatting,
}

fn main() -> Result<()> {
    type Backend = NdArray;
    let device = Default::default();

    let args = Args::parse();

    let path = args.model.as_path();
    let name = match args.name {
        BiRefNetModels::BiRefNetPortrait => "BiRefNetPortrait",
        BiRefNetModels::BiRefNet => "BiRefNet",
        BiRefNetModels::BiRefNetCOD => "BiRefNetCOD",
        BiRefNetModels::BiRefNetDIS5k => "BiRefNetDIS5k",
        BiRefNetModels::BiRefNetHRSOD => "BiRefNetHRSOD",
        BiRefNetModels::BiRefNetDIS5KTRTEs => "BiRefNetDIS5KTRTEs",
        BiRefNetModels::BiRefNetLite => "BiRefNetLite",
        BiRefNetModels::BiRefNetLite2k => "BiRefNetLite2k",
        BiRefNetModels::BiRefNetMatting => "BiRefNetMatting",
    };
    let dir = path.parent().or_else(|| Some(Path::new("."))).unwrap();

    let load_args = LoadArgs::new(path.to_path_buf())
        //
        .with_key_remap("decoder\\.conv_out1\\.0\\.(.+)", "decoder.conv_out1.$1")
        .with_key_remap(
            "decoder\\.gdt_convs_attn_([2-4])\\.0\\.(.+)",
            "decoder.gdt_convs_attn_$1.$2",
        )
        .with_key_remap(
            "decoder\\.gdt_convs_pred_([2-4])\\.0\\.(.+)",
            "decoder.gdt_convs_pred_$1.$2",
        )
        // Sequential
        .with_key_remap("bb\\.norm([0-3])\\.(.+)", "bb.norm_layers.$1.$2")
        .with_key_remap(
            "(.+?)\\.gdt_convs_([2-4])\\.0\\.(.+)",
            "$1.gdt_convs_$2.conv.$3",
        )
        .with_key_remap(
            "(.+?)\\.gdt_convs_([2-4])\\.1\\.(.+)",
            "$1.gdt_convs_$2.bn.$3",
        )
        .with_key_remap(
            "(.+?)\\.global_avg_pool\\.([0-3])\\.(.+)",
            "$1.global_avg_pool_$2.$3",
        );

    let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();
    let record: BiRefNetRecord<Backend> = recorder
        .load(load_args, &device)
        .with_context(|| "Should decode state successfully")?;

    let config = match args.name {
        BiRefNetModels::BiRefNetPortrait => {
            let base_config = ModelConfig::new();
            base_config
                .clone()
                .with_task(base_config.task.with_prompt4_loc(Prompt4loc::Dense))
                .with_decoder(
                    base_config
                        .decoder
                        .with_ms_supervision(true)
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
                .with_backbone(base_config.backbone.with_backbone(Backbone::SwinV1L))
        }
        BiRefNetModels::BiRefNet => {
            let base_config = ModelConfig::new();
            base_config
                .clone()
                .with_task(base_config.task.with_prompt4_loc(Prompt4loc::Dense))
                .with_decoder(
                    base_config
                        .decoder
                        .with_ms_supervision(true)
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
                .with_backbone(base_config.backbone.with_backbone(Backbone::SwinV1L))
        }
        BiRefNetModels::BiRefNetCOD => {
            let base_config = ModelConfig::new();
            base_config
                .clone()
                .with_task(base_config.task.with_prompt4_loc(Prompt4loc::Dense))
                .with_decoder(
                    base_config
                        .decoder
                        .with_ms_supervision(true)
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
                .with_backbone(base_config.backbone.with_backbone(Backbone::SwinV1L))
        }
        BiRefNetModels::BiRefNetDIS5k => {
            let base_config = ModelConfig::new();
            base_config
                .clone()
                .with_task(base_config.task.with_prompt4_loc(Prompt4loc::Dense))
                .with_decoder(
                    base_config
                        .decoder
                        .with_ms_supervision(true)
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
                .with_backbone(base_config.backbone.with_backbone(Backbone::SwinV1L))
        }
        BiRefNetModels::BiRefNetHRSOD => {
            let base_config = ModelConfig::new();
            base_config
                .clone()
                .with_task(base_config.task.with_prompt4_loc(Prompt4loc::Dense))
                .with_decoder(
                    base_config
                        .decoder
                        .with_ms_supervision(true)
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
                .with_backbone(base_config.backbone.with_backbone(Backbone::SwinV1L))
        }
        BiRefNetModels::BiRefNetDIS5KTRTEs => {
            let base_config = ModelConfig::new();
            base_config
                .clone()
                .with_task(base_config.task.with_prompt4_loc(Prompt4loc::Dense))
                .with_decoder(
                    base_config
                        .decoder
                        .with_ms_supervision(true)
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
                .with_backbone(base_config.backbone.with_backbone(Backbone::SwinV1L))
        }
        BiRefNetModels::BiRefNetLite => {
            let base_config = ModelConfig::new();
            base_config
                .clone()
                .with_task(base_config.task.with_prompt4_loc(Prompt4loc::Dense))
                .with_decoder(
                    base_config
                        .decoder
                        .with_ms_supervision(true)
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
                .with_backbone(base_config.backbone.with_backbone(Backbone::SwinV1T))
        }
        BiRefNetModels::BiRefNetLite2k => {
            let base_config = ModelConfig::new();
            base_config
                .clone()
                .with_task(base_config.task.with_prompt4_loc(Prompt4loc::Dense))
                .with_decoder(
                    base_config
                        .decoder
                        .with_ms_supervision(true)
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
                .with_backbone(base_config.backbone.with_backbone(Backbone::SwinV1T))
        }
        BiRefNetModels::BiRefNetMatting => {
            let base_config = ModelConfig::new();
            base_config
                .clone()
                .with_task(base_config.task.with_prompt4_loc(Prompt4loc::Dense))
                .with_decoder(
                    base_config
                        .decoder
                        .with_ms_supervision(true)
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
                .with_backbone(base_config.backbone.with_backbone(Backbone::SwinV1L))
        }
    };
    config
        .save(dir.join(format!("{name}.json")))
        .with_context(|| "Should save config successfully")?;

    if args.half {
        let recorder = CompactRecorder::default();
        recorder
            .record(record, dir.join(format!("{name}-half")))
            .with_context(|| "Should record successfully")?;
    } else {
        let recorder = DefaultRecorder::default();
        recorder
            .record(record, dir.join(name))
            .with_context(|| "Should record successfully")?;
    }

    Ok(())
}
