use anyhow::{Context, Result};
use birefnet_burn::BiRefNetRecord;
use burn::{
    backend::{ndarray::NdArrayDevice, NdArray},
    record::{CompactRecorder, DefaultRecorder, FullPrecisionSettings, Recorder},
};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use std::path::PathBuf;

use clap::{Parser, ValueEnum};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    model: PathBuf,
    name: BiRefNetModels,
    #[clap(short, long, default_value_t = false)]
    half: bool,
}

#[derive(ValueEnum, Debug, Clone)]
enum BiRefNetModels {
    BiRefNetLegacy,
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
    let device = NdArrayDevice::default();

    let args = Args::parse();

    let path = args.model.as_path();
    let name = match args.name {
        BiRefNetModels::BiRefNetLegacy => "BiRefNetLegacy",
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
    let dir = path
        .parent()
        .or_else(|| Some(std::path::Path::new(".")))
        .unwrap();

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

    if args.half {
        let recorder = CompactRecorder::default();
        recorder
            .record(record, dir.join(format!("{}-half", name)))
            .with_context(|| "Should record successfully")?;
    } else {
        let recorder = DefaultRecorder::default();
        recorder
            .record(record, dir.join(name))
            .with_context(|| "Should record successfully")?;
    }

    Ok(())
}
