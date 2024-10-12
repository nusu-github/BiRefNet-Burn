use birefnet_burn::{BiRefNetConfig, BiRefNetRecord, ModelConfig};
use burn::{
    backend::Wgpu,
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};

type Backend = Wgpu<f32>;

fn main() {
    let device = Default::default();
    let load_args = LoadArgs::new("ckpt/BiRefNet-general-epoch_244.pth".into())
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
        .expect("Should decode state successfully");

    let model = BiRefNetConfig::new(
        ModelConfig::load("config.json").unwrap_or_else(|_| ModelConfig::new()),
        true,
    )
    .init::<Backend>(&device)
    .load_record(record);

    // Save the model record to a file.
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    model
        .save_file("ckpt/BiRefNet-general-epoch_244", &recorder)
        .unwrap();
}
