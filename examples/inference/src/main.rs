use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use birefnet_burn::{BiRefNetConfig, ModelConfig};
use burn::{
    nn::Sigmoid,
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
};
use clap::Parser;

use crate::imageops_ai::{AlphaMaskImage, ForegroundEstimator};
use crate::utils::{postprocess_mask, preprocess, Normalizer};

mod imageops_ai;
mod utils;

const HEIGHT: usize = 1024;
const WIDTH: usize = 1024;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    model: PathBuf,
    image: PathBuf,
}

type Backend = burn::backend::wgpu::Wgpu;
// type Backend = burn::backend::cuda_jit::Cuda;

fn main() -> Result<()> {
    let device = Default::default();

    let args = Args::parse();
    let model_path = args.model.as_path();
    let image_path = args.image.as_path();

    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
        .load(model_path.to_path_buf(), &device)
        .with_context(|| "Should decode state successfully")?;

    let model_name = model_path.file_stem().unwrap().to_str().unwrap();
    let dir = model_path
        .parent()
        .or_else(|| return Some(Path::new(".")))
        .unwrap();
    let model = BiRefNetConfig::new(ModelConfig::load(dir.join(format!("{model_name}.json")))?)
        .init::<Backend>(&device)
        .load_record(record);

    let img = image::open(image_path)?.into_rgb8();
    let size = img.dimensions();

    let img_tensor = preprocess(&img, WIDTH as u32, &device);
    let img_tensor = img_tensor.unsqueeze::<4>();
    let x = Normalizer::new(&device).normalize(img_tensor);

    let mask = model.forward(x);
    let mask = Sigmoid::new().forward(mask).squeeze::<3>(0);

    let mask = postprocess_mask(mask, size.0, size.1);
    let mask_img = img.estimate_foreground(&mask, 90).add_alpha_mask(&mask)?;

    mask_img.save("mask.png")?;

    Ok(())
}
