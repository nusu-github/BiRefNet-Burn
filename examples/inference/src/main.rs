// NOT EDITED recursion_limit
// wgpu backends requires a higher recursion limit
#![recursion_limit = "256"]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use birefnet_burn::{BiRefNet, BiRefNetConfig, ModelConfig};
use burn::{
    nn::Sigmoid,
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
};
use clap::Parser;
use imageops_kit::{ApplyAlphaMaskExt, ForegroundEstimationExt};

use crate::utils::{apply_threshold_mask, postprocess_mask, preprocess, Normalizer};

mod utils;

#[derive(Debug, Clone)]
struct InferenceConfig {
    pub image_size: u32,
    pub output_path: PathBuf,
    pub save_mask_only: bool,
    pub threshold: Option<f32>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            image_size: 1024,
            output_path: PathBuf::from("mask.png"),
            save_mask_only: false,
            threshold: Some(0.5),
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the model file
    model: PathBuf,

    /// Path to the input image or directory
    image: PathBuf,

    /// Batch size for processing multiple images
    #[arg(short, long, default_value = "1")]
    batch_size: usize,

    /// Output path for the result
    #[arg(short, long, default_value = "mask.png")]
    output: PathBuf,

    /// Image size for processing
    #[arg(short, long, default_value = "1024")]
    size: u32,

    /// Save only the mask without applying to original image
    #[arg(long)]
    mask_only: bool,

    /// Threshold for mask binarization (0.0-1.0)
    #[arg(short, long, default_value = "0.5")]
    threshold: f32,

    /// Disable threshold and keep original mask values
    #[arg(long)]
    no_threshold: bool,
}

fn validate_args(args: &Args) -> Result<()> {
    // Validate model file exists
    if !args.model.exists() {
        return Err(anyhow::anyhow!(
            "Model file does not exist: {:?}",
            args.model
        ));
    }

    // Validate image file or directory exists
    if !args.image.exists() {
        return Err(anyhow::anyhow!(
            "Image file or directory does not exist: {:?}",
            args.image
        ));
    }

    // Validate image size
    if args.size < 64 || args.size > 4096 {
        return Err(anyhow::anyhow!(
            "Image size must be between 64 and 4096, got: {}",
            args.size
        ));
    }

    // Validate threshold (only if not disabled)
    if !args.no_threshold && (args.threshold < 0.0 || args.threshold > 1.0) {
        return Err(anyhow::anyhow!(
            "Threshold must be between 0.0 and 1.0, got: {}",
            args.threshold
        ));
    }

    Ok(())
}

fn load_model<B: Backend>(model_path: &Path, device: &Device<B>) -> Result<BiRefNet<B>> {
    let record = NamedMpkFileRecorder::<FullPrecisionSettings>::default()
        .load(model_path.to_path_buf(), device)
        .with_context(|| format!("Failed to load model from {:?}", model_path))?;

    let model_name = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .with_context(|| format!("Invalid model file name: {:?}", model_path))?;

    let dir = model_path.parent().unwrap_or(Path::new("."));

    let config_path = dir.join(format!("{model_name}.json"));
    if !config_path.exists() {
        return Err(anyhow::anyhow!(
            "Model config file does not exist: {:?}",
            config_path
        ));
    }

    let model = BiRefNetConfig::new(ModelConfig::load(config_path)?)
        .init::<B>(device)?
        .load_record(record);

    Ok(model)
}

fn discover_images(path: &Path) -> Result<Vec<PathBuf>> {
    let mut images = Vec::new();

    if path.is_file() {
        // Single image file
        images.push(path.to_path_buf());
    } else if path.is_dir() {
        // Directory of images
        let entries =
            fs::read_dir(path).with_context(|| format!("Failed to read directory: {:?}", path))?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Some(extension) = path.extension() {
                    match extension.to_str() {
                        Some("jpg") | Some("jpeg") | Some("png") | Some("bmp") | Some("tiff")
                        | Some("webp") => {
                            images.push(path);
                        }
                        _ => {}
                    }
                }
            }
        }

        images.sort();
    }

    if images.is_empty() {
        return Err(anyhow::anyhow!("No valid image files found in: {:?}", path));
    }

    Ok(images)
}

fn process_batch_images<B: Backend>(
    image_paths: &[PathBuf],
    model: &BiRefNet<B>,
    config: &InferenceConfig,
    device: &Device<B>,
    batch_size: usize,
) -> Result<()> {
    let batch_start_time = Instant::now();

    println!(
        "Processing {} images in batches of {}",
        image_paths.len(),
        batch_size
    );

    for (batch_idx, batch_paths) in image_paths.chunks(batch_size).enumerate() {
        println!(
            "Processing batch {}/{}",
            batch_idx + 1,
            image_paths.len().div_ceil(batch_size)
        );

        for (i, image_path) in batch_paths.iter().enumerate() {
            let output_path = if image_paths.len() == 1 {
                config.output_path.clone()
            } else {
                // Generate unique output name for each image
                let stem = image_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("image");
                let extension = config
                    .output_path
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("png");

                if let Some(parent) = config.output_path.parent() {
                    parent.join(format!("{}_mask.{}", stem, extension))
                } else {
                    PathBuf::from(format!("{}_mask.{}", stem, extension))
                }
            };

            let image_config = InferenceConfig {
                output_path,
                ..config.clone()
            };

            process_image::<B>(image_path, model, &image_config, device).with_context(|| {
                format!(
                    "Failed to process image {} in batch {}",
                    i + 1,
                    batch_idx + 1
                )
            })?;
        }
    }

    let batch_total_time = batch_start_time.elapsed();

    println!("Successfully processed all {} images", image_paths.len());
    println!(
        "Total batch processing time: {:.2}s ({:.2}s per image)",
        batch_total_time.as_secs_f64(),
        batch_total_time.as_secs_f64() / image_paths.len() as f64
    );

    Ok(())
}

fn process_image<B: Backend>(
    image_path: &Path,
    model: &BiRefNet<B>,
    config: &InferenceConfig,
    device: &Device<B>,
) -> Result<()> {
    let start_time = Instant::now();

    let img = image::open(image_path)
        .with_context(|| format!("Failed to open image: {:?}", image_path))?
        .into_rgb8();
    let original_size = img.dimensions();

    // Preprocess image
    let preprocess_start = Instant::now();
    let img_tensor = preprocess(&img, config.image_size, device);
    let img_tensor = img_tensor.unsqueeze::<4>();
    let x = Normalizer::new(device).normalize(img_tensor);
    let preprocess_time = preprocess_start.elapsed();

    // Run inference
    let inference_start = Instant::now();
    let mask = model.forward(x)?;
    let mask = Sigmoid::new().forward(mask).squeeze::<3>(0);
    let inference_time = inference_start.elapsed();

    // Postprocess mask
    let postprocess_start = Instant::now();
    let mask = postprocess_mask(mask, original_size.0, original_size.1);

    // Convert f32 mask to u8 mask with threshold
    let mask_u8 = apply_threshold_mask(&mask, config.threshold);
    let postprocess_time = postprocess_start.elapsed();

    if config.save_mask_only {
        mask_u8
            .save(&config.output_path)
            .with_context(|| format!("Failed to save mask to {:?}", config.output_path))?;
    } else {
        let mask_img = img
            .estimate_foreground_colors(&mask_u8, 91)
            .with_context(|| "Failed to estimate foreground colors")?;
        let mask_img = mask_img
            .apply_alpha_mask(&mask_u8)
            .with_context(|| "Failed to apply alpha mask")?;
        mask_img
            .save(&config.output_path)
            .with_context(|| format!("Failed to save result to {:?}", config.output_path))?;
    }

    let total_time = start_time.elapsed();

    println!("Successfully processed image: {:?}", image_path);
    println!("Output saved to: {:?}", config.output_path);
    println!(
        "Processing time: {:.2}s (preprocess: {:.2}s, inference: {:.2}s, postprocess: {:.2}s)",
        total_time.as_secs_f64(),
        preprocess_time.as_secs_f64(),
        inference_time.as_secs_f64(),
        postprocess_time.as_secs_f64()
    );

    if config.threshold.is_some() {
        println!("Threshold: {:.2}", config.threshold.unwrap());
    } else {
        println!("Threshold: disabled (preserving original values)");
    }

    Ok(())
}

fn main() -> Result<()> {
    type Backend = burn::backend::NdArray;
    // type Backend = burn::backend::wgpu::Wgpu;
    // type Backend = burn::backend::cuda::Cuda;

    let device = Default::default();
    let args = Args::parse();

    // Validate arguments
    validate_args(&args)?;

    // Create inference configuration
    let config = InferenceConfig {
        image_size: args.size,
        output_path: args.output,
        save_mask_only: args.mask_only,
        threshold: if args.no_threshold {
            None
        } else {
            Some(args.threshold)
        },
    };

    // Load model
    let model = load_model::<Backend>(&args.model, &device)?;

    // Discover images to process
    let image_paths = discover_images(&args.image)?;

    // Process images
    if image_paths.len() == 1 {
        process_image::<Backend>(&image_paths[0], &model, &config, &device)?;
    } else {
        process_batch_images::<Backend>(&image_paths, &model, &config, &device, args.batch_size)?;
    }

    Ok(())
}
