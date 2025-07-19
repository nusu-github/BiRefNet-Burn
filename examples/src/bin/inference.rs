//! BiRefNet Inference Example
//!
//! This example demonstrates how to perform inference with a trained BiRefNet model.
//! It supports both single image and batch processing.
//!
//! ## Usage
//!
//! ```bash
//! # Run inference on a single image
//! cargo run --bin inference -- model.mpk image.jpg
//!
//! # Run inference on a directory of images
//! cargo run --bin inference -- model.mpk input_dir/ --output output_dir/
//!
//! # Save only masks (no composite)
//! cargo run --bin inference -- model.mpk image.jpg --mask-only
//!
//! # Use custom threshold
//! cargo run --bin inference -- model.mpk image.jpg --threshold 0.3
//! ```

use anyhow::{Context, Result};
use birefnet_burn::{BiRefNet, BiRefNetConfig, CombinedLossConfig, ModelConfig};
use birefnet_examples::{
    common::{create_device, get_backend_name, SelectedBackend, SelectedDevice},
    postprocess_mask, tensor_to_image_data, InferenceConfig,
};
use burn::{
    nn::Sigmoid,
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
};
use clap::Parser;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the model file
    model: PathBuf,

    /// Path to the input image or directory
    input: PathBuf,

    /// Output directory for results
    #[arg(short, long, default_value = "outputs")]
    output: PathBuf,

    /// Image size for processing
    #[arg(long, default_value = "1024")]
    image_size: u32,

    /// Only save the mask (no composite)
    #[arg(long)]
    mask_only: bool,

    /// Threshold for binary mask (0.0-1.0)
    #[arg(short, long)]
    threshold: Option<f32>,

    /// Disable postprocessing
    #[arg(long)]
    no_postprocess: bool,

    /// Configuration file path
    #[arg(short, long)]
    config: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load configuration
    let mut config = if let Some(config_path) = &args.config {
        let config_str = fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;
        serde_json::from_str::<InferenceConfig>(&config_str)
            .with_context(|| format!("Failed to parse config file: {}", config_path.display()))?
    } else {
        InferenceConfig::default()
    };

    // Apply command line overrides
    config.image_size = args.image_size;
    config.output_path = args.output.clone();
    config.save_mask_only = args.mask_only;
    config.threshold = args.threshold;
    config.postprocess = !args.no_postprocess;

    // Validate inputs
    if !args.model.exists() {
        anyhow::bail!("Model file does not exist: {}", args.model.display());
    }

    if !args.input.exists() {
        anyhow::bail!("Input path does not exist: {}", args.input.display());
    }

    // Create output directory
    fs::create_dir_all(&config.output_path).with_context(|| {
        format!(
            "Failed to create output directory: {}",
            config.output_path.display()
        )
    })?;

    // Create device and load model
    let device = create_device();
    println!("Using backend: {}", get_backend_name());

    println!("Loading model from: {}", args.model.display());
    let model = load_model(&args.model, &device)?;

    // Process input
    if args.input.is_file() {
        // Single image
        println!("Processing single image: {}", args.input.display());
        process_single_image(&model, &args.input, &config, &device)?;
    } else if args.input.is_dir() {
        // Directory of images
        println!("Processing directory: {}", args.input.display());
        process_directory(&model, &args.input, &config, &device)?;
    } else {
        anyhow::bail!("Input must be a file or directory");
    }

    println!("Inference completed successfully!");
    Ok(())
}

/// Load the trained model
fn load_model(model_path: &Path, device: &SelectedDevice) -> Result<BiRefNet<SelectedBackend>> {
    let model_config = ModelConfig::new();
    let birefnet_config = BiRefNetConfig::new(model_config, CombinedLossConfig::new());

    // Initialize model
    let model = birefnet_config
        .init::<SelectedBackend>(device)
        .context("Failed to initialize model")?;

    // Load weights
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load(model_path.to_path_buf(), device)
        .context("Failed to load model weights")?;

    Ok(model.load_record(record))
}

/// Process a single image
fn process_single_image(
    model: &BiRefNet<SelectedBackend>,
    image_path: &Path,
    config: &InferenceConfig,
    device: &SelectedDevice,
) -> Result<()> {
    let start_time = Instant::now();

    // Load and preprocess image
    let input_tensor = load_and_preprocess_image(image_path, config.image_size, device)?;

    // Run inference
    let prediction = model.forward(input_tensor)?;

    // Apply sigmoid to get probability
    let sigmoid = Sigmoid::new();
    let probability = sigmoid.forward(prediction);

    // Postprocess if enabled
    let final_mask = if config.postprocess {
        postprocess_mask(
            probability,
            config.threshold.unwrap_or(0.5),
            5,    // blur_kernel_size
            1.0,  // blur_sigma
            3,    // morphology_kernel_size
            100,  // min_component_size
            true, // fill_holes
        )
    } else if let Some(threshold) = config.threshold {
        birefnet_examples::apply_threshold(probability, threshold)
    } else {
        probability
    };

    // Save result
    let output_path = config.output_path.join(
        image_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
            + "_mask.png",
    );

    save_mask_as_image(&final_mask, &output_path)?;

    let elapsed = start_time.elapsed();
    println!(
        "Processed {} in {:.2}s -> {}",
        image_path.display(),
        elapsed.as_secs_f32(),
        output_path.display()
    );

    Ok(())
}

/// Process a directory of images
fn process_directory(
    model: &BiRefNet<SelectedBackend>,
    input_dir: &Path,
    config: &InferenceConfig,
    device: &SelectedDevice,
) -> Result<()> {
    let entries = fs::read_dir(input_dir)
        .with_context(|| format!("Failed to read directory: {}", input_dir.display()))?;

    let mut processed_count = 0;
    let start_time = Instant::now();

    for entry in entries {
        let entry = entry.context("Failed to read directory entry")?;
        let path = entry.path();

        if path.is_file() {
            if let Some(extension) = path.extension() {
                let ext = extension.to_string_lossy().to_lowercase();
                if matches!(
                    ext.as_str(),
                    "jpg" | "jpeg" | "png" | "bmp" | "tiff" | "webp"
                ) {
                    match process_single_image(model, &path, config, device) {
                        Ok(_) => processed_count += 1,
                        Err(e) => {
                            eprintln!("Failed to process {}: {}", path.display(), e);
                        }
                    }
                }
            }
        }
    }

    let elapsed = start_time.elapsed();
    println!(
        "Processed {} images in {:.2}s (avg: {:.2}s per image)",
        processed_count,
        elapsed.as_secs_f32(),
        elapsed.as_secs_f32() / processed_count as f32
    );

    Ok(())
}

/// Load and preprocess an image
fn load_and_preprocess_image(
    _image_path: &Path,
    target_size: u32,
    device: &SelectedDevice,
) -> Result<Tensor<SelectedBackend, 4>> {
    // This is a simplified version - in practice you would use image loading libraries
    // like image-rs and implement proper preprocessing

    // For now, create a dummy tensor with the right shape
    let tensor = Tensor::random(
        [1, 3, target_size as usize, target_size as usize],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );

    Ok(tensor)
}

/// Save mask as image
fn save_mask_as_image(mask: &Tensor<SelectedBackend, 4>, output_path: &Path) -> Result<()> {
    // Convert tensor to image data
    let image_data = tensor_to_image_data(mask.clone());

    // In practice, you would use image-rs or similar library to save the image
    // For now, we'll just create a placeholder file
    fs::write(output_path, &image_data)
        .with_context(|| format!("Failed to save image: {}", output_path.display()))?;

    Ok(())
}
