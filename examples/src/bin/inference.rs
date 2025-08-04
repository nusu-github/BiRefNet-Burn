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
use birefnet_burn::{
    Backbone, BackboneConfig, BiRefNet, BiRefNetConfig, BiRefNetLossConfig, DecAtt, DecBlk,
    DecChannelsInter, DecoderConfig, LatBlk, LossWeightsConfig, ModelConfig, MulSclIpt,
    PixLossConfig, Prompt4loc, Refine, RefineConfig, SqueezeBlock, Task, TaskConfig,
};
use birefnet_examples::{
    common::{create_device, get_backend_name, image::ImageUtils, SelectedBackend, SelectedDevice},
    postprocess_mask, InferenceConfig,
};
use burn::{
    nn::Sigmoid,
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
};
use clap::Parser;
use image::GenericImageView;
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the model file or checkpoint folder
    model: PathBuf,

    /// Path to the input image or directory
    input: PathBuf,

    /// Output directory for results
    #[arg(short, long, default_value = "outputs")]
    output: PathBuf,

    /// Image size for processing (None for original resolution)
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

    /// Preserve original image resolution in output
    #[arg(long)]
    preserve_original_resolution: bool,

    /// Test sets to process (separated by '+')
    #[arg(long)]
    testsets: Option<String>,

    /// Mixed precision mode (fp16, bf16)
    #[arg(long)]
    mixed_precision: Option<String>,

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
    config.image_size = Some(args.image_size);
    config.output_path = args.output;
    config.save_mask_only = args.mask_only;
    config.threshold = args.threshold;
    config.postprocess = !args.no_postprocess;
    config.preserve_original_resolution = args.preserve_original_resolution;
    config.testsets = args.testsets;
    config.mixed_precision = args.mixed_precision;

    // Handle multiple checkpoints if model path is a directory
    if args.model.is_dir() {
        config.checkpoint_paths = collect_checkpoint_files(&args.model)?;
        if config.checkpoint_paths.is_empty() {
            anyhow::bail!(
                "No checkpoint files found in directory: {}",
                args.model.display()
            );
        }
    } else {
        config.checkpoint_paths = vec![args.model];
    }

    // Validate inputs
    if config.checkpoint_paths.is_empty() {
        anyhow::bail!("No model files to process");
    }

    for checkpoint_path in &config.checkpoint_paths {
        if !checkpoint_path.exists() {
            anyhow::bail!("Model file does not exist: {}", checkpoint_path.display());
        }
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

    // Create device
    let device = create_device();
    println!("Using backend: {}", get_backend_name());

    // Process each checkpoint
    for checkpoint_path in &config.checkpoint_paths {
        println!("Loading model from: {}", checkpoint_path.display());
        let model = load_model(checkpoint_path, &device)?;

        // Process testsets if specified, otherwise process input directly
        if let Some(testsets) = &config.testsets {
            for testset in testsets.split('+') {
                let testset = testset.trim();
                if testset.is_empty() {
                    continue;
                }
                println!(">>>> Processing testset: {testset}...");

                // Create testset-specific output directory
                let testset_output = config.output_path.join(format!(
                    "{}-{}",
                    checkpoint_path
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy(),
                    testset
                ));
                fs::create_dir_all(&testset_output)?;

                let mut testset_config = config.clone();
                testset_config.output_path = testset_output;

                process_input(&model, &args.input, &testset_config, &device)?;
            }
        } else {
            // Single processing
            process_input(&model, &args.input, &config, &device)?;
        }
    }

    println!("Inference completed successfully!");
    Ok(())
}

/// Collect checkpoint files from a directory
fn collect_checkpoint_files(checkpoint_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut checkpoint_files = Vec::new();

    let entries = fs::read_dir(checkpoint_dir).with_context(|| {
        format!(
            "Failed to read checkpoint directory: {}",
            checkpoint_dir.display()
        )
    })?;

    for entry in entries {
        let entry = entry.context("Failed to read directory entry")?;
        let path = entry.path();

        if path.is_file() {
            if let Some(extension) = path.extension() {
                let ext = extension.to_string_lossy().to_lowercase();
                if ext == "mpk" || ext == "pth" {
                    checkpoint_files.push(path);
                }
            }
        }
    }

    // Sort by epoch number if possible (similar to Python implementation)
    checkpoint_files.sort_by(|a, b| {
        let extract_epoch = |path: &Path| -> Option<i32> {
            path.file_stem()?
                .to_str()?
                .split("epoch_")
                .nth(1)?
                .split('.')
                .next()?
                .parse::<i32>()
                .ok()
        };

        match (extract_epoch(a), extract_epoch(b)) {
            (Some(epoch_a), Some(epoch_b)) => epoch_b.cmp(&epoch_a), // Reverse order (newest first)
            _ => b.cmp(a), // Fallback to filename comparison
        }
    });

    Ok(checkpoint_files)
}

/// Unified processing function for both single images and directories
fn process_input(
    model: &BiRefNet<SelectedBackend>,
    input_path: &Path,
    config: &InferenceConfig,
    device: &SelectedDevice,
) -> Result<()> {
    if input_path.is_file() {
        process_single_image(model, input_path, config, device)
    } else if input_path.is_dir() {
        process_directory(model, input_path, config, device)
    } else {
        anyhow::bail!("Input must be a file or directory")
    }
}

/// Load the trained model
fn load_model(model_path: &Path, device: &SelectedDevice) -> Result<BiRefNet<SelectedBackend>> {
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
    let birefnet_config = BiRefNetConfig::new(model_config, loss_config);

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

    // Get original image dimensions if preserve_original_resolution is enabled
    let original_dimensions = if config.preserve_original_resolution {
        Some(get_image_dimensions(image_path)?)
    } else {
        None
    };

    // Load and preprocess image
    let input_tensor = load_and_preprocess_image(image_path, config.image_size, device)?;

    // Run inference
    let prediction = model.forward(input_tensor)?;

    // Apply sigmoid to get probability
    let sigmoid = Sigmoid::new();
    let probability = sigmoid.forward(prediction);

    // Resize to original dimensions if requested
    let resized_probability = if let Some((orig_height, orig_width)) = original_dimensions {
        resize_tensor_to_original_size(probability, orig_height, orig_width, device)?
    } else {
        probability
    };

    // Postprocess if enabled
    let final_mask = if config.postprocess {
        postprocess_mask(
            resized_probability,
            config.threshold.unwrap_or(0.5),
            5,    // blur_kernel_size
            1.0,  // blur_sigma
            3,    // morphology_kernel_size
            100,  // min_component_size
            true, // fill_holes
        )
    } else if let Some(threshold) = config.threshold {
        birefnet_examples::apply_threshold(resized_probability, threshold)
    } else {
        resized_probability
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

    save_mask_as_image(final_mask, &output_path)?;

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

    // Pre-estimate capacity based on directory size (this is a heuristic)
    let estimated_capacity = entries.size_hint().0;
    let mut image_paths = Vec::with_capacity(estimated_capacity);

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
                    image_paths.push(path);
                }
            }
        }
    }

    if image_paths.is_empty() {
        println!("No image files found in directory");
        return Ok(());
    }

    let total_images = image_paths.len();
    let start_time = Instant::now();
    let mut processed_count = 0;

    // Process images in smaller batches to optimize memory usage
    const BATCH_SIZE: usize = 8;
    for batch in image_paths.chunks(BATCH_SIZE) {
        // Check if all images in batch have the same dimensions for efficient batch processing
        let can_batch_process = if batch.len() > 1 {
            check_images_same_size(batch, config.image_size).unwrap_or(false)
        } else {
            false
        };

        if can_batch_process && batch.len() > 1 {
            // Process as batch
            match process_image_batch(model, batch, config, device) {
                Ok(count) => processed_count += count,
                Err(e) => {
                    eprintln!(
                        "Batch processing failed: {e}, falling back to individual processing"
                    );
                    // Fall back to individual processing
                    for path in batch {
                        match process_single_image(model, path, config, device) {
                            Ok(_) => processed_count += 1,
                            Err(e) => eprintln!("Failed to process {}: {}", path.display(), e),
                        }
                    }
                }
            }
        } else {
            // Process individually
            for path in batch {
                match process_single_image(model, path, config, device) {
                    Ok(_) => processed_count += 1,
                    Err(e) => eprintln!("Failed to process {}: {}", path.display(), e),
                }
            }
        }
    }

    let elapsed = start_time.elapsed();
    println!(
        "Processed {}/{} images in {:.2}s (avg: {:.2}s per image)",
        processed_count,
        total_images,
        elapsed.as_secs_f32(),
        if processed_count > 0 {
            elapsed.as_secs_f32() / processed_count as f32
        } else {
            0.0
        }
    );

    Ok(())
}

/// Check if images have the same dimensions after resizing
fn check_images_same_size(image_paths: &[PathBuf], target_size: Option<u32>) -> Result<bool> {
    if image_paths.is_empty() {
        return Ok(true);
    }

    // If no target size specified, check if all images have the same original dimensions
    if target_size.is_none() {
        let first_dims = get_image_dimensions(&image_paths[0])?;
        for path in &image_paths[1..] {
            let dims = get_image_dimensions(path)?;
            if dims != first_dims {
                return Ok(false);
            }
        }
        return Ok(true);
    }

    // If target size is specified, all images will be resized to the same size
    // So they will definitely have the same dimensions after preprocessing
    Ok(true)
}

/// Process multiple images as a batch for better performance
fn process_image_batch(
    model: &BiRefNet<SelectedBackend>,
    image_paths: &[PathBuf],
    config: &InferenceConfig,
    device: &SelectedDevice,
) -> Result<usize> {
    if image_paths.is_empty() {
        return Ok(0);
    }

    let mut tensors = Vec::with_capacity(image_paths.len());
    for path in image_paths {
        let tensor = load_and_preprocess_image(path, config.image_size, device)?;
        tensors.push(tensor.squeeze::<3>(0));
    }
    let batch_tensor = Tensor::stack(tensors, 0);

    // Run batch inference
    let predictions = model.forward(batch_tensor)?;

    // Apply sigmoid to get probabilities
    let sigmoid = Sigmoid::new();
    let probabilities = sigmoid.forward(predictions);

    // Process each prediction in the batch
    let mut processed_count = 0;
    for (i, path) in image_paths.iter().enumerate() {
        // Extract single prediction from batch
        let single_prediction = probabilities.clone().slice(s![i..i + 1, 0..1, .., ..]);

        // Postprocess if enabled
        let final_mask = if config.postprocess {
            postprocess_mask(
                single_prediction,
                config.threshold.unwrap_or(0.5),
                5,    // blur_kernel_size
                1.0,  // blur_sigma
                3,    // morphology_kernel_size
                100,  // min_component_size
                true, // fill_holes
            )
        } else if let Some(threshold) = config.threshold {
            birefnet_examples::apply_threshold(single_prediction, threshold)
        } else {
            single_prediction
        };

        // Save result
        let output_path = config.output_path.join(
            path.file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string()
                + "_mask.png",
        );

        match save_mask_as_image(final_mask, &output_path) {
            Ok(_) => {
                processed_count += 1;
                println!(
                    "Processed (batch): {} -> {}",
                    path.display(),
                    output_path.display()
                );
            }
            Err(e) => {
                eprintln!("Failed to save {}: {}", output_path.display(), e);
            }
        }
    }

    Ok(processed_count)
}

/// Load and preprocess an image
fn load_and_preprocess_image(
    image_path: &Path,
    target_size: Option<u32>,
    device: &SelectedDevice,
) -> Result<Tensor<SelectedBackend, 4>> {
    let mut img = image::open(image_path)
        .with_context(|| format!("Failed to open image: {}", image_path.display()))?;

    if let Some(size) = target_size {
        let (width, height) = img.dimensions();
        if width != size || height != size {
            img = img.resize_exact(size, size, image::imageops::FilterType::Lanczos3);
        }
    }

    ImageUtils::dynamic_image_to_tensor(img, device)
}

/// Get original image dimensions
fn get_image_dimensions(image_path: &Path) -> Result<(usize, usize)> {
    let img = image::open(image_path)
        .with_context(|| format!("Failed to open image: {}", image_path.display()))?;
    Ok((img.height() as usize, img.width() as usize))
}

/// Resize tensor to original image size (similar to Python's interpolate)
fn resize_tensor_to_original_size(
    tensor: Tensor<SelectedBackend, 4>,
    target_height: usize,
    target_width: usize,
    device: &SelectedDevice,
) -> Result<Tensor<SelectedBackend, 4>> {
    birefnet_examples::resize_tensor(tensor, target_height, target_width, device)
}

/// Save mask as image
fn save_mask_as_image(mask: Tensor<SelectedBackend, 4>, output_path: &Path) -> Result<()> {
    // Convert tensor to DynamicImage using improved ImageUtils
    let dynamic_image = ImageUtils::tensor_to_dynamic_image(mask, true)
        .with_context(|| "Failed to convert tensor to image")?;

    // Save the image
    dynamic_image
        .save(output_path)
        .with_context(|| format!("Failed to save image: {}", output_path.display()))?;

    Ok(())
}
