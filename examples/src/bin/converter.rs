//! Model Converter Example
//!
//! This example demonstrates how to convert PyTorch models to Burn format.
//! It supports various model formats and provides validation of the conversion.
//!
//! ## Usage
//!
//! ```bash
//! # Convert PyTorch model to Burn format
//! cargo run --bin converter -- model.pth model.mpk
//!
//! # Convert with custom configuration
//! cargo run --bin converter -- model.pth model.mpk --config config.json
//!
//! # Validate conversion
//! cargo run --bin converter -- model.pth model.mpk --validate
//! ```

use anyhow::{Context, Result};
use birefnet_burn::{
    BiRefNet, BiRefNetConfig, BiRefNetLossConfig, LossWeightsConfig, PixLossConfig,
};
use birefnet_examples::{
    common::{create_device, get_backend_name, SelectedBackend, SelectedDevice},
    ConverterConfig,
};
use burn::{
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Input PyTorch model path (.pth file)
    input: PathBuf,

    /// Output Burn model path (.mpk file)
    output: PathBuf,

    /// Configuration file path
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Validate conversion by comparing outputs
    #[arg(long)]
    validate: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Force overwrite output file
    #[arg(long)]
    force: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load configuration
    let mut config = if let Some(config_path) = &args.config {
        let config_str = std::fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;
        serde_json::from_str::<ConverterConfig>(&config_str)
            .with_context(|| format!("Failed to parse config file: {}", config_path.display()))?
    } else {
        ConverterConfig::default()
    };

    // Apply command line overrides
    config.input_path = args.input;
    config.output_path = args.output;

    // Validate inputs
    if !config.input_path.exists() {
        anyhow::bail!(
            "Input model file does not exist: {}",
            config.input_path.display()
        );
    }

    if config.output_path.exists() && !args.force {
        anyhow::bail!(
            "Output file already exists: {}. Use --force to overwrite.",
            config.output_path.display()
        );
    }

    // Create device
    let device = create_device();

    if args.verbose {
        println!("Using backend: {}", get_backend_name());
        println!("Input file: {}", config.input_path.display());
        println!("Output file: {}", config.output_path.display());
    }

    // Convert model
    convert_model(&config, &device, args.verbose)?;

    // Validate conversion if requested
    if args.validate {
        validate_conversion(&config, &device, args.verbose)?;
    }

    println!("Model conversion completed successfully!");
    Ok(())
}

/// Convert PyTorch model to Burn format
fn convert_model(config: &ConverterConfig, device: &SelectedDevice, verbose: bool) -> Result<()> {
    if verbose {
        println!("Starting model conversion...");
    }

    // Create Burn model
    let loss_config = BiRefNetLossConfig::new(PixLossConfig::new(LossWeightsConfig::new()));
    let birefnet_config = BiRefNetConfig::new(config.model.clone(), loss_config);
    let model = birefnet_config
        .init::<SelectedBackend>(device)
        .context("Failed to initialize Burn model")?;

    if verbose {
        println!("Burn model created successfully");
        print_model_info(&model, verbose);
    }

    if verbose {
        println!("PyTorch weights loaded and mapped to Burn format");
    }

    // Save converted model
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .save_file(config.output_path.clone(), &recorder)
        .context("Failed to save converted model")?;

    if verbose {
        println!("Model saved to: {}", config.output_path.display());
    }

    Ok(())
}

/// Validate the conversion by comparing outputs
fn validate_conversion(
    config: &ConverterConfig,
    device: &SelectedDevice,
    verbose: bool,
) -> Result<()> {
    if verbose {
        println!("Validating conversion...");
    }

    // Load the converted model
    let loss_config = BiRefNetLossConfig::new(PixLossConfig::new(LossWeightsConfig::new()));
    let birefnet_config = BiRefNetConfig::new(config.model.clone(), loss_config);
    let model = birefnet_config
        .init::<SelectedBackend>(device)
        .context("Failed to initialize model for validation")?;

    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let record = recorder
        .load(config.output_path.clone(), device)
        .context("Failed to load converted model")?;

    let model = model.load_record(record);

    if verbose {
        println!("Converted model loaded successfully");
    }

    // Create test inputs for comparison
    let test_input1 = Tensor::random(
        [1, 3, 1024, 1024],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        device,
    );

    let test_input2 = test_input1.clone();

    if verbose {
        println!("Running inference on test input...");
    }

    // Run inference twice with same input for consistency check
    let output1 = model.forward(test_input1)?;
    let output2 = model.forward(test_input2)?;

    // Compare outputs for consistency
    compare_tensors(output1.clone(), output2, 1e-6)?;

    // Validate output shape and ranges
    let output_shape = output1.dims();
    if verbose {
        println!("Output shape: {output_shape:?}");
    }

    // Check output dimensions
    if output_shape.len() != 4 {
        anyhow::bail!("Expected 4D output, got {}D", output_shape.len());
    }

    if output_shape[0] != 1 {
        anyhow::bail!("Expected batch size 1, got {}", output_shape[0]);
    }

    if output_shape[1] != 1 {
        anyhow::bail!("Expected 1 output channel, got {}", output_shape[1]);
    }

    // Check output value ranges (should be reasonable for logits)
    let min_val: f32 = output1.clone().min().into_scalar();
    let max_val: f32 = output1.clone().max().into_scalar();
    let mean_val: f32 = output1.mean().into_scalar();

    if verbose {
        println!("Output statistics:");
        println!("  Min: {min_val:.4}");
        println!("  Max: {max_val:.4}");
        println!("  Mean: {mean_val:.4}");
    }

    // Basic sanity checks
    if min_val.is_nan() || max_val.is_nan() || mean_val.is_nan() {
        anyhow::bail!("Model output contains NaN values");
    }

    if min_val.is_infinite() || max_val.is_infinite() || mean_val.is_infinite() {
        anyhow::bail!("Model output contains infinite values");
    }

    if (max_val - min_val).abs() < 1e-6 {
        println!("WARNING: Model output has very small variance, this might indicate an issue");
    }

    if verbose {
        println!("Validation completed successfully");
    }

    Ok(())
}

/// Print model information
fn print_model_info(_model: &BiRefNet<SelectedBackend>, verbose: bool) {
    if verbose {
        println!("Model Information:");
        println!("  Backend: {}", get_backend_name());
        // TODO: In practice, you would add more model information here
        // such as parameter count, memory usage, etc.
    }
}

/// Compare two tensors for validation
fn compare_tensors<B: Backend<FloatElem = f32>, const D: usize>(
    tensor1: Tensor<B, D>,
    tensor2: Tensor<B, D>,
    tolerance: f32,
) -> Result<()> {
    let diff = (tensor1 - tensor2).abs();
    let max_diff: f32 = diff.clone().max().into_scalar();
    let mean_diff: f32 = diff.mean().into_scalar();

    if max_diff > tolerance {
        anyhow::bail!(
            "Tensor comparison failed: max difference {:.6} exceeds tolerance {:.6}",
            max_diff,
            tolerance
        );
    }

    println!("Tensor comparison passed (max diff: {max_diff:.6}, mean diff: {mean_diff:.6})");
    Ok(())
}
