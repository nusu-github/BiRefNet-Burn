//! Dataset Testing Example
//!
//! This example demonstrates how to test dataset loading and visualization.
//! It's useful for debugging dataset issues and understanding data formats.
//!
//! ## Usage
//!
//! ```bash
//! # Test dataset loading
//! cargo run --bin dataset_test -- --dataset-path datasets/test
//!
//! # Test with visualization
//! cargo run --bin dataset_test -- --dataset-path datasets/test --visualize
//!
//! # Test specific number of samples
//! cargo run --bin dataset_test -- --dataset-path datasets/test --num-samples 5
//! ```

use anyhow::{Context, Result};
use birefnet_burn::{BiRefNetBatcher, BiRefNetDataset, ModelConfig};
use birefnet_examples::{
    common::{create_device, get_backend_name, SelectedBackend, SelectedDevice},
    DatasetTestConfig,
};
use burn::tensor::cast::ToElement;
use burn::{
    data::dataloader::{DataLoaderBuilder, Dataset},
    prelude::*,
};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to dataset directory
    #[arg(long, default_value = "datasets/test")]
    dataset_path: PathBuf,

    /// Number of samples to test
    #[arg(long, default_value = "10")]
    num_samples: usize,

    /// Enable visualization
    #[arg(long)]
    visualize: bool,

    /// Configuration file path
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Batch size for testing
    #[arg(long, default_value = "4")]
    batch_size: usize,

    /// Number of workers for data loading
    #[arg(long, default_value = "2")]
    num_workers: usize,

    /// Test data split (train/val/test)
    #[arg(long, default_value = "test")]
    split: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load configuration
    let mut config = if let Some(config_path) = &args.config {
        let config_str = std::fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;
        serde_json::from_str::<DatasetTestConfig>(&config_str)
            .with_context(|| format!("Failed to parse config file: {}", config_path.display()))?
    } else {
        DatasetTestConfig::default()
    };

    // Apply command line overrides
    config.dataset_path = args.dataset_path;
    config.num_samples = args.num_samples;
    config.visualize = args.visualize;

    // Validate inputs
    if !config.dataset_path.exists() {
        anyhow::bail!(
            "Dataset path does not exist: {}",
            config.dataset_path.display()
        );
    }

    // Create device
    let device = create_device();
    println!("Using backend: {}", get_backend_name());

    println!("Testing dataset loading...");
    println!("Dataset path: {}", config.dataset_path.display());
    println!("Split: {}", args.split);
    println!("Number of samples to test: {}", config.num_samples);

    // Create dataset
    let dataset = create_dataset(&config, &args.split, &device)?;

    // Test individual samples
    test_individual_samples(&dataset, &config)?;

    // Test data statistics
    test_data_statistics(&dataset, &config)?;

    // Test batch loading
    test_batch_loading(dataset, args.batch_size, args.num_workers)?;

    println!("Dataset testing completed successfully!");
    Ok(())
}

/// Create the dataset
fn create_dataset(
    config: &DatasetTestConfig,
    split: &str,
    device: &SelectedDevice,
) -> Result<BiRefNetDataset<SelectedBackend>> {
    let mut model_config = ModelConfig::new();
    model_config.path.data_root_dir = config.dataset_path.to_string_lossy().to_string();

    let dataset = BiRefNetDataset::<SelectedBackend>::new(&model_config, split, device)
        .context("Failed to create dataset")?;

    println!(
        "Dataset created successfully with {} samples",
        dataset.len()
    );
    Ok(dataset)
}

/// Test individual samples
fn test_individual_samples(
    dataset: &BiRefNetDataset<SelectedBackend>,
    config: &DatasetTestConfig,
) -> Result<()> {
    println!("\n=== Testing Individual Samples ===");

    let num_samples = config.num_samples.min(dataset.len());

    for i in 0..num_samples {
        let sample = dataset.get(i).context("Failed to get sample")?;

        println!("Sample {i}:");
        println!("  Image shape: {:?}", sample.image.dims());
        println!("  Mask shape: {:?}", sample.mask.dims());

        // Check data ranges
        let image_stats = calculate_tensor_stats(sample.image);
        let mask_stats = calculate_tensor_stats(sample.mask);

        println!(
            "  Image stats: min={:.4}, max={:.4}, mean={:.4}",
            image_stats.0, image_stats.1, image_stats.2
        );
        println!(
            "  Mask stats: min={:.4}, max={:.4}, mean={:.4}",
            mask_stats.0, mask_stats.1, mask_stats.2
        );

        // Validate data ranges
        if image_stats.0 < -3.0 || image_stats.1 > 3.0 {
            println!("  WARNING: Image values outside expected range [-3, 3]");
        }

        if mask_stats.0 < 0.0 || mask_stats.1 > 1.0 {
            println!("  WARNING: Mask values outside expected range [0, 1]");
        }
    }

    Ok(())
}

/// Test batch loading
fn test_batch_loading(
    dataset: BiRefNetDataset<SelectedBackend>,
    batch_size: usize,
    num_workers: usize,
) -> Result<()> {
    println!("\n=== Testing Batch Loading ===");

    let dataloader = DataLoaderBuilder::new(BiRefNetBatcher::<SelectedBackend>::new())
        .batch_size(batch_size)
        .shuffle(42)
        .num_workers(num_workers)
        .build(dataset);

    let mut batch_count = 0;
    let max_batches = 3; // Test first 3 batches

    for batch in dataloader.iter() {
        batch_count += 1;

        println!("Batch {batch_count}:");
        println!("  Images shape: {:?}", batch.images.dims());
        println!("  Masks shape: {:?}", batch.masks.dims());

        // Validate batch dimensions
        let [batch_images, channels, height, width] = batch.images.dims();
        let [batch_masks, mask_channels, _mask_height, _mask_width] = batch.masks.dims();

        if batch_images != batch_masks {
            println!("  ERROR: Batch size mismatch between images and masks");
        }

        if channels != 3 {
            println!("  WARNING: Expected 3 channels for images, got {channels}");
        }

        if mask_channels != 1 {
            println!("  WARNING: Expected 1 channel for masks, got {mask_channels}");
        }

        if height != width {
            println!("  WARNING: Non-square images: {height}x{width}");
        }

        if batch_count >= max_batches {
            break;
        }
    }

    println!("Batch loading test completed ({batch_count} batches tested)");
    Ok(())
}

/// Test data statistics
fn test_data_statistics(
    dataset: &BiRefNetDataset<SelectedBackend>,
    config: &DatasetTestConfig,
) -> Result<()> {
    println!("\n=== Testing Data Statistics ===");

    let mut image_stats = StatisticsAccumulator::new();
    let mut mask_stats = StatisticsAccumulator::new();

    let num_samples = config.num_samples.min(dataset.len());

    for i in 0..num_samples {
        let sample = dataset.get(i).context("Failed to get sample")?;

        let (img_min, img_max, img_mean) = calculate_tensor_stats(sample.image);
        let (mask_min, mask_max, mask_mean) = calculate_tensor_stats(sample.mask);

        image_stats.add(img_min, img_max, img_mean);
        mask_stats.add(mask_min, mask_max, mask_mean);
    }

    println!("Image statistics across {num_samples} samples:");
    println!(
        "  Min: {:.4} (avg: {:.4})",
        image_stats.min_val,
        image_stats.avg_min()
    );
    println!(
        "  Max: {:.4} (avg: {:.4})",
        image_stats.max_val,
        image_stats.avg_max()
    );
    println!("  Mean: avg={:.4}", image_stats.avg_mean());

    println!("Mask statistics across {num_samples} samples:");
    println!(
        "  Min: {:.4} (avg: {:.4})",
        mask_stats.min_val,
        mask_stats.avg_min()
    );
    println!(
        "  Max: {:.4} (avg: {:.4})",
        mask_stats.max_val,
        mask_stats.avg_max()
    );
    println!("  Mean: avg={:.4}", mask_stats.avg_mean());

    Ok(())
}

/// Calculate tensor statistics
fn calculate_tensor_stats<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> (f32, f32, f32) {
    let min_val = tensor.clone().min().into_scalar().to_f32();
    let max_val = tensor.clone().max().into_scalar().to_f32();
    let mean_val = tensor.mean().into_scalar().to_f32();

    (min_val, max_val, mean_val)
}

/// Statistics accumulator for aggregating across samples
struct StatisticsAccumulator {
    min_val: f32,
    max_val: f32,
    min_sum: f32,
    max_sum: f32,
    mean_sum: f32,
    count: usize,
}

impl StatisticsAccumulator {
    const fn new() -> Self {
        Self {
            min_val: f32::INFINITY,
            max_val: f32::NEG_INFINITY,
            min_sum: 0.0,
            max_sum: 0.0,
            mean_sum: 0.0,
            count: 0,
        }
    }

    fn add(&mut self, min: f32, max: f32, mean: f32) {
        self.min_val = self.min_val.min(min);
        self.max_val = self.max_val.max(max);
        self.min_sum += min;
        self.max_sum += max;
        self.mean_sum += mean;
        self.count += 1;
    }

    fn avg_min(&self) -> f32 {
        if self.count > 0 {
            self.min_sum / self.count as f32
        } else {
            0.0
        }
    }

    fn avg_max(&self) -> f32 {
        if self.count > 0 {
            self.max_sum / self.count as f32
        } else {
            0.0
        }
    }

    fn avg_mean(&self) -> f32 {
        if self.count > 0 {
            self.mean_sum / self.count as f32
        } else {
            0.0
        }
    }
}
