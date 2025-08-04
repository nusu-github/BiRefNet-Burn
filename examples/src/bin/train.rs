//! BiRefNet Training Example
//!
//! This example demonstrates how to train a BiRefNet model using the Burn framework.
//! It implements a complete training pipeline with loss functions, metrics, and checkpointing.
//!
//! ## Features
//!
//! This training example supports multiple backends through feature flags:
//! - `ndarray`: CPU backend using ndarray (default)
//! - `wgpu`: GPU backend using WGPU
//! - `cuda`: NVIDIA GPU backend using CUDA
//! - `tch`: LibTorch backend
//! - `candle`: Candle backend
//!
//! ## Usage
//!
//! ```bash
//! # Train with default configuration
//! cargo run --bin train
//!
//! # Train with specific configuration file
//! cargo run --bin train -- --config train_config.json
//!
//! # Train with WGPU backend
//! cargo run --bin train --features wgpu --no-default-features
//! ```

use anyhow::{bail, ensure, Context, Result};
use birefnet_burn::{
    BiRefNetBatch, BiRefNetBatcher, BiRefNetConfig, BiRefNetDataset, BiRefNetLossConfig,
    FMeasureMetric, IoUMetric, LossMetric, LossWeightsConfig, MAEMetric, ModelConfig,
    PixLossConfig,
};
use birefnet_examples::{
    common::{create_device, get_backend_name, SelectedBackend, SelectedDevice},
    TrainingConfig,
};
use burn::{
    backend::Autodiff,
    data::dataloader::{DataLoader, DataLoaderBuilder, Dataset},
    optim::AdamWConfig,
    prelude::*,
    record::FullPrecisionSettings,
    record::NamedMpkFileRecorder,
    train::LearnerBuilder,
};
use clap::Parser;
use std::{path::PathBuf, sync::Arc};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Configuration file path
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Override batch size
    #[arg(long)]
    batch_size: Option<usize>,

    /// Override number of epochs
    #[arg(long)]
    num_epochs: Option<usize>,

    /// Override learning rate
    #[arg(long)]
    learning_rate: Option<f64>,

    /// Override dataset path
    #[arg(long)]
    dataset_path: Option<PathBuf>,

    /// Override checkpoint path
    #[arg(long)]
    checkpoint_path: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Load configuration
    let mut config = if let Some(config_path) = &args.config {
        let config_str = std::fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;
        serde_json::from_str::<TrainingConfig>(&config_str)
            .with_context(|| format!("Failed to parse config file: {}", config_path.display()))?
    } else {
        TrainingConfig::default()
    };

    // Apply command line overrides
    if let Some(batch_size) = args.batch_size {
        config.batch_size = batch_size;
    }
    if let Some(num_epochs) = args.num_epochs {
        config.num_epochs = num_epochs;
    }
    if let Some(learning_rate) = args.learning_rate {
        config.learning_rate = learning_rate;
    }
    if let Some(dataset_path) = args.dataset_path {
        config.train_dataset_path = dataset_path.clone();
        config.val_dataset_path = dataset_path.join("val");
    }
    if let Some(checkpoint_path) = args.checkpoint_path {
        config.checkpoint_path = checkpoint_path;
    }

    // Validate configuration
    ensure!(config.batch_size > 0, "Batch size must be greater than 0");
    ensure!(
        config.num_epochs > 0,
        "Number of epochs must be greater than 0"
    );
    ensure!(config.learning_rate > 0.0, "Learning rate must be positive");

    if !config.train_dataset_path.exists() {
        bail!(
            "Training dataset path does not exist: {}",
            config.train_dataset_path.display()
        );
    }

    if !config.val_dataset_path.exists() {
        bail!(
            "Validation dataset path does not exist: {}",
            config.val_dataset_path.display()
        );
    }

    println!("Starting BiRefNet training with configuration:");
    println!("  Batch size: {}", config.batch_size);
    println!("  Number of epochs: {}", config.num_epochs);
    println!("  Learning rate: {}", config.learning_rate);
    println!(
        "  Training dataset: {}",
        config.train_dataset_path.display()
    );
    println!(
        "  Validation dataset: {}",
        config.val_dataset_path.display()
    );
    println!(
        "  Checkpoint directory: {}",
        config.checkpoint_path.display()
    );

    std::fs::create_dir_all(&config.checkpoint_path).with_context(|| {
        format!(
            "Failed to create checkpoint directory at {}",
            config.checkpoint_path.display()
        )
    })?;

    // Create device
    let device = create_device();
    println!("Using backend: {}", get_backend_name());

    // Create and initialize model
    println!("Creating BiRefNet model...");
    let model = create_model(&device)?;

    // Create datasets
    let (train_dataset, valid_dataset) = create_datasets(&config, &device)?;

    // Create data loaders
    let (train_dataloader, valid_dataloader) =
        create_dataloaders(&config, train_dataset, valid_dataset);

    // Create learner with optimizer and metrics
    let optimizer_config = AdamWConfig::new();

    let learner = LearnerBuilder::new(&config.checkpoint_path)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(FMeasureMetric::new())
        .metric_valid_numeric(MAEMetric::new())
        .metric_valid_numeric(IoUMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(model, optimizer_config.init(), config.learning_rate);

    // Start training
    println!("Starting training...");
    let model_trained = learner.fit(train_dataloader, valid_dataloader);

    // Save final model
    save_final_model(&config, model_trained)?;

    println!("Training completed successfully!");

    Ok(())
}

/// Creates and initializes the BiRefNet model
fn create_model(
    device: &SelectedDevice,
) -> Result<birefnet_burn::BiRefNet<Autodiff<SelectedBackend>>> {
    let base_config = ModelConfig::new();
    let loss_config = BiRefNetLossConfig::new(PixLossConfig::new(LossWeightsConfig::new()));
    let model_config = BiRefNetConfig::new(base_config, loss_config);

    let model = model_config
        .init::<Autodiff<SelectedBackend>>(device)
        .context("Failed to initialize BiRefNet model")?;

    Ok(model)
}

/// Creates training and validation datasets
fn create_datasets(
    config: &TrainingConfig,
    device: &SelectedDevice,
) -> Result<(
    BiRefNetDataset<Autodiff<SelectedBackend>>,
    BiRefNetDataset<SelectedBackend>,
)> {
    println!("Loading training dataset...");
    let mut model_config_for_dataset = ModelConfig::new();
    model_config_for_dataset.path.data_root_dir =
        config.train_dataset_path.to_string_lossy().to_string();

    let train_dataset = BiRefNetDataset::<Autodiff<SelectedBackend>>::new(
        &model_config_for_dataset,
        "train",
        device,
    )
    .context("Failed to create training dataset")?;
    println!(
        "Training dataset loaded with {} samples",
        train_dataset.len()
    );

    println!("Loading validation dataset...");
    let mut model_config_for_valid = ModelConfig::new();
    model_config_for_valid.path.data_root_dir =
        config.val_dataset_path.to_string_lossy().to_string();

    let valid_dataset =
        BiRefNetDataset::<SelectedBackend>::new(&model_config_for_valid, "val", device)
            .context("Failed to create validation dataset")?;
    println!(
        "Validation dataset loaded with {} samples",
        valid_dataset.len()
    );

    Ok((train_dataset, valid_dataset))
}

/// Creates training and validation data loaders
fn create_dataloaders(
    config: &TrainingConfig,
    train_dataset: BiRefNetDataset<Autodiff<SelectedBackend>>,
    valid_dataset: BiRefNetDataset<SelectedBackend>,
) -> (
    Arc<dyn DataLoader<Autodiff<SelectedBackend>, BiRefNetBatch<Autodiff<SelectedBackend>>>>,
    Arc<dyn DataLoader<SelectedBackend, BiRefNetBatch<SelectedBackend>>>,
) {
    let train_dataloader = DataLoaderBuilder::new(BiRefNetBatcher::new())
        .batch_size(config.batch_size)
        .shuffle(42) // Seed for reproducibility
        .num_workers(config.num_workers)
        .build(train_dataset);

    let valid_dataloader = DataLoaderBuilder::new(BiRefNetBatcher::<SelectedBackend>::new())
        .batch_size(config.batch_size)
        .shuffle(42) // Seed for reproducibility
        .num_workers(config.num_workers)
        .build(valid_dataset);

    (train_dataloader, valid_dataloader)
}

/// Saves the final trained model
fn save_final_model(
    config: &TrainingConfig,
    model: birefnet_burn::BiRefNet<Autodiff<SelectedBackend>>,
) -> Result<()> {
    let final_model_path = config.checkpoint_path.join("final_model.mpk");
    println!("Saving final model to: {}", final_model_path.display());

    model
        .save_file::<NamedMpkFileRecorder<FullPrecisionSettings>, &PathBuf>(
            &final_model_path,
            &burn::record::DefaultFileRecorder::new(),
        )
        .with_context(|| {
            format!(
                "Failed to save final model to {}",
                final_model_path.display()
            )
        })?;

    Ok(())
}
