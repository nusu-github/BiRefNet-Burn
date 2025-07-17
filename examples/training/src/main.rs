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
//! # Train with default ndarray backend
//! cargo run --release
//!
//! # Train with WGPU backend
//! cargo run --release --features wgpu --no-default-features
//!
//! # Train with CUDA backend
//! cargo run --release --features cuda --no-default-features
//! ```

use anyhow::{bail, ensure, Context, Result};
use birefnet_burn::{
    BiRefNetBatch, BiRefNetBatcher, BiRefNetConfig, BiRefNetDataset, CombinedLossConfig,
    FMeasureMetric, IoUMetric, LossMetric, MAEMetric, ModelConfig,
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
use std::sync::Arc;

// Backend selection based on feature flags
#[cfg(feature = "ndarray")]
use burn::backend::ndarray::{NdArray, NdArrayDevice};
#[cfg(feature = "ndarray")]
type SelectedBackend = NdArray;
#[cfg(feature = "ndarray")]
type SelectedDevice = NdArrayDevice;

#[cfg(feature = "wgpu")]
use burn::backend::wgpu::{Wgpu, WgpuDevice};
#[cfg(feature = "wgpu")]
type SelectedBackend = Wgpu;
#[cfg(feature = "wgpu")]
type SelectedDevice = WgpuDevice;

#[cfg(feature = "cuda")]
use burn::backend::cuda::{Cuda, CudaDevice};
#[cfg(feature = "cuda")]
type SelectedBackend = Cuda;
#[cfg(feature = "cuda")]
type SelectedDevice = CudaDevice;

use std::path::PathBuf;

// Configuration for the training run
#[derive(Config)]
pub struct TrainingConfig {
    #[config(default = "64")]
    pub batch_size: usize,
    #[config(default = "10")]
    pub num_epochs: usize,
    #[config(default = "0.001")]
    pub learning_rate: f64,
    #[config(default = "1e-4")]
    pub weight_decay: f64,
    #[config(default = "1.0")]
    pub bce_weight: f32,
    #[config(default = "1.0")]
    pub iou_weight: f32,
    #[config(default = "4")]
    pub num_workers: usize,
    #[config(default = "false")]
    pub use_tui: bool,
    pub dataset_path: PathBuf,
    pub checkpoint_dir: PathBuf,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 4, // Smaller batch size for segmentation
            num_epochs: 100,
            learning_rate: 1e-4,
            weight_decay: 1e-4,
            bce_weight: 1.0,
            iou_weight: 1.0,
            num_workers: 4,
            use_tui: true,
            dataset_path: PathBuf::from("dataset"),
            checkpoint_dir: PathBuf::from("checkpoints"),
        }
    }
}

/// Creates the appropriate device based on the selected backend
const fn create_device() -> SelectedDevice {
    #[cfg(feature = "ndarray")]
    {
        NdArrayDevice::Cpu
    }

    #[cfg(feature = "wgpu")]
    {
        WgpuDevice::default()
    }

    #[cfg(feature = "cuda")]
    {
        CudaDevice::default()
    }
}

/// Gets the backend name for logging purposes
const fn get_backend_name() -> &'static str {
    #[cfg(feature = "ndarray")]
    {
        "NdArray (CPU)"
    }

    #[cfg(feature = "wgpu")]
    {
        "WGPU (GPU)"
    }

    #[cfg(feature = "cuda")]
    {
        "CUDA (NVIDIA GPU)"
    }
}

fn main() -> Result<()> {
    // Load configuration
    let config = TrainingConfig::default();

    // Validate configuration
    ensure!(config.batch_size > 0, "Batch size must be greater than 0");
    ensure!(
        config.num_epochs > 0,
        "Number of epochs must be greater than 0"
    );
    ensure!(config.learning_rate > 0.0, "Learning rate must be positive");
    ensure!(
        config.weight_decay >= 0.0,
        "Weight decay must be non-negative"
    );
    ensure!(config.bce_weight > 0.0, "BCE weight must be positive");
    ensure!(config.iou_weight > 0.0, "IoU weight must be positive");

    if !config.dataset_path.exists() {
        bail!(
            "Dataset path does not exist: {}",
            config.dataset_path.display()
        );
    }

    println!("Starting BiRefNet training with configuration:");
    println!("  Batch size: {}", config.batch_size);
    println!("  Number of epochs: {}", config.num_epochs);
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Weight decay: {}", config.weight_decay);
    println!("  Dataset path: {}", config.dataset_path.display());
    println!(
        "  Checkpoint directory: {}",
        config.checkpoint_dir.display()
    );

    std::fs::create_dir_all(&config.checkpoint_dir).with_context(|| {
        format!(
            "Failed to create checkpoint directory at {}",
            config.checkpoint_dir.display()
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
    let optimizer_config = AdamWConfig::new().with_weight_decay(config.weight_decay as f32);

    let learner = LearnerBuilder::new(&config.checkpoint_dir)
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
    let model_config = BiRefNetConfig::new(base_config, CombinedLossConfig::new());

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
    model_config_for_dataset.path.data_root_dir = config.dataset_path.clone();

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
    let valid_dataset =
        BiRefNetDataset::<SelectedBackend>::new(&model_config_for_dataset, "val", device)
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
    let final_model_path = config.checkpoint_dir.join("final_model.bin");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.batch_size, 4);
        assert_eq!(config.num_epochs, 100);
        assert_eq!(config.learning_rate, 1e-4);
        assert_eq!(config.weight_decay, 1e-4);
        assert_eq!(config.bce_weight, 1.0);
        assert_eq!(config.iou_weight, 1.0);
        assert_eq!(config.num_workers, 4);
        assert!(config.use_tui);
        assert_eq!(config.dataset_path, PathBuf::from("dataset"));
        assert_eq!(config.checkpoint_dir, PathBuf::from("checkpoints"));
    }
}
