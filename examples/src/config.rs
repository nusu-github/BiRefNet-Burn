//! Configuration for BiRefNet examples.
//!
//! This module provides configuration structures for training, inference,
//! and other example applications.

use std::path::PathBuf;

use birefnet_burn::ModelConfig;
use serde::{Deserialize, Serialize};

/// Configuration for training examples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Model configuration.
    pub model: ModelConfig,
    /// Number of training epochs.
    pub num_epochs: usize,
    /// Batch size for training.
    pub batch_size: usize,
    /// Learning rate for optimization.
    pub learning_rate: f64,
    /// Path to training dataset.
    pub train_dataset_path: PathBuf,
    /// Path to validation dataset.
    pub val_dataset_path: PathBuf,
    /// Path to save model checkpoints.
    pub checkpoint_path: PathBuf,
    /// Checkpoint saving frequency (epochs).
    pub checkpoint_frequency: usize,
    /// Number of workers for data loading.
    pub num_workers: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::new(),
            num_epochs: 100,
            batch_size: 4,
            learning_rate: 1e-4,
            train_dataset_path: PathBuf::from("datasets/train"),
            val_dataset_path: PathBuf::from("datasets/val"),
            checkpoint_path: PathBuf::from("checkpoints"),
            checkpoint_frequency: 10,
            num_workers: 4,
        }
    }
}

/// Configuration for inference examples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Input image size (will be resized to this size). None means use original size.
    pub image_size: Option<u32>,
    /// Output path for results.
    pub output_path: PathBuf,
    /// Whether to save only the mask (without composite).
    pub save_mask_only: bool,
    /// Threshold for binary mask (None for soft mask).
    pub threshold: Option<f32>,
    /// Whether to apply postprocessing.
    pub postprocess: bool,
    /// Whether to preserve original resolution in output.
    pub preserve_original_resolution: bool,
    /// Multiple checkpoint paths to process.
    pub checkpoint_paths: Vec<PathBuf>,
    /// Multiple test sets to process (separated by '+').
    pub testsets: Option<String>,
    /// Mixed precision mode ("fp16", "bf16", or None).
    pub mixed_precision: Option<String>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            image_size: Some(1024),
            output_path: PathBuf::from("outputs"),
            save_mask_only: false,
            threshold: Some(0.5),
            postprocess: true,
            preserve_original_resolution: false,
            checkpoint_paths: Vec::new(),
            testsets: None,
            mixed_precision: None,
        }
    }
}

/// Configuration for dataset testing examples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetTestConfig {
    /// Path to dataset.
    pub dataset_path: PathBuf,
    /// Number of samples to test.
    pub num_samples: usize,
    /// Whether to visualize samples.
    pub visualize: bool,
}

impl Default for DatasetTestConfig {
    fn default() -> Self {
        Self {
            dataset_path: PathBuf::from("datasets/test"),
            num_samples: 10,
            visualize: true,
        }
    }
}

/// Configuration for model conversion examples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConverterConfig {
    /// Input model path (PyTorch .pth file).
    pub input_path: PathBuf,
    /// Output model path (Burn .mpk file).
    pub output_path: PathBuf,
    /// Model configuration.
    pub model: ModelConfig,
}

impl Default for ConverterConfig {
    fn default() -> Self {
        Self {
            input_path: PathBuf::from("models/model.pth"),
            output_path: PathBuf::from("models/model.mpk"),
            model: ModelConfig::new(),
        }
    }
}
