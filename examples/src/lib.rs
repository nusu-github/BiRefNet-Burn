//! BiRefNet Examples
//!
//! This crate provides example applications for the BiRefNet model,
//! including training, inference, dataset testing, and model conversion.
//!
//! ## Available Examples
//!
//! - `train`: Training pipeline with loss functions and metrics
//! - `inference`: Model inference on single images or batches
//! - `dataset_test`: Dataset loading and visualization utilities
//! - `converter`: Convert PyTorch models to Burn format
//!
//! ## Usage
//!
//! ```bash
//! # Train a model
//! cargo run --bin train -- --config config.json
//!
//! # Run inference
//! cargo run --bin inference -- model.mpk image.jpg
//!
//! # Test dataset loading
//! cargo run --bin dataset_test -- --dataset-path datasets/test
//!
//! # Convert PyTorch model
//! cargo run --bin converter -- model.pth model.mpk
//! ```

pub mod common;
pub mod config;

// Re-export commonly used items
pub use common::{
    apply_threshold, create_device, fill_holes, gaussian_blur, get_backend_name,
    morphological_closing, morphological_opening, postprocess_mask, remove_small_components,
    resize_tensor, tensor_to_image_data, SelectedBackend, SelectedDevice,
};

// Re-export metrics from birefnet
pub use birefnet_burn::{
    calculate_all_metrics, calculate_f_measure, calculate_iou, calculate_mae, MetricsAggregator,
};
pub use config::{ConverterConfig, DatasetTestConfig, InferenceConfig, TrainingConfig};
