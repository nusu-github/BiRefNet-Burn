//! # BiRefNet-Burn
//!
//! This crate provides a Rust implementation of the BiRefNet (Bilateral Reference Network)
//! for dichotomous image segmentation, built using the Burn deep learning framework.
//!
//! ## Modules
//!
//! - `config`: Contains all configuration structures for the model, allowing for
//!   fine-grained control over the architecture and hyperparameters.
//! - `error`: Defines the custom error types used throughout the crate.
//! - `models`: Implements the core model architecture, including the backbone,
//!   decoder, and various sub-modules.
//! - `special`: Contains special-purpose modules and functions, such as `DropPath`
//!   and custom tensor operations.
//! - `tests`: Includes unit and integration tests to ensure correctness.
//!
//! ## Key Components
//!
//! - `BiRefNet`: The main model struct.
//! - `ModelConfig`: The primary configuration struct that drives the model's construction.
//! - `BiRefNetError`: The enum for all possible errors.

mod config; // Now organized as a module with core.rs and enums.rs
mod error;
mod models;

#[cfg(feature = "train")]
mod dataset;

#[cfg(feature = "train")]
mod losses;

#[cfg(feature = "train")]
mod metrics;

#[cfg(feature = "train")]
mod training;

#[cfg(test)]
mod tests;

// Re-export specific configuration types instead of wildcard
pub use config::{
    Backbone, BackboneConfig, DecAtt, DecBlk, DecChannelsInter, DecoderConfig, LatBlk, ModelConfig,
    MulSclIpt, Optimizer, PathConfig, PreprocMethods, Prompt4loc, Refine, RefineConfig,
    SqueezeBlock, Task, TaskConfig,
};
pub use error::{BiRefNetError, BiRefNetResult};
pub use models::birefnet::{BiRefNet, BiRefNetConfig};

#[cfg(feature = "train")]
pub use dataset::{BiRefNetBatch, BiRefNetBatcher, BiRefNetDataset, BiRefNetItem};

#[cfg(feature = "train")]
pub use losses::{
    BiRefNetLossConfig, ClsLoss, ClsLossConfig, CombinedLoss, CombinedLossConfig, ContourLoss,
    ContourLossConfig, LossWeightsConfig, MaeLoss, MaeLossConfig, MseLoss, MseLossConfig,
    MultiScaleLoss, MultiScaleLossConfig, PatchIoULoss, PatchIoULossConfig, PixLoss, PixLossConfig,
    SSIMLoss, SSIMLossConfig, StructureLoss, StructureLossConfig, ThrRegLoss, ThrRegLossConfig,
};

#[cfg(feature = "train")]
pub use metrics::{
    calculate_all_metrics, calculate_e_measure, calculate_f_measure, calculate_iou, calculate_mae,
    calculate_mse, calculate_s_measure, calculate_weighted_f_measure, AllMetricsResult,
    EMeasureMetric, FMeasureMetric, IoUMetric, LossMetric, MAEMetric, MSEMetric, MetricsAggregator,
    SMeasureMetric, WeightedFMeasureMetric,
};

#[cfg(feature = "train")]
pub use training::BiRefNetOutput;
