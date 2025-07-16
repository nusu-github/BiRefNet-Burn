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

mod config;
mod error;
mod models;
mod special;

#[cfg(feature = "train")]
mod dataset;

#[cfg(test)]
mod tests;

pub use config::*;
pub use error::{BiRefNetError, BiRefNetResult};
pub use models::{BiRefNet, BiRefNetConfig, BiRefNetRecord};

#[cfg(feature = "train")]
pub use dataset::{BiRefNetDataset, BiRefNetItem};
