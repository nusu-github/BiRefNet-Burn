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
pub mod training;

// Re-export specific configuration types instead of wildcard
#[doc(inline)]
pub use config::{
    Backbone, BackboneConfig, DecBlk, DecChannelsInter, DecoderAttention, DecoderConfig,
    InterpolationStrategy, LateralBlock, ModelConfig, MultiScaleInput, Optimizer, PathConfig,
    PreprocMethods, PromptForLocation, Refine, RefineConfig, SqueezeBlock, Task, TaskConfig,
};
#[doc(inline)]
pub use error::{BiRefNetError, BiRefNetResult};
#[doc(inline)]
pub use models::birefnet::{BiRefNet, BiRefNetConfig, BiRefNetRecord};
#[cfg(feature = "train")]
#[doc(inline)]
pub use training::{BiRefNetBatch, BiRefNetOutput};

#[cfg(test)]
mod tests {
    use burn::backend::{Autodiff, Cpu};

    pub type TestBackend = Cpu;

    pub type TestAutodiffBackend = Autodiff<TestBackend>;
}
