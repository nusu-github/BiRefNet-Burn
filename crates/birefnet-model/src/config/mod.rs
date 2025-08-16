//! Configuration module for BiRefNet.
//!
//! This module provides configuration structures and enums for the BiRefNet model.
//! It is organized into two main submodules:
//! - `core`: Contains the main configuration structures
//! - `enums`: Contains all enumeration types used in configurations

mod core;
mod enums;

// Re-export all configuration structures from core
pub use core::{BackboneConfig, DecoderConfig, ModelConfig, PathConfig, RefineConfig, TaskConfig};

// Re-export all enums from enums
pub use enums::{
    Backbone, DecBlk, DecChannelsInter, DecoderAttention, InterpolationStrategy, LateralBlock,
    MultiScaleInput, Optimizer, PreprocMethods, PromptForLocation, Refine, SqueezeBlock, Task,
};
