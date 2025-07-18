//! Configuration module for BiRefNet.
//!
//! This module provides configuration structures and enums for the BiRefNet model.
//! It is organized into two main submodules:
//! - `core`: Contains the main configuration structures
//! - `enums`: Contains all enumeration types used in configurations

pub mod core;
pub mod enums;

// Re-export all configuration structures from core
pub use core::{BackboneConfig, DecoderConfig, ModelConfig, PathConfig, RefineConfig, TaskConfig};

// Re-export all enums from enums
pub use enums::{
    Backbone, DecAtt, DecBlk, DecChannelsInter, LatBlk, MulSclIpt, Optimizer, PreprocMethods,
    Prompt4loc, Refine, SqueezeBlock, Task,
};
