//! # Neural Network Building Blocks
//!
//! This module provides a collection of reusable neural network modules that form
//! the building blocks of the BiRefNet architecture, particularly the decoder.
//!
//! ## Modules
//!
//! - `aspp`: Implements Atrous Spatial Pyramid Pooling (ASPP) and its deformable variant,
//!   used for capturing multi-scale context.
//! - `decoder_blocks`: Contains the `BasicDecBlk` and `ResBlk` modules, which are the
//!   primary components for processing features within the decoder stages.
//! - `deform_conv`: Provides the `DeformableConv2d` layer, a key component for the
//!   deformable ASPP variant.
//! - `lateral_blocks`: Defines simple lateral connection blocks (`BasicLatBlk`) used to
//!   combine features from the encoder and decoder pathways.
//! - `utils`: Utility functions and enums for building layers.

pub mod aspp;
pub mod decoder_blocks;
pub mod deform_conv;
pub mod lateral_blocks;
pub mod simple_convs;
pub mod utils;

// Re-export specific types from each module instead of using wildcards
pub use aspp::{ASPPConfig, ASPPDeformable, ASPPDeformableConfig, ASPP};
pub use decoder_blocks::{BasicDecBlk, BasicDecBlkConfig, ResBlk, ResBlkConfig};
pub use deform_conv::DeformableConv2d;
pub use lateral_blocks::{BasicLatBlk, BasicLatBlkConfig};
pub use simple_convs::{SimpleConvs, SimpleConvsConfig};
