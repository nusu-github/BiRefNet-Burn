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
//! - `mlp`: (If present) Contains MLP-related structures.
//! - `utils`: Utility functions and enums for building layers.

mod aspp;
mod decoder_blocks;
mod deform_conv;
mod lateral_blocks;
mod mlp;
mod simple_convs;
pub mod utils;

pub use aspp::*;
pub use decoder_blocks::*;
pub use deform_conv::*;
pub use lateral_blocks::*;
pub use simple_convs::*;
