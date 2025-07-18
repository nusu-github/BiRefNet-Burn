//! # Model Architectures
//!
//! This module aggregates the core components of the BiRefNet model architecture.
//! It is organized into sub-modules for clarity:
//!
//! - `backbones`: Contains the implementation of backbone networks like Swin Transformer.
//! - `birefnet`: Defines the main `BiRefNet` model, which integrates the backbone and decoder.
//! - `modules`: Provides various neural network building blocks, such as ASPP, decoder blocks,
//!   and lateral connection blocks.
//!
//! The components are re-exported for easy access from the parent `models` module.

pub mod birefnet;
pub mod decoder;
pub mod modules;
