//! # Backbone Networks
//!
//! This module provides the backbone architectures used for feature extraction in BiRefNet.
//!
//! ## Components
//!
//! - `build_backbone`: A factory function that constructs a backbone model based on the
//!   provided configuration. It currently supports various versions of the Swin Transformer, ResNet, and VGG.
//! - `swin_v1`: The implementation of the Swin Transformer v1 model.
//! - `resnet`: The implementation of the ResNet model family.
//! - `vgg`: The implementation of the VGG model family.
//!
//! The `BackboneEnum` is used to encapsulate different backbone types, allowing for
//! flexible model construction.

mod build_backbone;
mod resnet;
mod resnet_block;
mod swin_v1;
mod vgg;

pub use build_backbone::*;
pub use resnet::*;
pub use swin_v1::*;
pub use vgg::*;
