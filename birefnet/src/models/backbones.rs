//! # Backbone Networks
//!
//! This module provides the backbone architectures used for feature extraction in BiRefNet.
//!
//! ## Components
//!
//! - `build_backbone`: A factory function that constructs a backbone model based on the
//!   provided configuration. It currently supports various versions of the Swin Transformer.
//! - `swin_v1`: The implementation of the Swin Transformer v1 model.
//!
//! The `BackboneEnum` is used to encapsulate different backbone types, allowing for
//! flexible model construction.

mod build_backbone;
mod swin_v1;

pub use build_backbone::*;
pub use swin_v1::*;
