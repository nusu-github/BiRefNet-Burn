//! The `BiRefNet` architecture on Burn: encoder backbone, squeeze stage, bilateral decoder,
//! optional refinement, and validated configuration.
//!
//! # Features
//!
//! | Feature | Default | Effect |
//! | --- | --- | --- |
//! | `train` | no | Enables `BiRefNetBatch`, `BiRefNetOutput`, `BiRefNet::forward_step`, and Burn `TrainStep` / `InferenceStep` |
//!
//! # Core components
//!
//! - **Model** ([`BiRefNet`], [`BiRefNetConfig`]): encoder backbone, squeeze block
//!   (`BasicDecBlk`, `ResBlk`, `ASPP`, `ASPPDeformable`), bilateral multi-scale decoder, and
//!   optional refinement (`RefUNet`, `Refiner`, `RefinerPVTInChannels4`).
//! - **Configuration** ([`ModelConfig`], [`TaskConfig`], [`BackboneConfig`], [`DecoderConfig`],
//!   [`RefineConfig`]): serializable architecture settings validated by [`ModelConfig::validate`].
//! - **Errors** ([`ModelError`]): typed configuration and shape validation errors.
//!
//! # Examples
//!
//! ```rust
//! use birefnet_model::{
//!     Backbone, BackboneConfig, BiRefNetConfig, InterpolationStrategy, ModelConfig,
//! };
//! use burn::prelude::*;
//!
//! let device = Device::flex();
//! let config = BiRefNetConfig::new(
//!     ModelConfig::new(InterpolationStrategy::Bilinear)
//!         .with_backbone(BackboneConfig::new().with_backbone(Backbone::PvtV2B0)),
//! );
//!
//! let model = config.init(&device)?;
//! let output = model.forward(Tensor::zeros([1, 3, 64, 64], &device))?;
//! assert_eq!(output.dims(), [1, 1, 64, 64]);
//! # Ok::<(), birefnet_model::ModelError>(())
//! ```

mod config;
mod error;
mod models;
#[cfg(feature = "train")]
mod training;

#[doc(inline)]
pub use config::{
    Backbone, BackboneConfig, DecBlk, DecChannelsInter, DecoderAttention, DecoderConfig,
    InterpolationStrategy, LateralBlock, ModelConfig, MultiScaleInput, Refine, RefineConfig,
    SqueezeBlock, Task, TaskConfig,
};
#[doc(inline)]
pub use error::ModelError;
#[doc(inline)]
pub use models::birefnet::{BiRefNet, BiRefNetConfig};
#[cfg(feature = "train")]
#[doc(inline)]
pub use training::{BiRefNetBatch, BiRefNetOutput};
