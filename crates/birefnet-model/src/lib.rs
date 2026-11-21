//! `BiRefNet`: Bilateral Reference Network model architecture in Burn.

mod config;
mod error;
mod models;
#[cfg(feature = "train")]
mod training;

#[doc(inline)]
pub use config::{
    Backbone, BackboneConfig, DecBlk, DecChannelsInter, DecoderAttention, DecoderConfig,
    InterpolationStrategy, LateralBlock, ModelConfig, MultiScaleInput, Optimizer,
    PromptForLocation, Refine, RefineConfig, SqueezeBlock, Task, TaskConfig,
};
#[doc(inline)]
pub use error::ModelError;
#[doc(inline)]
pub use models::birefnet::{BiRefNet, BiRefNetConfig};
#[cfg(feature = "train")]
#[doc(inline)]
pub use training::{BiRefNetBatch, BiRefNetOutput};
