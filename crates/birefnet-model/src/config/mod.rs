//! Model configuration structs and enums for `BiRefNet`.

mod core;
mod enums;

pub use core::{BackboneConfig, DecoderConfig, ModelConfig, RefineConfig, TaskConfig};
pub use enums::{
    Backbone, DecBlk, DecChannelsInter, DecoderAttention, InterpolationStrategy, LateralBlock,
    MultiScaleInput, Optimizer, PromptForLocation, Refine, SqueezeBlock, Task,
};
