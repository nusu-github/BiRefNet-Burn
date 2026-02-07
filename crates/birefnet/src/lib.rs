//! `BiRefNet`: Bilateral Reference Network for dichotomous image segmentation.
//!
//! This crate provides a unified interface for `BiRefNet` models, supporting both
//! inference and training capabilities across multiple backends.

pub mod backend;
#[cfg(feature = "inference")]
pub mod inference;
#[cfg(feature = "train")]
pub mod training;

// Re-export core modules
// Re-export backend types for convenience
#[doc(inline)]
pub use backend::burn_backend_types;
#[cfg(feature = "inference")]
#[doc(inline)]
pub use birefnet_inference;
#[cfg(feature = "train")]
#[doc(inline)]
pub use birefnet_loss as loss;
#[cfg(feature = "train")]
#[doc(inline)]
pub use birefnet_metric as metric;
#[doc(inline)]
pub use birefnet_model as model;
#[cfg(feature = "train")]
#[doc(inline)]
pub use birefnet_train as train;
#[doc(inline)]
pub use birefnet_util as util;
