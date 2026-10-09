//! `BiRefNet`: Bilateral Reference Network for dichotomous image segmentation.
//!
//! This crate provides a unified interface for `BiRefNet` models, supporting both
//! inference and training capabilities across multiple backends. The backend is selected at
//! run time with [`device::select_device`]; Cargo features only decide which backends are
//! compiled in.

pub mod device;
#[cfg(feature = "inference")]
pub mod inference;
#[cfg(feature = "train")]
pub mod training;

// Re-export core modules
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
