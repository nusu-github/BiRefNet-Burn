//! BiRefNet: Bilateral Reference Network for high-resolution dichotomous image segmentation
//!
//! This crate provides a unified interface for BiRefNet models, supporting both
//! inference and training capabilities across multiple backends.

pub mod backend;

// Re-export core modules
// Re-export backend types for convenience
pub use backend::{create_device, get_backend_name, SelectedBackend, SelectedDevice};
#[cfg(feature = "inference")]
pub use birefnet_inference as inference;
#[cfg(feature = "train")]
pub use birefnet_loss as loss;
#[cfg(feature = "train")]
pub use birefnet_metric as metric;
pub use birefnet_model as model;
#[cfg(feature = "train")]
pub use birefnet_train as train;
pub use birefnet_util as util;
