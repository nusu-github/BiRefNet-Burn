//! `BiRefNet`: Bilateral Reference Network for dichotomous image segmentation.
//!
//! This umbrella crate provides a single entrypoint to the modular `BiRefNet` workspace
//! crates behind additive Cargo feature flags (`weights`, `hub`, `inference`, `train`).
//! Backend selection happens at run time via [`device::select_device`], while Cargo features
//! decide which backends are compiled in.

pub mod device;

#[cfg(feature = "inference")]
#[doc(inline)]
pub use birefnet_inference as inference;
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
#[cfg(feature = "weights")]
#[doc(inline)]
pub use birefnet_weights as weights;
