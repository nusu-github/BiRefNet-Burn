#![doc = include_str!("../README.md")]

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
