//! Tensor operations and layers that `BiRefNet` needs but Burn does not ship.
//!
//! | Item | Purpose |
//! | --- | --- |
//! | [`DropPath`] / [`DropPathConfig`] | Stochastic depth: drops whole samples of a residual branch during training. |
//! | [`trunc_normal`] | Samples `N(mean, std²)` truncated to `[a, b]` (`PyTorch` `trunc_normal_` semantics). |
//! | [`erfinv`] / [`Erfinv`] | Element-wise inverse error function. |
//!
//! All operations work on any Burn backend and preserve the input float dtype.
//!
//! # Examples
//!
//! [`DropPath`] follows the contract of `burn::nn::Dropout`: it is active only while its training
//! flag is set (cleared by `Module::valid`) and its input lives on an autodiff device.
//!
//! ```rust
//! use birefnet_extra_ops::DropPathConfig;
//! use burn::prelude::*;
//!
//! let drop_path = DropPathConfig::new().with_drop_prob(0.1).init();
//!
//! // Plain (non-autodiff) devices run inference, so the layer is the identity.
//! let x = Tensor::<3>::ones([4, 49, 96], &Device::flex());
//! let y = drop_path.forward(x.clone());
//! y.into_data().assert_eq(&x.into_data(), true);
//! ```
//!
//! [`trunc_normal`] creates a new tensor:
//!
//! ```rust
//! use birefnet_extra_ops::trunc_normal;
//! use burn::prelude::*;
//!
//! let device = Device::flex();
//! let weights = trunc_normal::<2>([64, 64], 0.0, 0.02, -0.04, 0.04, &device);
//! let max = weights.abs().max().into_scalar::<f32>();
//! assert!(max <= 0.04);
//! ```
//!
//! [`erfinv`] is available as a free function and as a tensor method via [`Erfinv`]:
//!
//! ```rust
//! use birefnet_extra_ops::Erfinv;
//! use burn::prelude::*;
//!
//! let x = Tensor::<1>::from_floats([0.5], &Device::flex());
//! let y = x.erfinv().into_scalar::<f32>();
//! assert!((y - 0.476_936).abs() < 1e-4);
//! ```

mod drop_path;
mod erfinv;
mod trunc_normal;

#[doc(inline)]
pub use drop_path::{DropPath, DropPathConfig};
#[doc(inline)]
pub use erfinv::{Erfinv, erfinv};
#[doc(inline)]
pub use trunc_normal::trunc_normal;
