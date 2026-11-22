//! Image and mask utilities for `BiRefNet`: normalization, distance transforms, binary
//! morphology, Gaussian filtering, foreground color refinement, and optional image codecs.
//!
//! # Features
//!
//! | Feature | Default | Effect |
//! | --- | --- | --- |
//! | `image-io` | no | Enables the `image` module (decoding, encoding, and tensor conversion via the `image` crate) |
//!
//! # Modules
//!
//! | Module | Contents |
//! | --- | --- |
//! | [`normalize`] | `ImageNet` standardization and alpha masking |
//! | [`distance`] | Exact Euclidean (with optional nearest-site indices) and Manhattan transforms |
//! | [`morphology`] | [`StructuringElement`](morphology::StructuringElement) plus erosion, dilation, opening, closing, gradient |
//! | [`filters`] | MATLAB-compatible Gaussian kernels and edge-replicated filtering |
//! | [`foreground`] | Two-pass blur-fusion foreground color estimation |
//! | `image` | Image decoding, encoding, and tensor conversion (`image-io` feature) |
//!
//! Every function accepts any float dtype and returns results in the input dtype. Without the
//! `image-io` feature the crate depends only on Burn.
//!
//! # Examples
//!
//! ```rust
//! use birefnet_util::{filters::gaussian_filter_matlab, normalize::apply_imagenet_normalization};
//! use burn::prelude::*;
//!
//! let device = Device::flex();
//! let image = Tensor::<4>::full([1, 3, 16, 16], 0.5, &device);
//! let normalized = apply_imagenet_normalization(image)?;
//! let smoothed = gaussian_filter_matlab(normalized, (3, 3), 1.0);
//! assert_eq!(smoothed.dims(), [1, 3, 16, 16]);
//! # Ok::<(), birefnet_util::UtilError>(())
//! ```
//!
//! Morphology and distance transforms treat values `>= 0.5` as foreground:
//!
//! ```rust
//! use birefnet_util::{
//!     distance::euclidean_distance_transform,
//!     morphology::{StructuringElement, dilation},
//! };
//! use burn::prelude::*;
//!
//! let device = Device::flex();
//! let mut dot = [[0.0_f32; 5]; 5];
//! dot[2][2] = 1.0;
//! let mask = Tensor::<4>::from_floats([[dot]], &device);
//!
//! let cross = dilation(mask.clone(), &StructuringElement::disk(1));
//! assert_eq!(cross.sum().into_scalar::<f32>(), 5.0);
//!
//! // Distance from every pixel to the nearest pixel `< 0.5` (the background here).
//! let inside = euclidean_distance_transform(&mask)?;
//! assert_eq!(inside.max().into_scalar::<f32>(), 1.0);
//! # Ok::<(), birefnet_util::UtilError>(())
//! ```

pub mod distance;
mod error;
pub mod filters;
pub mod foreground;
#[cfg(feature = "image-io")]
pub mod image;
pub mod morphology;
pub mod normalize;

#[doc(inline)]
pub use error::UtilError;
