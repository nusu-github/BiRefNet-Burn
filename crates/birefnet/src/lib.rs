//! Umbrella library crate for `BiRefNet` (Bilateral Reference Network) on
//! [Burn](https://burn.dev).
//!
//! This crate re-exports the workspace model, utility, weight loading, inference, and training
//! crates behind feature flags and adds runtime [`device`] selection. The `birefnet` CLI binary
//! lives in the sibling `birefnet-cli` crate.
//!
//! # Features
//!
//! No compute backend is enabled by default; the final binary or test target enables one or more
//! backend features and selects one at run time via [`device::select_device`].
//!
//! | Feature | Default | Effect |
//! | --- | --- | --- |
//! | `weights` | via `hub` / `inference` | Enables `weights` (Burnpack, `SafeTensors`, `PyTorch` checkpoint IO) |
//! | `hub` | yes | Enables Hugging Face Hub downloads in `weights` |
//! | `inference` | yes | Enables `inference` (`Segmenter`) and image codecs |
//! | `train` | no | Enables `loss`, `metric`, `train`, and `TrainStep` / `InferenceStep` |
//! | `flex` | no | Pure-Rust CPU backend with SIMD and threads |
//! | `cpu` | no | `CubeCL` CPU backend (LLVM JIT) |
//! | `wgpu` | no | `WebGPU` backend (Vulkan / Metal / DX12 chosen at run time) |
//! | `vulkan` | no | `WebGPU` pinned to Vulkan |
//! | `metal` | no | `WebGPU` pinned to Metal |
//! | `cuda` | no | NVIDIA CUDA backend |
//! | `rocm` | no | AMD `ROCm` backend |
//! | `fusion` | no | Kernel fusion and autotuning for `CubeCL` backends |
//!
//! # Stack size on Windows
//!
//! Initializing or running deep `BiRefNet` module trees (such as Swin-L) recurses past the
//! 1 MiB default main-thread stack on Windows. Run model initialization, weight loading, and
//! forward passes on a thread with a larger stack (for example 64 MiB) or via `stacker::grow`.
//!
//! # Examples
//!
//! Parsing device and precision names with typed [`device::DeviceError`] predicates:
//!
//! ```rust
//! use birefnet::device::{Precision, select_device};
//!
//! let precision: Precision = "bf16".parse()?;
//! assert_eq!(precision.to_string(), "bf16");
//!
//! let err = select_device("tpu").unwrap_err();
//! assert!(err.is_unsupported_device());
//! assert!(select_device("cuda:x").unwrap_err().is_invalid_index());
//! # Ok::<(), birefnet::device::DeviceError>(())
//! ```
//!
//! Segmenting an image with a pretrained preset (requires the `inference` and `hub` features plus
//! a backend feature such as `flex` in the final binary):
//!
//! ```rust,no_run
//! # #[cfg(all(feature = "inference", feature = "hub"))]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use birefnet::{
//!     device::{Precision, configure_precision, select_device},
//!     inference::{Segmenter, SegmenterConfig},
//!     model::BiRefNet,
//!     weights::{BiRefNetWeightLoading, HfHubFetcher, ManagedModel},
//! };
//!
//! let mut device = select_device("default")?;
//! configure_precision(&mut device, Precision::F32)?;
//!
//! let managed = ManagedModel::from_pretrained("General")?;
//! let model = BiRefNet::from_managed(&managed, &HfHubFetcher::new()?, &device)?;
//!
//! let segmenter = Segmenter::new(&model, SegmenterConfig::new().with_input_size([1024, 1024])?);
//! let output = segmenter.segment_file("photo.png", &device)?;
//! output.to_rgba_image()?.save("photo_mask.png")?;
//! # Ok(())
//! # }
//! # #[cfg(not(all(feature = "inference", feature = "hub")))]
//! # fn main() {}
//! ```

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
