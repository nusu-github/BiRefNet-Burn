//! Pretrained `BiRefNet` model presets, Hugging Face Hub fetching, and checkpoint loading.
//!
//! # Features
//!
//! | Feature | Default | Effect |
//! | --- | --- | --- |
//! | `hub` | no | Enables `HfHubFetcher`, a blocking Hugging Face Hub [`Fetcher`] |
//! | `test-util` | no | Enables `MockFetcher`, an offline [`Fetcher`] for tests |
//!
//! # Checkpoint formats
//!
//! Formats are detected from the file header via [`WeightFormat::detect`], not from extensions:
//!
//! - **Burnpack**, written by `burn::module::Module::save_file`;
//! - **`SafeTensors`**, the format of the upstream Hugging Face releases;
//! - **`PyTorch`** `torch.save` archives (files only; in-memory byte buffers are unsupported).
//!
//! Upstream `PyTorch` parameter names are remapped to the Burn module tree, and floating-point
//! parameters are cast to the device's default float dtype. [`MODEL_SPECS`] lists the 15 official
//! presets (`General`, `General-HR`, `Matting`, `General-Lite`, `DIS`, and more).
//!
//! # Examples
//!
//! ```rust
//! use birefnet_weights::{WeightFormat, load_weights_from_bytes, load_weights_from_path, model_spec};
//! use burn::{module::Module, nn::LinearConfig, prelude::*};
//!
//! let spec = model_spec("General-Lite").expect("registered preset");
//! assert_eq!(spec.repo_id(), "ZhengPeng7/BiRefNet_lite");
//!
//! let device = Device::flex();
//! let dir = std::env::temp_dir().join(format!("birefnet-weights-doc-{}", std::process::id()));
//! std::fs::create_dir_all(&dir)?;
//! let path = dir.join("linear.bpk");
//! let saved = LinearConfig::new(4, 2).init(&device);
//! saved.clone().save_file(&path)?;
//!
//! assert_eq!(WeightFormat::from_path(&path)?, WeightFormat::Burnpack);
//! let from_file = load_weights_from_path(LinearConfig::new(4, 2).init(&device), &path, &device)?;
//! let from_bytes =
//!     load_weights_from_bytes(LinearConfig::new(4, 2).init(&device), std::fs::read(&path)?, &device)?;
//! from_file.weight.val().into_data().assert_eq(&saved.weight.val().into_data(), true);
//! from_bytes.weight.val().into_data().assert_eq(&saved.weight.val().into_data(), true);
//! # std::fs::remove_dir_all(&dir)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Pretrained presets are fetched through any [`Fetcher`]; with the `hub` feature:
//!
//! ```rust,no_run
//! # #[cfg(feature = "hub")]
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use birefnet_model::BiRefNet;
//! use birefnet_weights::{BiRefNetWeightLoading, HfHubFetcher, ManagedModel};
//!
//! let fetcher = HfHubFetcher::new()?;
//! let managed = ManagedModel::from_pretrained("General")?;
//! let model = BiRefNet::from_managed(&managed, &fetcher, &burn::tensor::Device::flex())?;
//! # let _ = model;
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "hub"))]
//! # fn main() {}
//! ```

mod error;
mod fetcher;
mod loader;
mod registry;

#[doc(inline)]
pub use error::WeightError;
#[doc(inline)]
pub use fetcher::Fetcher;
#[cfg(feature = "hub")]
#[doc(inline)]
pub use fetcher::HfHubFetcher;
#[cfg(any(test, feature = "test-util"))]
#[doc(inline)]
pub use fetcher::MockFetcher;
#[doc(inline)]
pub use loader::{
    BiRefNetWeightLoading, ManagedModel, load_weights_from_bytes, load_weights_from_path,
};
#[doc(inline)]
pub use registry::{
    MODEL_SPECS, ModelName, ModelSpec, WeightFormat, WeightSource, list_available_models,
    model_spec,
};
