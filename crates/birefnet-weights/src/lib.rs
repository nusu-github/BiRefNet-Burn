//! Pretrained model registry, remote hub fetching, and checkpoint loaders for `BiRefNet`.
//!
//! Supported formats:
//! - Native `Burnpack` (`.bpk`) checkpoints saved with [`burn::module::Module::save_file`].
//! - `SafeTensors` (`.safetensors`) checkpoints with upstream `PyTorch` key remapping.
//! - `PyTorch` pickle (`.pt` / `.pth`) checkpoints.
//!
//! Enable the optional `hub` feature for `HuggingFace` Hub downloads via [`HfHubFetcher`].

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
pub use loader::{BiRefNetWeightLoading, ManagedModel, load_weights_from_path};
#[doc(inline)]
pub use registry::{
    MODEL_SPECS, ModelName, ModelSpec, WeightFormat, WeightSource, get_model_spec,
    list_available_models,
};
