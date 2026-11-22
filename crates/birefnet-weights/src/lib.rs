#![doc = include_str!("../README.md")]

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
