//! Weight resolution and checkpoint loading for `BiRefNet` models.

use std::path::{Path, PathBuf};

use birefnet_model::{
    Backbone, BackboneConfig, BiRefNet, BiRefNetConfig, DecoderConfig, InterpolationStrategy,
    ModelConfig, Task,
};
use burn::{
    module::{Module, ModuleMapper, Param},
    store::{
        ApplyResult, FloatCastAdapter, ModuleAdapter, ModuleRecord, ModuleSnapshot,
        PyTorchToBurnAdapter, PytorchStore, SafetensorsStore,
    },
    tensor::{DType, Device, FloatDType, Tensor},
};

#[cfg(feature = "hub")]
use crate::fetcher::HfHubFetcher;
use crate::{
    error::WeightError,
    fetcher::Fetcher,
    registry::{
        ModelName, ModelSpec, WeightFormat, WeightSource, get_model_spec, list_available_models,
    },
};

/// Resolved model identity, architecture config builder, and weight source.
#[derive(Debug, Clone)]
pub struct ManagedModel {
    name: ModelName,
    config: Option<BiRefNetConfig>,
    weights: WeightSource,
}

impl ManagedModel {
    /// Creates a model record with explicit name, optional config override, and weight source.
    #[must_use]
    pub const fn new(
        name: ModelName,
        config: Option<BiRefNetConfig>,
        weights: WeightSource,
    ) -> Self {
        Self {
            name,
            config,
            weights,
        }
    }

    /// Looks up a registered pretrained model preset by name.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError`] (`is_unknown_model`) if `model_name` is not registered.
    pub fn from_pretrained(model_name: &str) -> Result<Self, WeightError> {
        let spec = get_model_spec(model_name)
            .ok_or_else(|| WeightError::unknown_model(model_name, &list_available_models()))?;

        let weights = WeightSource::Remote {
            repo_id: spec.repo_id(),
            filename: String::from("model.safetensors"),
        };
        let config = Some(spec.build_config());

        Ok(Self::new(ModelName::new(model_name), config, weights))
    }

    /// Creates a managed model pointing to a local checkpoint file.
    #[must_use]
    pub fn from_local(
        name: impl Into<ModelName>,
        path: impl Into<PathBuf>,
        config: Option<BiRefNetConfig>,
    ) -> Self {
        Self::new(
            name.into(),
            config,
            WeightSource::Local { path: path.into() },
        )
    }

    /// Returns the model identifier.
    #[must_use]
    pub const fn name(&self) -> &ModelName {
        &self.name
    }

    /// Returns the explicit architecture configuration override, if set.
    #[must_use]
    pub const fn config(&self) -> Option<&BiRefNetConfig> {
        self.config.as_ref()
    }

    /// Returns the weight source descriptor.
    #[must_use]
    pub const fn weight_source(&self) -> &WeightSource {
        &self.weights
    }

    /// Returns the architecture configuration for this model, falling back to `General` defaults.
    #[must_use]
    pub fn build_config(&self) -> BiRefNetConfig {
        if let Some(config) = &self.config {
            return config.clone();
        }

        get_model_spec(self.name.as_str()).map_or_else(
            || {
                let model_config = ModelConfig::new(InterpolationStrategy::Bilinear)
                    .with_backbone(BackboneConfig::new().with_backbone(Backbone::SwinV1L))
                    .with_decoder(DecoderConfig::new())
                    .with_task(birefnet_model::TaskConfig::new().with_task(Task::General));
                BiRefNetConfig::new(model_config)
            },
            ModelSpec::build_config,
        )
    }

    /// Returns the preset `(height, width)` input resolution if `name` is in the registry.
    #[must_use]
    pub fn default_resolution(&self) -> Option<(u32, u32)> {
        get_model_spec(self.name.as_str()).map(|spec| spec.default_resolution)
    }

    /// Returns whether this model supports dynamic input resolutions during inference.
    #[must_use]
    pub fn supports_dynamic_resolution(&self) -> bool {
        get_model_spec(self.name.as_str()).is_some_and(|spec| spec.supports_dynamic_resolution)
    }

    /// Resolves the checkpoint to a local file path using `fetcher` for remote sources.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError`] (`is_not_found`) if a local path is missing on disk, or any
    /// fetcher error when resolving a remote artifact.
    pub fn resolve_path_with(&self, fetcher: &dyn Fetcher) -> Result<PathBuf, WeightError> {
        match &self.weights {
            WeightSource::Local { path } => {
                if path.exists() {
                    Ok(path.clone())
                } else {
                    Err(WeightError::not_found(path))
                }
            }
            WeightSource::Remote { repo_id, filename } => fetcher.fetch(repo_id, filename),
        }
    }

    /// Returns `true` if the local checkpoint exists on disk or is cached in `fetcher`.
    #[must_use]
    pub fn is_available_with(&self, fetcher: &dyn Fetcher) -> bool {
        match &self.weights {
            WeightSource::Local { path } => path.exists(),
            WeightSource::Remote { repo_id, filename } => fetcher.is_cached(repo_id, filename),
        }
    }

    /// Resolves the checkpoint to a local file path using `HuggingFace` Hub when remote.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError`] if downloading or resolving the checkpoint fails.
    #[cfg(feature = "hub")]
    pub fn resolve_path(&self) -> Result<PathBuf, WeightError> {
        match &self.weights {
            WeightSource::Local { .. } => self.resolve_path_with(&NoopFetcher),
            WeightSource::Remote { .. } => {
                let fetcher = HfHubFetcher::new()?;
                self.resolve_path_with(&fetcher)
            }
        }
    }

    /// Returns `true` if the checkpoint exists locally or in the `HuggingFace` Hub cache.
    #[cfg(feature = "hub")]
    #[must_use]
    pub fn is_available(&self) -> bool {
        match &self.weights {
            WeightSource::Local { path } => path.exists(),
            WeightSource::Remote { .. } => {
                HfHubFetcher::new().is_ok_and(|fetcher| self.is_available_with(&fetcher))
            }
        }
    }
}

#[cfg(feature = "hub")]
#[derive(Debug)]
struct NoopFetcher;

#[cfg(feature = "hub")]
impl Fetcher for NoopFetcher {
    fn fetch(&self, _repo_id: &str, _filename: &str) -> Result<PathBuf, WeightError> {
        Err(WeightError::unsupported_format(
            "unexpected remote fetch request for local weight source",
        ))
    }

    fn is_cached(&self, _repo_id: &str, _filename: &str) -> bool {
        false
    }
}

struct CastFloatParams(FloatDType);

impl ModuleMapper for CastFloatParams {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let dtype = self.0;
        param.map(|tensor| tensor.cast(dtype))
    }
}

fn check_apply_result(result: &ApplyResult) -> Result<(), WeightError> {
    if !result.missing.is_empty() || !result.unused.is_empty() {
        tracing::debug!(
            applied = result.applied.len(),
            missing = result.missing.len(),
            unused = result.unused.len(),
            "checkpoint parameter keys differed from model structure",
        );
    }
    if let Some(first_error) = result.errors.first() {
        return Err(WeightError::load(
            format!(
                "failed to apply {} tensor(s) (first: {first_error})",
                result.errors.len()
            ),
            None,
        ));
    }
    Ok(())
}

fn load_pytorch_model<M: Module>(
    mut model: M,
    weights_path: &Path,
    device: &Device,
) -> Result<M, WeightError> {
    let mut store = PytorchStore::from_file(weights_path);
    let result = model.load_from(&mut store)?;
    check_apply_result(&result)?;
    Ok(model.map(&mut CastFloatParams(device.settings().float_dtype)))
}

fn load_safetensors_model<M: Module>(
    mut model: M,
    weights_path: &Path,
    device: &Device,
) -> Result<M, WeightError> {
    let cast_adapter = FloatCastAdapter::to(DType::from(device.settings().float_dtype));
    let mut store = SafetensorsStore::from_file(weights_path)
        .skip_enum_variants(true)
        .with_from_adapter(PyTorchToBurnAdapter.chain(cast_adapter))
        .with_key_remapping(r"decoder\.conv_out1\.0\.(.+)", "decoder.conv_out1.$1")
        .with_key_remapping(
            r"decoder\.gdt_convs_attn_([2-4])\.0\.(.+)",
            "decoder.gdt_convs_attn_$1.$2",
        )
        .with_key_remapping(
            r"decoder\.gdt_convs_pred_([2-4])\.0\.(.+)",
            "decoder.gdt_convs_pred_$1.$2",
        )
        .with_key_remapping(r"bb\.norm([0-3])\.(.+)", "bb.norm_layers.$1.$2")
        .with_key_remapping(
            r"(.+?)\.gdt_convs_([2-4])\.0\.(.+)",
            "$1.gdt_convs_$2.conv.$3",
        )
        .with_key_remapping(
            r"(.+?)\.gdt_convs_([2-4])\.1\.(.+)",
            "$1.gdt_convs_$2.bn.$3",
        )
        .with_key_remapping(
            r"(.+)\.global_avg_pool\.1\.(.+)",
            "$1.global_avg_pool.conv.$2",
        )
        .with_key_remapping(
            r"(.+)\.global_avg_pool\.2\.(.+)",
            "$1.global_avg_pool.bn.$2",
        );

    let result = model.load_from(&mut store)?;
    check_apply_result(&result)?;
    Ok(model)
}

fn load_burnpack_model<M: Module>(model: M, weights_path: &Path) -> Result<M, WeightError> {
    let record = ModuleRecord::load(weights_path)?.cast_to_module_dtype();
    Ok(model.try_load_record(record)?)
}

/// Loads checkpoint weights from a local path into any Burn [`Module`], inferring format.
///
/// # Errors
///
/// Returns [`WeightError`] (`is_not_found`) if `weights_path` does not exist, or `is_load`
/// if deserialization fails or tensor shapes mismatch.
pub fn load_weights_from_path<M: Module>(
    model: M,
    weights_path: &Path,
    device: &Device,
) -> Result<M, WeightError> {
    if !weights_path.exists() {
        return Err(WeightError::not_found(weights_path));
    }
    if !weights_path.is_file() {
        return Err(WeightError::unsupported_format(format!(
            "expected a checkpoint file path, got directory or special file: `{}`",
            weights_path.display()
        )));
    }

    match WeightFormat::from_path(weights_path) {
        WeightFormat::PyTorch => load_pytorch_model(model, weights_path, device),
        WeightFormat::SafeTensors => load_safetensors_model(model, weights_path, device),
        WeightFormat::Burnpack => load_burnpack_model(model, weights_path),
        WeightFormat::Auto => match load_burnpack_model(model.clone(), weights_path) {
            Ok(loaded) => Ok(loaded),
            Err(burnpack_err) => match load_safetensors_model(model.clone(), weights_path, device)
            {
                Ok(loaded) => Ok(loaded),
                Err(safetensors_err) => load_pytorch_model(model, weights_path, device).map_err(
                    |pytorch_err| {
                        WeightError::load(
                            format!(
                                "failed to load `{}` as Burnpack ({burnpack_err}), SafeTensors ({safetensors_err}), or PyTorch ({pytorch_err})",
                                weights_path.display()
                            ),
                            None,
                        )
                    },
                ),
            },
        },
    }
}

/// Extension trait adding checkpoint loading convenience methods onto [`BiRefNet`].
pub trait BiRefNetWeightLoading: Sized {
    /// Loads weights from a local checkpoint file on `device`.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError`] if the file is missing or cannot be loaded into `self`.
    fn load_weights(self, path: impl AsRef<Path>, device: &Device) -> Result<Self, WeightError>;

    /// Loads weights described by `managed` using `fetcher` for remote sources.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError`] if resolution or checkpoint loading fails.
    fn load_weights_from_managed_with(
        self,
        managed: &ManagedModel,
        fetcher: &dyn Fetcher,
        device: &Device,
    ) -> Result<Self, WeightError>;

    /// Initializes the architecture configured by `managed` on `device` and loads its weights.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError`] if model construction, fetching, or weight loading fails.
    fn from_managed_model_with(
        managed: &ManagedModel,
        fetcher: &dyn Fetcher,
        device: &Device,
    ) -> Result<Self, WeightError>;

    /// Loads weights described by `managed` using `HuggingFace` Hub for remote sources.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError`] if fetching or checkpoint loading fails.
    #[cfg(feature = "hub")]
    fn load_weights_from_managed(
        self,
        managed: &ManagedModel,
        device: &Device,
    ) -> Result<Self, WeightError>;

    /// Initializes the architecture configured by `managed` on `device` and fetches weights
    /// from `HuggingFace` Hub when remote.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError`] if initialization, hub download, or weight loading fails.
    #[cfg(feature = "hub")]
    fn from_managed_model(managed: &ManagedModel, device: &Device) -> Result<Self, WeightError>;
}

impl BiRefNetWeightLoading for BiRefNet {
    fn load_weights(self, path: impl AsRef<Path>, device: &Device) -> Result<Self, WeightError> {
        load_weights_from_path(self, path.as_ref(), device)
    }

    fn load_weights_from_managed_with(
        self,
        managed: &ManagedModel,
        fetcher: &dyn Fetcher,
        device: &Device,
    ) -> Result<Self, WeightError> {
        let weights_path = managed.resolve_path_with(fetcher)?;
        load_weights_from_path(self, &weights_path, device)
    }

    fn from_managed_model_with(
        managed: &ManagedModel,
        fetcher: &dyn Fetcher,
        device: &Device,
    ) -> Result<Self, WeightError> {
        let model = managed.build_config().init(device)?;
        model.load_weights_from_managed_with(managed, fetcher, device)
    }

    #[cfg(feature = "hub")]
    fn load_weights_from_managed(
        self,
        managed: &ManagedModel,
        device: &Device,
    ) -> Result<Self, WeightError> {
        let weights_path = managed.resolve_path()?;
        load_weights_from_path(self, &weights_path, device)
    }

    #[cfg(feature = "hub")]
    fn from_managed_model(managed: &ManagedModel, device: &Device) -> Result<Self, WeightError> {
        let model = managed.build_config().init(device)?;
        model.load_weights_from_managed(managed, device)
    }
}
