//! Weight resolution and checkpoint loading for `BiRefNet` models.

use std::path::{Path, PathBuf};

use birefnet_model::{
    Backbone, BackboneConfig, BiRefNet, BiRefNetConfig, DecoderConfig, InterpolationStrategy,
    ModelConfig, Task, TaskConfig,
};
use burn::{
    module::{Module, ModuleMapper, Param},
    store::{
        ApplyResult, FloatCastAdapter, ModuleAdapter, ModuleRecord, ModuleSnapshot,
        PyTorchToBurnAdapter, PytorchStore, SafetensorsStore,
    },
    tensor::{Bytes, DType, Device, FloatDType, Tensor},
};

use crate::{
    error::WeightError,
    fetcher::Fetcher,
    registry::{ModelName, WeightFormat, WeightSource, model_spec},
};

/// Checkpoint file name inside every registered Hugging Face repository.
const HUB_WEIGHTS_FILE: &str = "model.safetensors";

/// Resolved model identity, architecture config, and weight source.
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
    pub fn from_pretrained(model_name: impl AsRef<str>) -> Result<Self, WeightError> {
        let model_name = model_name.as_ref();
        let spec = model_spec(model_name).ok_or_else(|| WeightError::unknown_model(model_name))?;
        Ok(Self::new(
            ModelName::new(spec.name)?,
            Some(spec.build_config()),
            WeightSource::Remote {
                repo_id: spec.repo_id(),
                filename: HUB_WEIGHTS_FILE.to_owned(),
            },
        ))
    }

    /// Creates a managed model for a local checkpoint file.
    ///
    /// Without a `config`, [`build_config`](Self::build_config) uses the preset matching `name`
    /// or the `General` architecture.
    #[must_use]
    pub fn from_local(
        name: ModelName,
        path: impl Into<PathBuf>,
        config: Option<BiRefNetConfig>,
    ) -> Self {
        Self::new(name, config, WeightSource::Local { path: path.into() })
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

    /// Returns the architecture configuration, falling back to the `General` preset layout.
    #[must_use]
    pub fn build_config(&self) -> BiRefNetConfig {
        if let Some(config) = &self.config {
            return config.clone();
        }
        model_spec(&self.name).map_or_else(
            || {
                BiRefNetConfig::new(
                    ModelConfig::new(InterpolationStrategy::Bilinear)
                        .with_backbone(BackboneConfig::new().with_backbone(Backbone::SwinV1L))
                        .with_decoder(DecoderConfig::new())
                        .with_task(TaskConfig::new().with_task(Task::General)),
                )
            },
            crate::registry::ModelSpec::build_config,
        )
    }

    /// Returns the preset `[height, width]` inference resolution if `name` is registered.
    #[must_use]
    pub fn default_resolution(&self) -> Option<[u32; 2]> {
        model_spec(&self.name).map(|spec| spec.default_resolution)
    }

    /// Returns whether this model supports dynamic input resolutions during inference.
    #[must_use]
    pub fn supports_dynamic_resolution(&self) -> bool {
        model_spec(&self.name).is_some_and(|spec| spec.supports_dynamic_resolution)
    }

    /// Resolves the checkpoint to a local file, using `fetcher` for remote sources.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError`] (`is_not_found`) if a local file is missing, or the fetcher's
    /// error for a remote artifact.
    pub fn resolve_path(&self, fetcher: &(impl Fetcher + ?Sized)) -> Result<PathBuf, WeightError> {
        match &self.weights {
            WeightSource::Local { path } if path.is_file() => Ok(path.clone()),
            WeightSource::Local { path } => Err(WeightError::not_found(path)),
            WeightSource::Remote { repo_id, filename } => fetcher.fetch(repo_id, filename),
        }
    }

    /// Returns `true` if the local checkpoint exists or the remote one is cached by `fetcher`.
    #[must_use]
    pub fn is_available(&self, fetcher: &(impl Fetcher + ?Sized)) -> bool {
        match &self.weights {
            WeightSource::Local { path } => path.is_file(),
            WeightSource::Remote { repo_id, filename } => fetcher.is_cached(repo_id, filename),
        }
    }
}

struct CastFloatParams(FloatDType);

impl ModuleMapper for CastFloatParams {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let dtype = self.0;
        param.map(|tensor| tensor.cast(dtype))
    }
}

/// Logs keys that differed between checkpoint and model.
///
/// The stores validate by default, so apply errors and missing keys already surface as
/// `Err`; only unused checkpoint keys (such as `num_batches_tracked`) remain.
fn log_apply_result(result: &ApplyResult) {
    if !result.unused.is_empty() {
        tracing::debug!(
            name: "birefnet.weights.load.unused_keys",
            applied = result.applied.len(),
            unused = result.unused.len(),
            "checkpoint had {} keys the model does not use",
            result.unused.len()
        );
    }
}

/// Upstream PyTorch state-dict keys that differ from the Burn module tree.
const BIREFNET_KEY_REMAPPINGS: &[(&str, &str)] = &[
    (r"decoder\.conv_out1\.0\.(.+)", "decoder.conv_out1.$1"),
    (
        r"decoder\.gdt_convs_attn_([2-4])\.0\.(.+)",
        "decoder.gdt_convs_attn_$1.$2",
    ),
    (
        r"decoder\.gdt_convs_pred_([2-4])\.0\.(.+)",
        "decoder.gdt_convs_pred_$1.$2",
    ),
    (r"bb\.norm([0-3])\.(.+)", "bb.norm_layers.$1.$2"),
    (
        r"(.+?)\.gdt_convs_([2-4])\.0\.(.+)",
        "$1.gdt_convs_$2.conv.$3",
    ),
    (
        r"(.+?)\.gdt_convs_([2-4])\.1\.(.+)",
        "$1.gdt_convs_$2.bn.$3",
    ),
    (
        r"(.+)\.global_avg_pool\.1\.(.+)",
        "$1.global_avg_pool.conv.$2",
    ),
    (
        r"(.+)\.global_avg_pool\.2\.(.+)",
        "$1.global_avg_pool.bn.$2",
    ),
];

/// Where checkpoint bytes are read from.
enum Source<'a> {
    File(&'a Path),
    Memory(Vec<u8>),
}

fn configure_safetensors(store: SafetensorsStore, device: &Device) -> SafetensorsStore {
    let cast = FloatCastAdapter::to(DType::from(device.settings().float_dtype));
    BIREFNET_KEY_REMAPPINGS.iter().fold(
        store
            .skip_enum_variants(true)
            .with_from_adapter(PyTorchToBurnAdapter.chain(cast)),
        |store, &(from, to)| store.with_key_remapping(from, to),
    )
}

fn load_checkpoint<M: Module>(
    mut model: M,
    source: Source<'_>,
    format: WeightFormat,
    device: &Device,
) -> Result<M, WeightError> {
    match format {
        WeightFormat::PyTorch => {
            let Source::File(path) = source else {
                return Err(WeightError::unsupported_format(
                    "PyTorch checkpoints can only be loaded from files",
                    None,
                ));
            };
            let mut store = BIREFNET_KEY_REMAPPINGS.iter().fold(
                PytorchStore::from_file(path).map_indices_contiguous(false),
                |store, &(from, to)| store.with_key_remapping(from, to),
            );
            log_apply_result(&model.load_from(&mut store)?);
            Ok(model.map(&mut CastFloatParams(device.settings().float_dtype)))
        }
        WeightFormat::SafeTensors => {
            let store = match source {
                Source::File(path) => SafetensorsStore::from_file(path),
                Source::Memory(bytes) => SafetensorsStore::from_bytes(Some(bytes)),
            };
            let mut store = configure_safetensors(store, device);
            log_apply_result(&model.load_from(&mut store)?);
            Ok(model)
        }
        WeightFormat::Burnpack => {
            let record = match source {
                Source::File(path) => ModuleRecord::load(path)?,
                Source::Memory(bytes) => ModuleRecord::from_bytes(Bytes::from_bytes_vec(bytes))?,
            };
            Ok(model.try_load_record(record.cast_to_module_dtype())?)
        }
    }
}

/// Loads a checkpoint file into any Burn [`Module`], detecting its format from the file header.
///
/// Parameters are allocated in the device's persistent memory pool and cast to the device's
/// default float dtype. Upstream PyTorch / SafeTensors key names are remapped to Burn's.
///
/// # Errors
///
/// Returns [`WeightError`] (`is_not_found`) if `path` is not a file, `is_unsupported_format`
/// if the header is not recognized, or `is_load` if decoding or applying the weights fails.
pub fn load_weights_from_path<M: Module>(
    model: M,
    path: impl AsRef<Path>,
    device: &Device,
) -> Result<M, WeightError> {
    let path = path.as_ref();
    if !path.is_file() {
        return Err(WeightError::not_found(path));
    }
    let format = WeightFormat::from_path(path)?;
    device
        .memory_persistent_allocations(model, |model| {
            load_checkpoint(model, Source::File(path), format, device)
        })
        .map_err(|err| err.at(path))
}

/// Loads an in-memory Burnpack or SafeTensors checkpoint into any Burn [`Module`].
///
/// # Errors
///
/// Returns [`WeightError`] (`is_unsupported_format`) if the bytes are not a Burnpack or
/// SafeTensors checkpoint, or `is_load` if decoding or applying the weights fails.
pub fn load_weights_from_bytes<M: Module>(
    model: M,
    bytes: Vec<u8>,
    device: &Device,
) -> Result<M, WeightError> {
    let format = WeightFormat::detect(&bytes).ok_or_else(|| {
        WeightError::unsupported_format("expected a Burnpack or SafeTensors checkpoint", None)
    })?;
    device.memory_persistent_allocations((model, bytes), |(model, bytes)| {
        load_checkpoint(model, Source::Memory(bytes), format, device)
    })
}

/// Checkpoint loading for [`BiRefNet`].
pub trait BiRefNetWeightLoading: Sized {
    /// Loads weights from a local checkpoint file on `device`.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError`] if the file is missing or cannot be loaded into `self`.
    fn load_weights(self, path: impl AsRef<Path>, device: &Device) -> Result<Self, WeightError>;

    /// Loads the weights described by `managed`, using `fetcher` for remote sources.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError`] if resolution or checkpoint loading fails.
    fn load_managed_weights(
        self,
        managed: &ManagedModel,
        fetcher: &(impl Fetcher + ?Sized),
        device: &Device,
    ) -> Result<Self, WeightError>;

    /// Builds the architecture configured by `managed` on `device` and loads its weights.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError`] if model construction, fetching, or weight loading fails.
    fn from_managed(
        managed: &ManagedModel,
        fetcher: &(impl Fetcher + ?Sized),
        device: &Device,
    ) -> Result<Self, WeightError>;
}

impl BiRefNetWeightLoading for BiRefNet {
    fn load_weights(self, path: impl AsRef<Path>, device: &Device) -> Result<Self, WeightError> {
        load_weights_from_path(self, path, device)
    }

    fn load_managed_weights(
        self,
        managed: &ManagedModel,
        fetcher: &(impl Fetcher + ?Sized),
        device: &Device,
    ) -> Result<Self, WeightError> {
        load_weights_from_path(self, managed.resolve_path(fetcher)?, device)
    }

    fn from_managed(
        managed: &ManagedModel,
        fetcher: &(impl Fetcher + ?Sized),
        device: &Device,
    ) -> Result<Self, WeightError> {
        managed
            .build_config()
            .init(device)?
            .load_managed_weights(managed, fetcher, device)
    }
}

#[cfg(all(test, feature = "hub"))]
mod tests {
    use super::*;
    use crate::fetcher::HfHubFetcher;

    fn inspect_preset_checkpoint(fetcher: &HfHubFetcher, preset_name: &str) {
        let device = Device::flex();
        let managed = ManagedModel::from_pretrained(preset_name).expect("preset must resolve");
        let path = managed
            .resolve_path(fetcher)
            .expect("pretrained weights must download from Hugging Face Hub");
        let mut model = managed
            .build_config()
            .init(&device)
            .expect("model config must initialize");
        let mut store = configure_safetensors(SafetensorsStore::from_file(&path), &device);
        let result = model
            .load_from(&mut store)
            .expect("safetensors store must apply");

        assert!(
            !result.applied.is_empty(),
            "preset {preset_name} applied zero parameters"
        );
        assert!(
            result
                .unused
                .iter()
                .all(|key| key.ends_with(".num_batches_tracked")),
            "preset {preset_name} had unexpected unused keys: {:?}",
            result.unused
        );
    }

    #[test]
    #[ignore = "downloads pretrained weights from Hugging Face Hub"]
    fn hub_safetensors_presets_apply_cleanly() {
        std::thread::Builder::new()
            .name(String::from("hf-weights-check"))
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let fetcher = HfHubFetcher::new().expect("hub client");
                inspect_preset_checkpoint(&fetcher, "General-Lite");
                inspect_preset_checkpoint(&fetcher, "General");
            })
            .expect("spawn diagnostic thread")
            .join()
            .expect("diagnostic thread panicked");
    }
}
