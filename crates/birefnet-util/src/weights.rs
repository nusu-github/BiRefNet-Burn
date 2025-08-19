use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::LazyLock,
};

use birefnet_model::{
    Backbone, BackboneConfig, BiRefNetConfig, BiRefNetRecord, DecoderConfig, InterpolationStrategy,
    ModelConfig, Task, TaskConfig,
};
use burn::{
    module::Module,
    record::{BinFileRecorder, FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::backend::Backend,
};
use burn_import::{
    pytorch::{LoadArgs as PyTorchLoadArgs, PyTorchFileRecorder},
    safetensors::{LoadArgs as SafetensorsLoadArgs, SafetensorsFileRecorder},
};
use hf_hub::{api::sync, Repo, RepoType};
use thiserror::Error;

/// Errors that can occur during model weight management operations.
#[derive(Debug, Error)]
pub enum WeightError {
    /// Model not found in the registry
    #[error("unknown model '{model_name}' - available models: {available_models}")]
    UnknownModel {
        model_name: String,
        available_models: String,
    },

    /// HuggingFace Hub API errors
    #[error("failed to access HuggingFace Hub: {reason}")]
    HubApiError { reason: String },

    /// File system errors
    #[error("file system error: {reason}")]
    FileSystemError { reason: String },

    /// Invalid configuration
    #[error("invalid configuration: {reason}")]
    InvalidConfiguration { reason: String },

    /// Model loading errors
    #[error("failed to load model weights: {reason}")]
    ModelLoadError { reason: String },

    /// Record loading errors
    #[error("failed to load model record: {reason}")]
    RecordLoadError { reason: String },

    /// Unsupported file format
    #[error("unsupported weight file format: {format}")]
    UnsupportedFormat { format: String },
}

/// Defines the model specification
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub hf_model_id: &'static str, // HuggingFace model ID (e.g., "BiRefNet")
    pub default_resolution: (u32, u32),
    pub supports_dynamic_resolution: bool,
    pub config_builder: fn() -> BiRefNetConfig,
}

/// Helper function to create a BiRefNetConfig
fn create_config(
    task: Task,
    backbone: Backbone,
    interpolation_strategy: Option<InterpolationStrategy>,
) -> BiRefNetConfig {
    let model_config =
        ModelConfig::new(interpolation_strategy.unwrap_or(InterpolationStrategy::Bilinear))
            .with_task(TaskConfig::new().with_task(task))
            .with_backbone(BackboneConfig::new().with_backbone(backbone))
            .with_decoder(DecoderConfig::new());
    BiRefNetConfig::new(model_config)
}

/// Model catalog - simple mapping approach same as original implementation
static MODEL_SPECS: LazyLock<HashMap<String, ModelSpec>> = LazyLock::new(|| {
    let mut specs = HashMap::new();

    // Define all models in one place
    let models = [
        (
            "General",
            ModelSpec {
                hf_model_id: "BiRefNet",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: false,
                config_builder: || {
                    create_config(
                        Task::General,
                        Backbone::SwinV1L,
                        Some(InterpolationStrategy::Bilinear),
                    )
                },
            },
        ),
        (
            "General-HR",
            ModelSpec {
                hf_model_id: "BiRefNet_HR",
                default_resolution: (2048, 2048),
                supports_dynamic_resolution: false,
                config_builder: || {
                    create_config(
                        Task::General2k,
                        Backbone::SwinV1L,
                        Some(InterpolationStrategy::Bilinear),
                    )
                },
            },
        ),
        (
            "Matting-HR",
            ModelSpec {
                hf_model_id: "BiRefNet_HR-matting",
                default_resolution: (2048, 2048),
                supports_dynamic_resolution: false,
                config_builder: || {
                    create_config(
                        Task::Matting,
                        Backbone::SwinV1L,
                        Some(InterpolationStrategy::Bilinear),
                    )
                },
            },
        ),
        (
            "Matting",
            ModelSpec {
                hf_model_id: "BiRefNet-matting",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: false,
                config_builder: || {
                    create_config(
                        Task::Matting,
                        Backbone::SwinV1L,
                        Some(InterpolationStrategy::Bilinear),
                    )
                },
            },
        ),
        (
            "Portrait",
            ModelSpec {
                hf_model_id: "BiRefNet-portrait",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: false,
                config_builder: || {
                    create_config(
                        Task::Matting,
                        Backbone::SwinV1L,
                        Some(InterpolationStrategy::Bilinear),
                    )
                },
            },
        ),
        (
            "General-reso_512",
            ModelSpec {
                hf_model_id: "BiRefNet_512x512",
                default_resolution: (512, 512),
                supports_dynamic_resolution: false,
                config_builder: || {
                    create_config(
                        Task::General,
                        Backbone::SwinV1T,
                        Some(InterpolationStrategy::Bilinear),
                    )
                },
            },
        ),
        (
            "General-Lite",
            ModelSpec {
                hf_model_id: "BiRefNet_lite",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: false,
                config_builder: || {
                    create_config(
                        Task::General,
                        Backbone::SwinV1T,
                        Some(InterpolationStrategy::Bilinear),
                    )
                },
            },
        ),
        (
            "General-Lite-2K",
            ModelSpec {
                hf_model_id: "BiRefNet_lite-2K",
                default_resolution: (2048, 2048),
                supports_dynamic_resolution: false,
                config_builder: || {
                    create_config(
                        Task::General2k,
                        Backbone::SwinV1T,
                        Some(InterpolationStrategy::Bilinear),
                    )
                },
            },
        ),
        (
            "DIS",
            ModelSpec {
                hf_model_id: "BiRefNet-DIS5K",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: false,
                config_builder: || {
                    create_config(
                        Task::DIS5K,
                        Backbone::SwinV1L,
                        Some(InterpolationStrategy::Bilinear),
                    )
                },
            },
        ),
        (
            "HRSOD",
            ModelSpec {
                hf_model_id: "BiRefNet-HRSOD",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: false,
                config_builder: || {
                    create_config(
                        Task::HRSOD,
                        Backbone::SwinV1L,
                        Some(InterpolationStrategy::Bilinear),
                    )
                },
            },
        ),
        (
            "COD",
            ModelSpec {
                hf_model_id: "BiRefNet-COD",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: false,
                config_builder: || {
                    create_config(
                        Task::COD,
                        Backbone::SwinV1L,
                        Some(InterpolationStrategy::Bilinear),
                    )
                },
            },
        ),
        (
            "DIS-TR_TEs",
            ModelSpec {
                hf_model_id: "BiRefNet-DIS5K-TR_TEs",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: false,
                config_builder: || {
                    create_config(
                        Task::DIS5K,
                        Backbone::SwinV1L,
                        Some(InterpolationStrategy::Bilinear),
                    )
                },
            },
        ),
        (
            "General-legacy",
            ModelSpec {
                hf_model_id: "BiRefNet-legacy",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: false,
                config_builder: || {
                    create_config(
                        Task::General,
                        Backbone::SwinV1L,
                        Some(InterpolationStrategy::Bilinear),
                    )
                },
            },
        ),
        (
            "General-dynamic",
            ModelSpec {
                hf_model_id: "BiRefNet_dynamic",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: true,
                config_builder: || {
                    create_config(
                        Task::General,
                        Backbone::SwinV1L,
                        Some(InterpolationStrategy::Bilinear),
                    )
                },
            },
        ),
        (
            "Matting-dynamic",
            ModelSpec {
                hf_model_id: "BiRefNet_dynamic-matting",
                default_resolution: (1024, 1024),
                supports_dynamic_resolution: true,
                config_builder: || {
                    create_config(
                        Task::Matting,
                        Backbone::SwinV1L,
                        Some(InterpolationStrategy::Bilinear),
                    )
                },
            },
        ),
    ];

    for (name, spec) in models {
        specs.insert(name.to_owned(), spec);
    }

    specs
});

/// Model identifier - simple string-based management
/// Can be used directly as HashMap key
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelName(pub String);

impl ModelName {
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl core::str::FromStr for ModelName {
    type Err = core::convert::Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Self::new(s))
    }
}

impl From<&str> for ModelName {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for ModelName {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl core::fmt::Display for ModelName {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Weight location (external key & variants)
#[derive(Debug, Clone)]
pub enum WeightSource {
    /// Retrieved from HuggingFace
    Remote {
        repo_id: String,  // Repo type for hub-hf
        filename: String, // includes blobs/paths
    },
    /// Loaded from local file
    Local { path: PathBuf },
}

/// One record equivalent to "table row"
/// BiRefNetConfig retains existing structure as-is
/// Allows "column addition" with extra_params
#[derive(Debug, Clone)]
pub struct ManagedModel {
    pub name: ModelName,                // Model name
    pub config: Option<BiRefNetConfig>, // Existing configuration
    pub weights: WeightSource,          // Weight location
}

/// Access trait
/// Actual download/load is not implemented here
pub trait ModelRecord {
    fn name(&self) -> &ModelName;
    fn config(&self) -> &Option<BiRefNetConfig>;
    fn weight_source(&self) -> &WeightSource;
}

pub trait ModelLoader<B: Backend> {
    /// Load model weights into an existing model
    fn load_model<M: Module<B>>(&self, model: M, device: &B::Device) -> Result<M, WeightError>;

    /// Load model record for later use
    fn load_record(&self, device: &B::Device) -> Result<BiRefNetRecord<B>, WeightError>;

    /// Check if the weight source is available
    fn is_available(&self) -> bool;
}

/// Supported weight file formats
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightFormat {
    /// PyTorch .pt or .pth files
    PyTorch,
    /// PyTorch SafeTensors .safetensors files
    SafeTensors,
    /// Burn MessagePack .mpk files
    MessagePack,
    /// Burn Binary .bin files
    Binary,
    /// Auto-detect from file extension
    Auto,
}

impl WeightFormat {
    /// Detect format from file path
    pub fn from_path(path: &std::path::Path) -> Self {
        match path.extension().and_then(|s| s.to_str()) {
            Some("pt") | Some("pth") => Self::PyTorch,
            Some("safetensors") => Self::SafeTensors,
            Some("mpk") => Self::MessagePack,
            Some("bin") => Self::Binary,
            _ => Self::Auto,
        }
    }
}

impl ModelRecord for ManagedModel {
    fn name(&self) -> &ModelName {
        &self.name
    }

    fn config(&self) -> &Option<BiRefNetConfig> {
        &self.config
    }

    fn weight_source(&self) -> &WeightSource {
        &self.weights
    }
}

impl ManagedModel {
    /// Create a new model record with the given name, config, and weight source.
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

    /// List all available pretrained models
    pub fn list_available_models() -> Vec<&'static str> {
        MODEL_SPECS.keys().map(String::as_str).collect()
    }

    /// Get model spec by name
    pub fn get_model_spec(model_name: &str) -> Option<&'static ModelSpec> {
        MODEL_SPECS.get(model_name)
    }

    /// Create a model from a known model name with default settings
    ///
    /// # Errors
    /// Returns `WeightError::UnknownModel` if the model name is not found in the registry.
    pub fn from_pretrained(model_name: &str) -> Result<Self, WeightError> {
        let spec = MODEL_SPECS.get(model_name).ok_or_else(|| {
            let available_models = MODEL_SPECS
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .join(", ");
            WeightError::UnknownModel {
                model_name: model_name.to_owned(),
                available_models,
            }
        })?;

        let weights = WeightSource::Remote {
            repo_id: format!("ZhengPeng7/{}", spec.hf_model_id),
            filename: "model.safetensors".to_owned(),
        };

        Ok(Self::new(ModelName::new(model_name), None, weights))
    }

    pub fn get_weights_path(&self) -> Option<PathBuf> {
        match &self.weights {
            WeightSource::Remote { repo_id, filename } => sync::Api::new().map_or(None, |api| {
                api.repo(Repo::new(repo_id.clone(), RepoType::Model))
                    .get(filename)
                    .ok()
            }),
            WeightSource::Local { path } => Some(path.clone()),
        }
    }

    pub fn get_config(&self) -> BiRefNetConfig {
        if let Some(config) = &self.config {
            return config.clone();
        }

        // Look up in the catalog and use config builder
        MODEL_SPECS.get(self.name.as_str()).map_or_else(
            || {
                // Default config for unknown models
                create_config(
                    Task::General,
                    Backbone::SwinV1L,
                    Some(InterpolationStrategy::Bilinear),
                )
            },
            |spec| (spec.config_builder)(),
        )
    }

    /// Get model resolution
    pub fn get_resolution(&self) -> Option<(u32, u32)> {
        MODEL_SPECS
            .get(self.name.as_str())
            .map(|spec| spec.default_resolution)
    }

    /// Check if model supports dynamic resolution
    pub fn supports_dynamic_resolution(&self) -> bool {
        MODEL_SPECS
            .get(self.name.as_str())
            .is_some_and(|spec| spec.supports_dynamic_resolution)
    }
}

impl<B: Backend> ModelLoader<B> for ManagedModel {
    fn load_model<M: Module<B>>(&self, model: M, device: &B::Device) -> Result<M, WeightError> {
        let weights_path = self
            .get_weights_path()
            .ok_or_else(|| WeightError::FileSystemError {
                reason: "Failed to resolve weights path".to_string(),
            })?;

        if !weights_path.exists() {
            return Err(WeightError::FileSystemError {
                reason: format!("Weight file not found: {}", weights_path.display()),
            });
        }

        let format = WeightFormat::from_path(&weights_path);

        match format {
            WeightFormat::PyTorch => self.load_pytorch_model(model, &weights_path, device),
            WeightFormat::SafeTensors => self.load_safetensors_model(model, &weights_path, device),
            WeightFormat::MessagePack => self.load_messagepack_model(model, &weights_path, device),
            WeightFormat::Binary => self.load_binary_model(model, &weights_path, device),
            WeightFormat::Auto => {
                // Try different formats in order of preference
                if let Ok(model) = self.load_pytorch_model(model.clone(), &weights_path, device) {
                    Ok(model)
                } else if let Ok(model) =
                    self.load_safetensors_model(model.clone(), &weights_path, device)
                {
                    Ok(model)
                } else if let Ok(model) =
                    self.load_messagepack_model(model.clone(), &weights_path, device)
                {
                    Ok(model)
                } else {
                    self.load_binary_model(model, &weights_path, device)
                }
            }
        }
    }

    fn load_record(&self, device: &B::Device) -> Result<BiRefNetRecord<B>, WeightError> {
        let weights_path = self
            .get_weights_path()
            .ok_or_else(|| WeightError::FileSystemError {
                reason: "Failed to resolve weights path".to_string(),
            })?;

        if !weights_path.exists() {
            return Err(WeightError::FileSystemError {
                reason: format!("Weight file not found: {}", weights_path.display()),
            });
        }

        let format = WeightFormat::from_path(&weights_path);

        match format {
            WeightFormat::PyTorch => self.load_pytorch_record(&weights_path, device),
            WeightFormat::SafeTensors => self.load_safetensors_record(&weights_path, device),
            WeightFormat::MessagePack => self.load_messagepack_record(&weights_path, device),
            WeightFormat::Binary => self.load_binary_record(&weights_path, device),
            WeightFormat::Auto => {
                // Try different formats in order of preference
                if let Ok(record) = self.load_pytorch_record(&weights_path, device) {
                    Ok(record)
                } else if let Ok(record) = self.load_safetensors_record(&weights_path, device) {
                    Ok(record)
                } else if let Ok(record) = self.load_messagepack_record(&weights_path, device) {
                    Ok(record)
                } else {
                    self.load_binary_record(&weights_path, device)
                }
            }
        }
    }

    fn is_available(&self) -> bool {
        match &self.weights {
            WeightSource::Remote { .. } => {
                // Check if we can get the path (which includes downloading)
                self.get_weights_path().is_some()
            }
            WeightSource::Local { path } => path.exists(),
        }
    }
}

impl ManagedModel {
    fn load_pytorch_model<B: Backend, M: Module<B>>(
        &self,
        model: M,
        weights_path: &Path,
        device: &B::Device,
    ) -> Result<M, WeightError> {
        let load_args = PyTorchLoadArgs::new(weights_path.to_path_buf());
        let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();
        let record = recorder
            .load(load_args, device)
            .map_err(|e| WeightError::ModelLoadError {
                reason: format!("PyTorch model loading failed: {}", e),
            })?;
        Ok(model.load_record(record))
    }

    fn load_safetensors_model<B: Backend, M: Module<B>>(
        &self,
        model: M,
        weights_path: &Path,
        device: &B::Device,
    ) -> Result<M, WeightError> {
        let load_args = SafetensorsLoadArgs::new(weights_path.to_path_buf())
            .with_key_remap("decoder\\.conv_out1\\.0\\.(.+)", "decoder.conv_out1.$1")
            .with_key_remap(
                "decoder\\.gdt_convs_attn_([2-4])\\.0\\.(.+)",
                "decoder.gdt_convs_attn_$1.$2",
            )
            .with_key_remap(
                "decoder\\.gdt_convs_pred_([2-4])\\.0\\.(.+)",
                "decoder.gdt_convs_pred_$1.$2",
            )
            // Sequential
            .with_key_remap("bb\\.norm([0-3])\\.(.+)", "bb.norm_layers.$1.$2")
            .with_key_remap(
                "(.+?)\\.gdt_convs_([2-4])\\.0\\.(.+)",
                "$1.gdt_convs_$2.conv.$3",
            )
            .with_key_remap(
                "(.+?)\\.gdt_convs_([2-4])\\.1\\.(.+)",
                "$1.gdt_convs_$2.bn.$3",
            )
            .with_key_remap(
                "(.+)\\.global_avg_pool\\.1\\.(.+)",
                "$1.global_avg_pool.conv.$2",
            )
            .with_key_remap(
                "(.+)\\.global_avg_pool\\.2\\.(.+)",
                "$1.global_avg_pool.bn.$2",
            );

        let recorder = SafetensorsFileRecorder::<FullPrecisionSettings>::default();
        let record = recorder
            .load(load_args, device)
            .map_err(|e| WeightError::ModelLoadError {
                reason: format!("Safetensors model loading failed: {}", e),
            })?;
        Ok(model.load_record(record))
    }

    /// Load MessagePack format weights
    fn load_messagepack_model<B: Backend, M: Module<B>>(
        &self,
        model: M,
        weights_path: &Path,
        device: &B::Device,
    ) -> Result<M, WeightError> {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model
            .load_file(weights_path, &recorder, device)
            .map_err(|e| WeightError::ModelLoadError {
                reason: format!("MessagePack model loading failed: {}", e),
            })
    }

    /// Load Binary format weights
    fn load_binary_model<B: Backend, M: Module<B>>(
        &self,
        model: M,
        weights_path: &Path,
        device: &B::Device,
    ) -> Result<M, WeightError> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        model
            .load_file(weights_path, &recorder, device)
            .map_err(|e| WeightError::ModelLoadError {
                reason: format!("Binary model loading failed: {}", e),
            })
    }

    fn load_pytorch_record<B: Backend>(
        &self,
        weights_path: &Path,
        device: &B::Device,
    ) -> Result<BiRefNetRecord<B>, WeightError> {
        let load_args = PyTorchLoadArgs::new(weights_path.to_path_buf());
        let recorder = PyTorchFileRecorder::<FullPrecisionSettings>::default();
        recorder
            .load(load_args, device)
            .map_err(|e| WeightError::RecordLoadError {
                reason: format!("PyTorch record loading failed: {}", e),
            })
    }

    fn load_safetensors_record<B: Backend>(
        &self,
        weights_path: &Path,
        device: &B::Device,
    ) -> Result<BiRefNetRecord<B>, WeightError> {
        let load_args = SafetensorsLoadArgs::new(weights_path.to_path_buf());
        let recorder = SafetensorsFileRecorder::<FullPrecisionSettings>::default();
        recorder
            .load(load_args, device)
            .map_err(|e| WeightError::RecordLoadError {
                reason: format!("Safetensors record loading failed: {}", e),
            })
    }

    /// Load MessagePack record
    fn load_messagepack_record<B: Backend>(
        &self,
        weights_path: &Path,
        device: &B::Device,
    ) -> Result<BiRefNetRecord<B>, WeightError> {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        recorder
            .load(weights_path.to_path_buf(), device)
            .map_err(|e| WeightError::RecordLoadError {
                reason: format!("MessagePack record loading failed: {}", e),
            })
    }

    /// Load Binary record
    fn load_binary_record<B: Backend>(
        &self,
        weights_path: &Path,
        device: &B::Device,
    ) -> Result<BiRefNetRecord<B>, WeightError> {
        let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
        recorder
            .load(weights_path.to_path_buf(), device)
            .map_err(|e| WeightError::RecordLoadError {
                reason: format!("Binary record loading failed: {}", e),
            })
    }
}

/// Extension trait for BiRefNet to provide convenient weight loading methods
///
/// This trait adds weight loading capabilities to BiRefNet models without creating
/// circular dependencies between crates.
pub trait BiRefNetWeightLoading<B: Backend> {
    /// Load weights from a managed model
    ///
    /// This convenience method allows loading pre-trained weights from various formats
    /// supported by the ManagedModel system.
    ///
    /// # Arguments
    /// * `managed_model` - A ManagedModel instance containing weight source information
    /// * `device` - The device to load the weights onto
    ///
    /// # Returns
    /// An updated BiRefNet instance with new weights
    ///
    /// # Errors
    /// Returns an error if weight loading fails or if weights are incompatible
    fn load_weights_from_managed_model<M>(
        self,
        managed_model: &M,
        device: &B::Device,
    ) -> Result<Self, WeightError>
    where
        Self: Sized + Module<B>,
        M: ModelLoader<B>;

    /// Create a new BiRefNet instance from a managed model
    ///
    /// This static method creates and loads a BiRefNet model from a managed model
    /// configuration and weights.
    ///
    /// # Arguments
    /// * `managed_model` - A ManagedModel instance containing configuration and weights
    /// * `device` - The device to create the model on
    ///
    /// # Returns
    /// A new BiRefNet instance with loaded weights
    ///
    /// # Errors
    /// Returns an error if model creation or weight loading fails
    fn from_managed_model<M>(managed_model: &M, device: &B::Device) -> Result<Self, WeightError>
    where
        Self: Sized,
        M: ModelLoader<B> + ModelRecord;
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    use super::*;

    type TestBackend = NdArray;

    #[test]
    fn test_managed_model_creation() {
        let model_name = ModelName::new("test-model");
        let weights = WeightSource::Local {
            path: PathBuf::from("/nonexistent/path/model.pt"),
        };
        let managed_model = ManagedModel::new(model_name.clone(), None, weights);

        assert_eq!(managed_model.name().as_str(), "test-model");
        assert!(managed_model.config().is_none());
        assert!(!<ManagedModel as ModelLoader<TestBackend>>::is_available(
            &managed_model
        ));
    }

    #[test]
    fn test_pretrained_model_names() {
        let models = ManagedModel::list_available_models();
        assert!(!models.is_empty());
        assert!(models.contains(&"General"));
        assert!(models.contains(&"General-HR"));
        assert!(models.contains(&"Matting"));
    }

    #[test]
    fn test_from_pretrained_invalid_model() {
        let result = ManagedModel::from_pretrained("NonexistentModel");
        assert!(result.is_err());
        match result.unwrap_err() {
            WeightError::UnknownModel { model_name, .. } => {
                assert_eq!(model_name, "NonexistentModel");
            }
            _ => panic!("Expected UnknownModel error"),
        }
    }

    #[test]
    fn test_from_pretrained_valid_model() {
        let result = ManagedModel::from_pretrained("General");
        assert!(result.is_ok());
        let model = result.unwrap();
        assert_eq!(model.name().as_str(), "General");
    }

    #[test]
    fn test_weight_format_detection() {
        let pytorch_path = PathBuf::from("model.pt");
        assert_eq!(
            WeightFormat::from_path(&pytorch_path),
            WeightFormat::PyTorch
        );

        let safetensors_path = PathBuf::from("model.safetensors");
        assert_eq!(
            WeightFormat::from_path(&safetensors_path),
            WeightFormat::SafeTensors
        );

        let mpk_path = PathBuf::from("model.mpk");
        assert_eq!(
            WeightFormat::from_path(&mpk_path),
            WeightFormat::MessagePack
        );

        let bin_path = PathBuf::from("model.bin");
        assert_eq!(WeightFormat::from_path(&bin_path), WeightFormat::Binary);

        let unknown_path = PathBuf::from("model.unknown");
        assert_eq!(WeightFormat::from_path(&unknown_path), WeightFormat::Auto);
    }

    #[test]
    fn test_model_spec_resolution() {
        let spec = ManagedModel::get_model_spec("General");
        assert!(spec.is_some());
        let spec = spec.unwrap();
        assert_eq!(spec.hf_model_id, "BiRefNet");
        assert_eq!(spec.default_resolution, (1024, 1024));
        assert!(!spec.supports_dynamic_resolution);
    }

    #[test]
    fn test_dynamic_resolution_models() {
        let spec = ManagedModel::get_model_spec("General-dynamic");
        assert!(spec.is_some());
        let spec = spec.unwrap();
        assert!(spec.supports_dynamic_resolution);

        let spec = ManagedModel::get_model_spec("Matting-dynamic");
        assert!(spec.is_some());
        let spec = spec.unwrap();
        assert!(spec.supports_dynamic_resolution);
    }
}

impl<B: Backend> BiRefNetWeightLoading<B> for birefnet_model::BiRefNet<B> {
    fn load_weights_from_managed_model<M>(
        self,
        managed_model: &M,
        device: &B::Device,
    ) -> Result<Self, WeightError>
    where
        M: ModelLoader<B>,
    {
        managed_model.load_model(self, device)
    }

    fn from_managed_model<M>(managed_model: &M, device: &B::Device) -> Result<Self, WeightError>
    where
        M: ModelLoader<B> + ModelRecord,
    {
        // Get the configuration from the managed model
        let config = if let Some(config) = managed_model.config() {
            config.clone()
        } else {
            // Default configuration if none provided
            create_config(
                Task::General,
                Backbone::SwinV1L,
                Some(InterpolationStrategy::Bilinear),
            )
        };

        // Initialize a model with the configuration
        let model = config
            .init(device)
            .map_err(|e| WeightError::ModelLoadError {
                reason: format!("Failed to initialize model: {}", e),
            })?;

        // Load weights into the model
        managed_model.load_model(model, device)
    }
}
