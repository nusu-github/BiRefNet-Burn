//! Model registry and weight format specifications for pretrained `BiRefNet` checkpoints.

use std::{
    convert::Infallible,
    fmt,
    path::{Path, PathBuf},
    str::FromStr,
};

use birefnet_model::{
    Backbone, BackboneConfig, BiRefNetConfig, DecoderConfig, InterpolationStrategy, ModelConfig,
    Task, TaskConfig,
};

/// Canonical model identifier string.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelName(String);

impl ModelName {
    /// Creates a new [`ModelName`] from any string-like value.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }

    /// Returns the model identifier as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl AsRef<str> for ModelName {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl FromStr for ModelName {
    type Err = Infallible;

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

impl fmt::Display for ModelName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Supported checkpoint serialization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightFormat {
    /// `PyTorch` pickle checkpoint (`.pt`, `.pth`).
    PyTorch,
    /// `SafeTensors` checkpoint (`.safetensors`).
    SafeTensors,
    /// Native `Burnpack` checkpoint (`.bpk`).
    Burnpack,
    /// Detect format by attempting `Burnpack`, `SafeTensors`, then `PyTorch`.
    Auto,
}

impl WeightFormat {
    /// Infers the checkpoint format from a file extension.
    #[must_use]
    pub fn from_path(path: &Path) -> Self {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some(ext) if ext.eq_ignore_ascii_case("pt") || ext.eq_ignore_ascii_case("pth") => {
                Self::PyTorch
            }
            Some(ext) if ext.eq_ignore_ascii_case("safetensors") => Self::SafeTensors,
            Some(ext) if ext.eq_ignore_ascii_case("bpk") => Self::Burnpack,
            _ => Self::Auto,
        }
    }
}

/// Location where model weights are retrieved or read from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightSource {
    /// Remote repository on `HuggingFace` Hub.
    Remote {
        /// Repository identifier (`"<owner>/<model>"`).
        repo_id: String,
        /// Checkpoint file path inside the repository.
        filename: String,
    },
    /// Local checkpoint file on disk.
    Local {
        /// Filesystem path to the checkpoint.
        path: PathBuf,
    },
}

/// Metadata and default architecture preset for a registered pretrained model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelSpec {
    /// User-facing registry name (for example `"General"` or `"Matting-HR"`).
    pub name: &'static str,
    /// Model repository name under the `ZhengPeng7` namespace on `HuggingFace` Hub.
    pub hf_model_id: &'static str,
    /// Default `(height, width)` input resolution for inference.
    pub default_resolution: (u32, u32),
    /// Whether the model supports dynamic input resolutions during inference.
    pub supports_dynamic_resolution: bool,
    /// Preset training task type.
    pub task: Task,
    /// Preset encoder backbone variant.
    pub backbone: Backbone,
}

impl ModelSpec {
    /// Builds the default [`BiRefNetConfig`] corresponding to this model preset.
    #[must_use]
    pub fn build_config(&self) -> BiRefNetConfig {
        let model_config = ModelConfig::new(InterpolationStrategy::Bilinear)
            .with_task(TaskConfig::new().with_task(self.task))
            .with_backbone(BackboneConfig::new().with_backbone(self.backbone))
            .with_decoder(DecoderConfig::new());
        BiRefNetConfig::new(model_config)
    }

    /// Returns the full `HuggingFace` Hub repository identifier (`ZhengPeng7/<id>`).
    #[must_use]
    pub fn repo_id(&self) -> String {
        format!("ZhengPeng7/{}", self.hf_model_id)
    }
}

/// Complete ordered catalog of supported pretrained `BiRefNet` checkpoints.
pub const MODEL_SPECS: &[ModelSpec] = &[
    ModelSpec {
        name: "General",
        hf_model_id: "BiRefNet",
        default_resolution: (1024, 1024),
        supports_dynamic_resolution: false,
        task: Task::General,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "General-HR",
        hf_model_id: "BiRefNet_HR",
        default_resolution: (2048, 2048),
        supports_dynamic_resolution: false,
        task: Task::General2k,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "Matting-HR",
        hf_model_id: "BiRefNet_HR-matting",
        default_resolution: (2048, 2048),
        supports_dynamic_resolution: false,
        task: Task::Matting,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "Matting",
        hf_model_id: "BiRefNet-matting",
        default_resolution: (1024, 1024),
        supports_dynamic_resolution: false,
        task: Task::Matting,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "Portrait",
        hf_model_id: "BiRefNet-portrait",
        default_resolution: (1024, 1024),
        supports_dynamic_resolution: false,
        task: Task::Matting,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "General-reso_512",
        hf_model_id: "BiRefNet_512x512",
        default_resolution: (512, 512),
        supports_dynamic_resolution: false,
        task: Task::General,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "General-Lite",
        hf_model_id: "BiRefNet_lite",
        default_resolution: (1024, 1024),
        supports_dynamic_resolution: false,
        task: Task::General,
        backbone: Backbone::SwinV1T,
    },
    ModelSpec {
        name: "General-Lite-2K",
        hf_model_id: "BiRefNet_lite-2K",
        default_resolution: (2048, 2048),
        supports_dynamic_resolution: false,
        task: Task::General2k,
        backbone: Backbone::SwinV1T,
    },
    ModelSpec {
        name: "DIS",
        hf_model_id: "BiRefNet-DIS5K",
        default_resolution: (1024, 1024),
        supports_dynamic_resolution: false,
        task: Task::DIS5K,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "HRSOD",
        hf_model_id: "BiRefNet-HRSOD",
        default_resolution: (1024, 1024),
        supports_dynamic_resolution: false,
        task: Task::HRSOD,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "COD",
        hf_model_id: "BiRefNet-COD",
        default_resolution: (1024, 1024),
        supports_dynamic_resolution: false,
        task: Task::COD,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "DIS-TR_TEs",
        hf_model_id: "BiRefNet-DIS5K-TR_TEs",
        default_resolution: (1024, 1024),
        supports_dynamic_resolution: false,
        task: Task::DIS5K,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "General-legacy",
        hf_model_id: "BiRefNet-legacy",
        default_resolution: (1024, 1024),
        supports_dynamic_resolution: false,
        task: Task::General,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "General-dynamic",
        hf_model_id: "BiRefNet_dynamic",
        default_resolution: (1024, 1024),
        supports_dynamic_resolution: true,
        task: Task::General,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "Matting-dynamic",
        hf_model_id: "BiRefNet_dynamic-matting",
        default_resolution: (1024, 1024),
        supports_dynamic_resolution: true,
        task: Task::Matting,
        backbone: Backbone::SwinV1L,
    },
];

/// Returns the ordered list of all available pretrained model names.
#[must_use]
pub fn list_available_models() -> Vec<&'static str> {
    MODEL_SPECS.iter().map(|spec| spec.name).collect()
}

/// Looks up a registered [`ModelSpec`] by its exact preset name.
#[must_use]
pub fn get_model_spec(name: &str) -> Option<&'static ModelSpec> {
    MODEL_SPECS.iter().find(|spec| spec.name == name)
}
