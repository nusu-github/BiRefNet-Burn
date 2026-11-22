//! Model registry and weight format specifications for pretrained `BiRefNet` checkpoints.

use std::{
    fmt,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    str::FromStr,
};

use birefnet_model::{
    Backbone, BackboneConfig, BiRefNetConfig, DecoderConfig, InterpolationStrategy, ModelConfig,
    Task, TaskConfig,
};

use crate::error::WeightError;

/// Non-empty model identifier (a registry preset name or a user label for local weights).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelName(String);

impl ModelName {
    /// Creates a model name.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError`] if `name` is empty or only whitespace.
    pub fn new(name: impl Into<String>) -> Result<Self, WeightError> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(WeightError::invalid_name());
        }
        Ok(Self(name))
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
    type Err = WeightError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::new(s)
    }
}

impl fmt::Display for ModelName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Supported checkpoint serialization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WeightFormat {
    /// `PyTorch` checkpoint (`torch.save` zip archive or legacy pickle).
    PyTorch,
    /// `SafeTensors` checkpoint.
    SafeTensors,
    /// Native Burnpack checkpoint written by `burn::module::Module::save_file`.
    Burnpack,
}

/// Burnpack files start with the little-endian `u32` `0x4255524E` ("BURN").
const BURNPACK_MAGIC: [u8; 4] = 0x4255_524E_u32.to_le_bytes();
/// `torch.save` writes a zip archive (`PK\x03\x04`).
const ZIP_MAGIC: [u8; 4] = *b"PK\x03\x04";
/// Legacy `torch.save` files are raw pickles, which start with the `PROTO` opcode.
const PICKLE_PROTO: u8 = 0x80;
/// Bytes needed to recognize every supported format.
const SNIFF_LEN: usize = 9;

impl WeightFormat {
    /// Detects the format from the first bytes of a checkpoint.
    ///
    /// SafeTensors is recognized by its `u64` header length followed by a JSON object.
    #[must_use]
    pub fn detect(prefix: &[u8]) -> Option<Self> {
        if prefix.starts_with(&BURNPACK_MAGIC) {
            Some(Self::Burnpack)
        } else if prefix.starts_with(&ZIP_MAGIC) || prefix.first() == Some(&PICKLE_PROTO) {
            Some(Self::PyTorch)
        } else if prefix.len() >= SNIFF_LEN && prefix[8] == b'{' {
            Some(Self::SafeTensors)
        } else {
            None
        }
    }

    /// Detects the format of the checkpoint at `path` from its first bytes.
    ///
    /// # Errors
    ///
    /// Returns [`WeightError`] if the file cannot be read or its format is not recognized.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, WeightError> {
        let path = path.as_ref();
        let mut prefix = Vec::with_capacity(SNIFF_LEN);
        File::open(path)
            .and_then(|file| file.take(SNIFF_LEN as u64).read_to_end(&mut prefix))
            .map_err(|err| WeightError::io(path, err))?;
        Self::detect(&prefix).ok_or_else(|| {
            WeightError::unsupported_format(
                "expected a Burnpack, SafeTensors, or PyTorch checkpoint",
                Some(path),
            )
        })
    }
}

impl fmt::Display for WeightFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::PyTorch => "PyTorch",
            Self::SafeTensors => "SafeTensors",
            Self::Burnpack => "Burnpack",
        })
    }
}

/// Location where model weights are retrieved or read from.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum WeightSource {
    /// File in a Hugging Face Hub repository.
    Remote {
        /// Repository identifier (`"<owner>/<model>"`).
        repo_id: String,
        /// File path inside the repository.
        filename: String,
    },
    /// Local checkpoint file on disk.
    Local {
        /// Filesystem path to the checkpoint.
        path: PathBuf,
    },
}

impl fmt::Display for WeightSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Remote { repo_id, filename } => write!(f, "hf://{repo_id}/{filename}"),
            Self::Local { path } => write!(f, "{}", path.display()),
        }
    }
}

/// Metadata and default architecture preset for a registered pretrained model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelSpec {
    /// User-facing registry name (for example `"General"` or `"Matting-HR"`).
    pub name: &'static str,
    /// Model repository name under the `ZhengPeng7` namespace on Hugging Face Hub.
    pub hf_model_id: &'static str,
    /// Default `[height, width]` inference resolution.
    pub default_resolution: [u32; 2],
    /// Whether the model supports dynamic input resolutions during inference.
    pub supports_dynamic_resolution: bool,
    /// Preset training task type.
    pub task: Task,
    /// Preset encoder backbone variant.
    pub backbone: Backbone,
}

impl fmt::Display for ModelSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [height, width] = self.default_resolution;
        write!(
            f,
            "{} ({}, {}, {height}x{width})",
            self.name, self.backbone, self.task
        )
    }
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

    /// Returns the full Hugging Face Hub repository identifier (`ZhengPeng7/<id>`).
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
        default_resolution: [1024, 1024],
        supports_dynamic_resolution: false,
        task: Task::General,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "General-HR",
        hf_model_id: "BiRefNet_HR",
        default_resolution: [2048, 2048],
        supports_dynamic_resolution: false,
        task: Task::General2k,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "Matting-HR",
        hf_model_id: "BiRefNet_HR-matting",
        default_resolution: [2048, 2048],
        supports_dynamic_resolution: false,
        task: Task::Matting,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "Matting",
        hf_model_id: "BiRefNet-matting",
        default_resolution: [1024, 1024],
        supports_dynamic_resolution: false,
        task: Task::Matting,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "Portrait",
        hf_model_id: "BiRefNet-portrait",
        default_resolution: [1024, 1024],
        supports_dynamic_resolution: false,
        task: Task::Matting,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "General-reso_512",
        hf_model_id: "BiRefNet_512x512",
        default_resolution: [512, 512],
        supports_dynamic_resolution: false,
        task: Task::General,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "General-Lite",
        hf_model_id: "BiRefNet_lite",
        default_resolution: [1024, 1024],
        supports_dynamic_resolution: false,
        task: Task::General,
        backbone: Backbone::SwinV1T,
    },
    ModelSpec {
        name: "General-Lite-2K",
        hf_model_id: "BiRefNet_lite-2K",
        default_resolution: [2048, 2048],
        supports_dynamic_resolution: false,
        task: Task::General2k,
        backbone: Backbone::SwinV1T,
    },
    ModelSpec {
        name: "DIS",
        hf_model_id: "BiRefNet-DIS5K",
        default_resolution: [1024, 1024],
        supports_dynamic_resolution: false,
        task: Task::DIS5K,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "HRSOD",
        hf_model_id: "BiRefNet-HRSOD",
        default_resolution: [1024, 1024],
        supports_dynamic_resolution: false,
        task: Task::HRSOD,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "COD",
        hf_model_id: "BiRefNet-COD",
        default_resolution: [1024, 1024],
        supports_dynamic_resolution: false,
        task: Task::COD,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "DIS-TR_TEs",
        hf_model_id: "BiRefNet-DIS5K-TR_TEs",
        default_resolution: [1024, 1024],
        supports_dynamic_resolution: false,
        task: Task::DIS5K,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "General-legacy",
        hf_model_id: "BiRefNet-legacy",
        default_resolution: [1024, 1024],
        supports_dynamic_resolution: false,
        task: Task::General,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "General-dynamic",
        hf_model_id: "BiRefNet_dynamic",
        default_resolution: [1024, 1024],
        supports_dynamic_resolution: true,
        task: Task::General,
        backbone: Backbone::SwinV1L,
    },
    ModelSpec {
        name: "Matting-dynamic",
        hf_model_id: "BiRefNet_dynamic-matting",
        default_resolution: [1024, 1024],
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
pub fn model_spec(name: impl AsRef<str>) -> Option<&'static ModelSpec> {
    let name = name.as_ref();
    MODEL_SPECS.iter().find(|spec| spec.name == name)
}
