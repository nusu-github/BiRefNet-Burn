//! Canonical error type for [`crate`].

use std::{
    backtrace::{Backtrace, BacktraceStatus},
    error::Error,
    fmt,
    path::{Path, PathBuf},
};

use birefnet_model::ModelError;
use burn::store::{PytorchStoreError, RecordError, SafetensorsStoreError};

/// Error returned by the model registry, weight fetching, and checkpoint loading.
#[derive(Debug)]
pub struct WeightError {
    inner: Box<ErrorInner>,
}

#[derive(Debug)]
struct ErrorInner {
    kind: ErrorKind,
    path: Option<PathBuf>,
    backtrace: Backtrace,
}

#[derive(Debug)]
enum ErrorKind {
    UnknownModel {
        name: String,
    },
    InvalidName,
    NotFound,
    UnsupportedFormat {
        message: String,
    },
    #[cfg(feature = "hub")]
    Hub(hf_hub::HFError),
    Io(std::io::Error),
    Model(ModelError),
    Burnpack(RecordError),
    SafeTensors(SafetensorsStoreError),
    PyTorch(PytorchStoreError),
}

impl WeightError {
    fn new(kind: ErrorKind, path: Option<&Path>) -> Self {
        Self {
            inner: Box::new(ErrorInner {
                kind,
                path: path.map(Path::to_path_buf),
                backtrace: Backtrace::capture(),
            }),
        }
    }

    pub(crate) fn unknown_model(name: &str) -> Self {
        Self::new(
            ErrorKind::UnknownModel {
                name: name.to_owned(),
            },
            None,
        )
    }

    pub(crate) fn invalid_name() -> Self {
        Self::new(ErrorKind::InvalidName, None)
    }

    pub(crate) fn not_found(path: &Path) -> Self {
        Self::new(ErrorKind::NotFound, Some(path))
    }

    pub(crate) fn unsupported_format(message: impl Into<String>, path: Option<&Path>) -> Self {
        Self::new(
            ErrorKind::UnsupportedFormat {
                message: message.into(),
            },
            path,
        )
    }

    pub(crate) fn io(path: &Path, source: std::io::Error) -> Self {
        Self::new(ErrorKind::Io(source), Some(path))
    }

    /// Attaches `path` to an error that does not carry one yet.
    pub(crate) fn at(mut self, path: &Path) -> Self {
        self.inner.path.get_or_insert_with(|| path.to_path_buf());
        self
    }

    /// Returns `true` if the requested name is not a registered pretrained model.
    #[must_use]
    pub fn is_unknown_model(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::UnknownModel { .. })
    }

    /// Returns `true` if a model name was empty or only whitespace.
    #[must_use]
    pub fn is_invalid_name(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::InvalidName)
    }

    /// Returns `true` if a checkpoint file or remote artifact does not exist.
    #[must_use]
    pub fn is_not_found(&self) -> bool {
        match &self.inner.kind {
            ErrorKind::NotFound => true,
            ErrorKind::Io(source) => source.kind() == std::io::ErrorKind::NotFound,
            _ => false,
        }
    }

    /// Returns `true` if the checkpoint format could not be determined or is unsupported.
    #[must_use]
    pub fn is_unsupported_format(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::UnsupportedFormat { .. })
    }

    /// Returns `true` if a Hugging Face Hub request failed.
    #[must_use]
    pub fn is_hub(&self) -> bool {
        cfg_select! {
            feature = "hub" => matches!(self.inner.kind, ErrorKind::Hub(_)),
            _ => false,
        }
    }

    /// Returns `true` if reading a checkpoint from disk failed.
    #[must_use]
    pub fn is_io(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::Io(_))
    }

    /// Returns `true` if building the model architecture failed.
    #[must_use]
    pub fn is_model(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::Model(_))
    }

    /// Returns `true` if a checkpoint could not be decoded or applied to the model.
    #[must_use]
    pub fn is_load(&self) -> bool {
        matches!(
            self.inner.kind,
            ErrorKind::Burnpack(_) | ErrorKind::SafeTensors(_) | ErrorKind::PyTorch(_)
        )
    }

    /// Returns the checkpoint path involved, if known.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.inner.path.as_deref()
    }

    /// Returns the captured [`Backtrace`] for this error.
    #[must_use]
    pub fn backtrace(&self) -> &Backtrace {
        &self.inner.backtrace
    }
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner.kind {
            ErrorKind::UnknownModel { name } => {
                write!(
                    f,
                    "unknown pretrained model `{name}` (see list_available_models)"
                )
            }
            ErrorKind::InvalidName => f.write_str("model names must not be empty"),
            ErrorKind::NotFound => f.write_str("weight file not found"),
            ErrorKind::UnsupportedFormat { message } => {
                write!(f, "unsupported checkpoint format: {message}")
            }
            #[cfg(feature = "hub")]
            ErrorKind::Hub(_) => f.write_str("Hugging Face Hub request failed"),
            ErrorKind::Io(_) => f.write_str("failed to read checkpoint"),
            ErrorKind::Model(_) => f.write_str("failed to build the model architecture"),
            ErrorKind::Burnpack(_) => f.write_str("failed to load Burnpack checkpoint"),
            ErrorKind::SafeTensors(_) => f.write_str("failed to load SafeTensors checkpoint"),
            ErrorKind::PyTorch(_) => f.write_str("failed to load PyTorch checkpoint"),
        }?;
        if let Some(path) = &self.inner.path {
            write!(f, " ('{}')", path.display())?;
        }
        if self.inner.backtrace.status() == BacktraceStatus::Captured {
            write!(f, "\n\nstack backtrace:\n{}", self.inner.backtrace)?;
        }
        Ok(())
    }
}

impl Error for WeightError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.inner.kind {
            #[cfg(feature = "hub")]
            ErrorKind::Hub(source) => Some(source),
            ErrorKind::Io(source) => Some(source),
            ErrorKind::Model(source) => Some(source),
            ErrorKind::Burnpack(source) => Some(source),
            ErrorKind::SafeTensors(source) => Some(source),
            ErrorKind::PyTorch(source) => Some(source),
            ErrorKind::UnknownModel { .. }
            | ErrorKind::InvalidName
            | ErrorKind::NotFound
            | ErrorKind::UnsupportedFormat { .. } => None,
        }
    }
}

macro_rules! from_source {
    ($($(#[$attr:meta])* $source:ty => $kind:ident),+ $(,)?) => {
        $($(#[$attr])* impl From<$source> for WeightError {
            fn from(source: $source) -> Self {
                Self::new(ErrorKind::$kind(source), None)
            }
        })+
    };
}

from_source! {
    #[cfg(feature = "hub")]
    hf_hub::HFError => Hub,
    std::io::Error => Io,
    ModelError => Model,
    RecordError => Burnpack,
    SafetensorsStoreError => SafeTensors,
    PytorchStoreError => PyTorch,
}

const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<WeightError>();
};
