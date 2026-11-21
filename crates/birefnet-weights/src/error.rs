//! Error types for model weight resolution and loading.

use std::{backtrace::Backtrace, fmt, path::Path};

use birefnet_model::ModelError;

type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ErrorKind {
    UnknownModel,
    NotFound,
    UnsupportedFormat,
    #[cfg(feature = "hub")]
    Hub,
    Io,
    Model,
    Load,
}

#[derive(Debug)]
struct ErrorInner {
    kind: ErrorKind,
    message: String,
    source: Option<BoxError>,
    backtrace: Backtrace,
}

/// Error returned by model registry, hub fetching, and checkpoint loading operations.
pub struct WeightError {
    inner: Box<ErrorInner>,
}

const _: () = {
    const fn assert_error_bounds<E: std::error::Error + Send + Sync + 'static>() {}
    assert_error_bounds::<WeightError>();
};

impl WeightError {
    fn new(kind: ErrorKind, message: impl Into<String>, source: Option<BoxError>) -> Self {
        Self {
            inner: Box::new(ErrorInner {
                kind,
                message: message.into(),
                source,
                backtrace: Backtrace::capture(),
            }),
        }
    }

    pub(crate) fn unknown_model(model_name: &str, available: &[&str]) -> Self {
        Self::new(
            ErrorKind::UnknownModel,
            format!(
                "unknown model `{model_name}`; available models: {}",
                available.join(", ")
            ),
            None,
        )
    }

    pub(crate) fn not_found(path: &Path) -> Self {
        Self::new(
            ErrorKind::NotFound,
            format!("weight file not found: `{}`", path.display()),
            None,
        )
    }

    pub(crate) fn load(message: impl Into<String>, source: Option<BoxError>) -> Self {
        Self::new(ErrorKind::Load, message, source)
    }

    pub(crate) fn unsupported_format(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::UnsupportedFormat, message, None)
    }

    /// Returns `true` if the requested model name is not present in the registry.
    #[must_use]
    pub fn is_unknown_model(&self) -> bool {
        self.inner.kind == ErrorKind::UnknownModel
    }

    /// Returns `true` if a weight file or cached artifact could not be found.
    #[must_use]
    pub fn is_not_found(&self) -> bool {
        self.inner.kind == ErrorKind::NotFound
    }

    /// Returns `true` if the weight format or source is unsupported in the current build.
    #[must_use]
    pub fn is_unsupported_format(&self) -> bool {
        self.inner.kind == ErrorKind::UnsupportedFormat
    }

    /// Returns `true` if the error originated from `HuggingFace` Hub access.
    #[must_use]
    pub fn is_hub(&self) -> bool {
        #[cfg(feature = "hub")]
        {
            self.inner.kind == ErrorKind::Hub
        }
        #[cfg(not(feature = "hub"))]
        {
            false
        }
    }

    /// Returns `true` if the error was caused by an I/O failure.
    #[must_use]
    pub fn is_io(&self) -> bool {
        self.inner.kind == ErrorKind::Io
    }

    /// Returns `true` if model initialization failed while constructing an architecture.
    #[must_use]
    pub fn is_model(&self) -> bool {
        self.inner.kind == ErrorKind::Model
    }

    /// Returns `true` if deserializing or applying checkpoint tensors failed.
    #[must_use]
    pub fn is_load(&self) -> bool {
        self.inner.kind == ErrorKind::Load
    }

    /// Captures the backtrace recorded when this error was constructed.
    #[must_use]
    pub fn backtrace(&self) -> &Backtrace {
        &self.inner.backtrace
    }
}

impl fmt::Debug for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WeightError")
            .field("kind", &self.inner.kind)
            .field("message", &self.inner.message)
            .field("source", &self.inner.source)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for WeightError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.inner.message)
    }
}

impl std::error::Error for WeightError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self.inner.source.as_deref() {
            Some(err) => Some(err),
            None => None,
        }
    }
}

impl From<ModelError> for WeightError {
    fn from(source: ModelError) -> Self {
        let message = format!("failed to initialize model configuration: {source}");
        Self::new(ErrorKind::Model, message, Some(Box::new(source)))
    }
}

impl From<std::io::Error> for WeightError {
    fn from(source: std::io::Error) -> Self {
        let kind = if source.kind() == std::io::ErrorKind::NotFound {
            ErrorKind::NotFound
        } else {
            ErrorKind::Io
        };
        let message = format!("weight I/O error: {source}");
        Self::new(kind, message, Some(Box::new(source)))
    }
}

#[cfg(feature = "hub")]
impl From<hf_hub::HFError> for WeightError {
    fn from(source: hf_hub::HFError) -> Self {
        let message = format!("HuggingFace Hub error: {source}");
        Self::new(ErrorKind::Hub, message, Some(Box::new(source)))
    }
}

impl From<burn::store::RecordError> for WeightError {
    fn from(source: burn::store::RecordError) -> Self {
        let message = format!("Burnpack record error: {source}");
        Self::new(ErrorKind::Load, message, Some(Box::new(source)))
    }
}

impl From<burn::store::SafetensorsStoreError> for WeightError {
    fn from(source: burn::store::SafetensorsStoreError) -> Self {
        let message = format!("SafeTensors store error: {source}");
        Self::new(ErrorKind::Load, message, Some(Box::new(source)))
    }
}

impl From<burn::store::PytorchStoreError> for WeightError {
    fn from(source: burn::store::PytorchStoreError) -> Self {
        let message = format!("PyTorch store error: {source}");
        Self::new(ErrorKind::Load, message, Some(Box::new(source)))
    }
}
