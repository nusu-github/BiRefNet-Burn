//! Error types for `BiRefNet` inference and mask postprocessing.

use std::backtrace::Backtrace;
use std::fmt;

use birefnet_model::ModelError;
use birefnet_util::UtilError;
use burn::tensor::TensorReadError;

type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ErrorKind {
    Model,
    Util,
    Image,
    Io,
    InvalidInput,
    TensorRead,
}

#[derive(Debug)]
struct ErrorInner {
    kind: ErrorKind,
    message: String,
    source: Option<BoxError>,
    backtrace: Backtrace,
}

/// Error returned by inference and mask postprocessing operations.
pub struct InferenceError {
    inner: Box<ErrorInner>,
}

const _: () = {
    const fn assert_error_bounds<E: std::error::Error + Send + Sync + 'static>() {}
    assert_error_bounds::<InferenceError>();
};

impl InferenceError {
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

    pub(crate) fn invalid_input(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::InvalidInput, message, None)
    }

    /// Returns `true` if the underlying `BiRefNet` model forward pass failed.
    #[must_use]
    pub fn is_model(&self) -> bool {
        self.inner.kind == ErrorKind::Model
    }

    /// Returns `true` if image tensor conversion or normalization failed.
    #[must_use]
    pub fn is_util(&self) -> bool {
        self.inner.kind == ErrorKind::Util
    }

    /// Returns `true` if image decoding or encoding failed.
    #[must_use]
    pub fn is_image(&self) -> bool {
        self.inner.kind == ErrorKind::Image
    }

    /// Returns `true` if filesystem I/O failed.
    #[must_use]
    pub fn is_io(&self) -> bool {
        self.inner.kind == ErrorKind::Io
    }

    /// Returns `true` if input dimensions or mask shapes were invalid.
    #[must_use]
    pub fn is_invalid_input(&self) -> bool {
        self.inner.kind == ErrorKind::InvalidInput
    }

    /// Returns `true` if copying tensor data from the backend device failed.
    #[must_use]
    pub fn is_tensor_read(&self) -> bool {
        self.inner.kind == ErrorKind::TensorRead
    }

    /// Returns the backtrace recorded when this error was constructed.
    #[must_use]
    pub fn backtrace(&self) -> &Backtrace {
        &self.inner.backtrace
    }
}

impl fmt::Debug for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InferenceError")
            .field("kind", &self.inner.kind)
            .field("message", &self.inner.message)
            .field("source", &self.inner.source)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.inner.message)
    }
}

impl std::error::Error for InferenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        let err = self.inner.source.as_deref()?;
        Some(err)
    }
}

impl From<ModelError> for InferenceError {
    fn from(source: ModelError) -> Self {
        let message = format!("model forward error: {source}");
        Self::new(ErrorKind::Model, message, Some(Box::new(source)))
    }
}

impl From<UtilError> for InferenceError {
    fn from(source: UtilError) -> Self {
        let message = format!("image utility error: {source}");
        Self::new(ErrorKind::Util, message, Some(Box::new(source)))
    }
}

impl From<image::ImageError> for InferenceError {
    fn from(source: image::ImageError) -> Self {
        let message = format!("image codec error: {source}");
        Self::new(ErrorKind::Image, message, Some(Box::new(source)))
    }
}

impl From<std::io::Error> for InferenceError {
    fn from(source: std::io::Error) -> Self {
        let message = format!("inference I/O error: {source}");
        Self::new(ErrorKind::Io, message, Some(Box::new(source)))
    }
}

impl From<TensorReadError> for InferenceError {
    fn from(source: TensorReadError) -> Self {
        let message = format!("tensor readback error: {source}");
        Self::new(ErrorKind::TensorRead, message, Some(Box::new(source)))
    }
}
