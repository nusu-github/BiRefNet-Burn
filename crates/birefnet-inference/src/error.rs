//! Canonical error type for [`crate`].

use std::{
    backtrace::{Backtrace, BacktraceStatus},
    error::Error,
    fmt,
};

use birefnet_model::ModelError;
use birefnet_util::UtilError;
use burn::tensor::TensorReadError;

/// Error returned by inference and mask postprocessing operations.
#[derive(Debug)]
pub struct InferenceError {
    inner: Box<ErrorInner>,
}

#[derive(Debug)]
struct ErrorInner {
    kind: ErrorKind,
    backtrace: Backtrace,
}

#[derive(Debug)]
enum ErrorKind {
    Model(ModelError),
    Util(UtilError),
    Image(image::ImageError),
    Io(std::io::Error),
    TensorRead(TensorReadError),
    InvalidInput { message: String },
    InvalidConfig { message: String },
}

impl InferenceError {
    fn from_kind(kind: ErrorKind) -> Self {
        Self {
            inner: Box::new(ErrorInner {
                kind,
                backtrace: Backtrace::capture(),
            }),
        }
    }

    pub(crate) fn invalid_input(message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::InvalidInput {
            message: message.into(),
        })
    }

    pub(crate) fn invalid_config(message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::InvalidConfig {
            message: message.into(),
        })
    }

    /// Returns `true` if the `BiRefNet` forward pass failed.
    #[must_use]
    pub fn is_model(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::Model(_))
    }

    /// Returns `true` if an image utility (decoding, normalization, masking) failed.
    #[must_use]
    pub fn is_util(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::Util(_))
    }

    /// Returns `true` if encoding or decoding an image failed.
    #[must_use]
    pub fn is_image(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::Image(_))
    }

    /// Returns `true` if a filesystem or reader operation failed.
    #[must_use]
    pub fn is_io(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::Io(_))
    }

    /// Returns `true` if reading a tensor back from the device failed.
    #[must_use]
    pub fn is_tensor_read(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::TensorRead(_))
    }

    /// Returns `true` if the input tensor or image had an unsupported shape.
    #[must_use]
    pub fn is_invalid_input(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::InvalidInput { .. })
    }

    /// Returns `true` if a segmenter or postprocessing parameter was out of range.
    #[must_use]
    pub fn is_invalid_config(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::InvalidConfig { .. })
    }

    /// Returns the captured [`Backtrace`] for this error.
    #[must_use]
    pub fn backtrace(&self) -> &Backtrace {
        &self.inner.backtrace
    }
}

impl fmt::Display for InferenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner.kind {
            ErrorKind::Model(_) => f.write_str("model forward pass failed"),
            ErrorKind::Util(_) => f.write_str("image utility failed"),
            ErrorKind::Image(_) => f.write_str("image codec failed"),
            ErrorKind::Io(_) => f.write_str("inference I/O failed"),
            ErrorKind::TensorRead(_) => f.write_str("failed to read tensor data from the device"),
            ErrorKind::InvalidInput { message } => write!(f, "invalid input: {message}"),
            ErrorKind::InvalidConfig { message } => write!(f, "invalid configuration: {message}"),
        }?;
        if self.inner.backtrace.status() == BacktraceStatus::Captured {
            write!(f, "\n\nstack backtrace:\n{}", self.inner.backtrace)?;
        }
        Ok(())
    }
}

impl Error for InferenceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.inner.kind {
            ErrorKind::Model(source) => Some(source),
            ErrorKind::Util(source) => Some(source),
            ErrorKind::Image(source) => Some(source),
            ErrorKind::Io(source) => Some(source),
            ErrorKind::TensorRead(source) => Some(source),
            ErrorKind::InvalidInput { .. } | ErrorKind::InvalidConfig { .. } => None,
        }
    }
}

macro_rules! from_source {
    ($($source:ty => $kind:ident),+ $(,)?) => {
        $(impl From<$source> for InferenceError {
            fn from(source: $source) -> Self {
                Self::from_kind(ErrorKind::$kind(source))
            }
        })+
    };
}

from_source! {
    ModelError => Model,
    UtilError => Util,
    image::ImageError => Image,
    std::io::Error => Io,
    TensorReadError => TensorRead,
}

const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<InferenceError>();
};
