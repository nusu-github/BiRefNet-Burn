//! Canonical error type for [`crate`].

use std::{
    backtrace::{Backtrace, BacktraceStatus},
    error::Error,
    fmt,
};

use burn::tensor::TensorReadError;

/// Error returned when loss validation, computation, or host readback fails.
#[derive(Debug)]
pub struct LossError {
    inner: Box<ErrorInner>,
}

#[derive(Debug)]
struct ErrorInner {
    kind: ErrorKind,
    backtrace: Backtrace,
}

#[derive(Debug)]
enum ErrorKind {
    EmptyPredictions,
    InvalidDimensions { message: String },
    TensorRead(TensorReadError),
}

impl LossError {
    fn from_kind(kind: ErrorKind) -> Self {
        Self {
            inner: Box::new(ErrorInner {
                kind,
                backtrace: Backtrace::capture(),
            }),
        }
    }

    pub(crate) fn empty_predictions() -> Self {
        Self::from_kind(ErrorKind::EmptyPredictions)
    }

    pub(crate) fn invalid_dimensions(message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::InvalidDimensions {
            message: message.into(),
        })
    }

    pub(crate) fn tensor_read(source: TensorReadError) -> Self {
        Self::from_kind(ErrorKind::TensorRead(source))
    }

    /// Returns `true` if the loss received an empty slice of predictions.
    #[must_use]
    pub fn is_empty_predictions(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::EmptyPredictions)
    }

    /// Returns `true` if input tensor dimensions or shapes were incompatible.
    #[must_use]
    pub fn is_invalid_dimensions(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::InvalidDimensions { .. })
    }

    /// Returns `true` if reading loss values back from the device failed.
    #[must_use]
    pub fn is_tensor_read(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::TensorRead(_))
    }

    /// Returns the captured [`Backtrace`] for this error.
    #[must_use]
    pub fn backtrace(&self) -> &Backtrace {
        &self.inner.backtrace
    }
}

impl fmt::Display for LossError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner.kind {
            ErrorKind::EmptyPredictions => f.write_str(
                "predictions cannot be empty: at least one prediction tensor is required",
            ),
            ErrorKind::InvalidDimensions { message } => {
                write!(f, "invalid tensor dimensions: {message}")
            }
            ErrorKind::TensorRead(_) => f.write_str("failed to read loss values from the device"),
        }?;
        if self.inner.backtrace.status() == BacktraceStatus::Captured {
            write!(f, "\n\nstack backtrace:\n{}", self.inner.backtrace)?;
        }
        Ok(())
    }
}

impl Error for LossError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.inner.kind {
            ErrorKind::TensorRead(source) => Some(source),
            ErrorKind::EmptyPredictions | ErrorKind::InvalidDimensions { .. } => None,
        }
    }
}

const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<LossError>();
};
