//! Canonical error type for [`crate`].

use std::{
    backtrace::{Backtrace, BacktraceStatus},
    error::Error,
    fmt,
};

use burn::tensor::TensorReadError;

/// Error returned when metric inputs are invalid or cannot be read back from the device.
#[derive(Debug)]
pub struct MetricError {
    inner: Box<ErrorInner>,
}

#[derive(Debug)]
struct ErrorInner {
    kind: ErrorKind,
    backtrace: Backtrace,
}

#[derive(Debug)]
enum ErrorKind {
    TensorRead(TensorReadError),
    InvalidInput { message: String },
    InvalidConfig { message: String },
}

impl MetricError {
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

    /// Returns `true` if reading prediction or target tensors from the device failed.
    #[must_use]
    pub fn is_tensor_read(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::TensorRead(_))
    }

    /// Returns `true` if predictions and targets had mismatched shapes or invalid values.
    #[must_use]
    pub fn is_invalid_input(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::InvalidInput { .. })
    }

    /// Returns `true` if a metric parameter (such as `beta_sq` or `alpha`) was out of range.
    #[must_use]
    pub fn is_invalid_config(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::InvalidConfig { .. })
    }

    /// Returns the captured [`Backtrace`] for this error.
    #[must_use]
    pub fn backtrace(&self) -> &Backtrace {
        &self.inner.backtrace
    }

    /// Splits off device read failures, the only errors Burn's `Metric::update` can report.
    #[cfg(feature = "train")]
    pub(crate) fn into_tensor_read(self) -> Result<TensorReadError, Self> {
        match self.inner.kind {
            ErrorKind::TensorRead(source) => Ok(source),
            kind => Err(Self {
                inner: Box::new(ErrorInner {
                    kind,
                    backtrace: self.inner.backtrace,
                }),
            }),
        }
    }
}

impl From<TensorReadError> for MetricError {
    fn from(source: TensorReadError) -> Self {
        Self::from_kind(ErrorKind::TensorRead(source))
    }
}

impl fmt::Display for MetricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner.kind {
            ErrorKind::TensorRead(_) => f.write_str("failed to read metric inputs from the device"),
            ErrorKind::InvalidInput { message } => write!(f, "invalid metric input: {message}"),
            ErrorKind::InvalidConfig { message } => {
                write!(f, "invalid metric configuration: {message}")
            }
        }?;
        if self.inner.backtrace.status() == BacktraceStatus::Captured {
            write!(f, "\n\nstack backtrace:\n{}", self.inner.backtrace)?;
        }
        Ok(())
    }
}

impl Error for MetricError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.inner.kind {
            ErrorKind::TensorRead(source) => Some(source),
            ErrorKind::InvalidInput { .. } | ErrorKind::InvalidConfig { .. } => None,
        }
    }
}

const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<MetricError>();
};
