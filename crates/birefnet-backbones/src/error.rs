//! Canonical error type for [`crate`].

use std::{
    backtrace::{Backtrace, BacktraceStatus},
    error::Error,
    fmt,
};

/// Error returned when constructing or running a backbone fails.
#[derive(Debug)]
pub struct BackboneError {
    inner: Box<ErrorInner>,
}

#[derive(Debug)]
struct ErrorInner {
    kind: ErrorKind,
    backtrace: Backtrace,
}

#[derive(Debug)]
enum ErrorKind {
    InvalidConfig { message: String },
    InvalidInputShape { message: String },
}

impl BackboneError {
    fn from_kind(kind: ErrorKind) -> Self {
        Self {
            inner: Box::new(ErrorInner {
                kind,
                backtrace: Backtrace::capture(),
            }),
        }
    }

    pub(crate) fn invalid_config(message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::InvalidConfig {
            message: message.into(),
        })
    }

    pub(crate) fn invalid_input_shape(message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::InvalidInputShape {
            message: message.into(),
        })
    }

    /// Returns `true` if this error was caused by invalid backbone configuration.
    #[must_use]
    pub fn is_invalid_config(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::InvalidConfig { .. })
    }

    /// Returns `true` if this error was caused by an incompatible input tensor shape.
    #[must_use]
    pub fn is_invalid_input_shape(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::InvalidInputShape { .. })
    }

    /// Returns the captured [`Backtrace`] for this error.
    #[must_use]
    pub fn backtrace(&self) -> &Backtrace {
        &self.inner.backtrace
    }
}

impl fmt::Display for BackboneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner.kind {
            ErrorKind::InvalidConfig { message } => {
                write!(f, "invalid backbone configuration: {message}")
            }
            ErrorKind::InvalidInputShape { message } => {
                write!(f, "invalid backbone input shape: {message}")
            }
        }?;
        if self.inner.backtrace.status() == BacktraceStatus::Captured {
            write!(f, "\n\nstack backtrace:\n{}", self.inner.backtrace)?;
        }
        Ok(())
    }
}

impl Error for BackboneError {}

const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<BackboneError>();
};
