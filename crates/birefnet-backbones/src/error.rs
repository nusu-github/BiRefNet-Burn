//! Canonical error type for [`crate`].

use std::backtrace::Backtrace;
use std::fmt;

/// Error returned when constructing or running a backbone fails.
#[derive(Debug)]
pub struct BackboneError {
    kind: ErrorKind,
    backtrace: Backtrace,
}

#[derive(Debug)]
enum ErrorKind {
    InvalidConfig { message: String },
    InvalidInputShape { message: String },
    UnsupportedVariant { message: String },
}

impl BackboneError {
    pub(crate) fn invalid_config(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::InvalidConfig {
                message: message.into(),
            },
            backtrace: Backtrace::capture(),
        }
    }

    pub(crate) fn invalid_input_shape(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::InvalidInputShape {
                message: message.into(),
            },
            backtrace: Backtrace::capture(),
        }
    }

    #[expect(dead_code, reason = "reserved for future variant validation paths")]
    pub(crate) fn unsupported_variant(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::UnsupportedVariant {
                message: message.into(),
            },
            backtrace: Backtrace::capture(),
        }
    }

    /// Returns `true` if this error was caused by invalid backbone configuration.
    #[must_use]
    pub const fn is_invalid_config(&self) -> bool {
        matches!(self.kind, ErrorKind::InvalidConfig { .. })
    }

    /// Returns `true` if this error was caused by an incompatible input tensor shape.
    #[must_use]
    pub const fn is_invalid_input_shape(&self) -> bool {
        matches!(self.kind, ErrorKind::InvalidInputShape { .. })
    }

    /// Returns `true` if this error was caused by an unsupported backbone variant.
    #[must_use]
    pub const fn is_unsupported_variant(&self) -> bool {
        matches!(self.kind, ErrorKind::UnsupportedVariant { .. })
    }

    /// Returns the captured [`Backtrace`] for this error.
    #[must_use]
    pub const fn backtrace(&self) -> &Backtrace {
        &self.backtrace
    }
}

impl fmt::Display for BackboneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ErrorKind::InvalidConfig { message } => {
                write!(f, "invalid backbone configuration: {message}")
            }
            ErrorKind::InvalidInputShape { message } => {
                write!(f, "invalid backbone input shape: {message}")
            }
            ErrorKind::UnsupportedVariant { message } => {
                write!(f, "unsupported backbone variant: {message}")
            }
        }
    }
}

impl std::error::Error for BackboneError {}

const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<BackboneError>();
};
