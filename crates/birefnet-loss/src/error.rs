//! Canonical error type for [`crate`].

use std::backtrace::Backtrace;
use std::fmt;

/// Error returned when loss validation or computation fails.
#[derive(Debug)]
pub struct LossError {
    kind: ErrorKind,
    backtrace: Backtrace,
}

#[derive(Debug)]
enum ErrorKind {
    EmptyPredictions,
    InvalidDimensions { message: String },
}

impl LossError {
    pub(crate) fn empty_predictions() -> Self {
        Self {
            kind: ErrorKind::EmptyPredictions,
            backtrace: Backtrace::capture(),
        }
    }

    pub(crate) fn invalid_dimensions(message: impl Into<String>) -> Self {
        Self {
            kind: ErrorKind::InvalidDimensions {
                message: message.into(),
            },
            backtrace: Backtrace::capture(),
        }
    }

    /// Returns `true` if the loss received an empty slice of predictions.
    #[must_use]
    pub const fn is_empty_predictions(&self) -> bool {
        matches!(self.kind, ErrorKind::EmptyPredictions)
    }

    /// Returns `true` if input tensor dimensions or shapes were incompatible.
    #[must_use]
    pub const fn is_invalid_dimensions(&self) -> bool {
        matches!(self.kind, ErrorKind::InvalidDimensions { .. })
    }

    /// Returns the captured [`Backtrace`] for this error.
    #[must_use]
    pub const fn backtrace(&self) -> &Backtrace {
        &self.backtrace
    }
}

impl fmt::Display for LossError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ErrorKind::EmptyPredictions => f.write_str(
                "predictions cannot be empty: at least one prediction tensor is required",
            ),
            ErrorKind::InvalidDimensions { message } => {
                write!(f, "invalid tensor dimensions: {message}")
            }
        }
    }
}

impl std::error::Error for LossError {}

const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<LossError>();
};
