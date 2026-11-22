//! Canonical error type for [`crate`].

use std::{
    backtrace::{Backtrace, BacktraceStatus},
    error::Error,
    fmt,
};

use birefnet_backbones::BackboneError;
#[cfg(feature = "train")]
use birefnet_loss::LossError;

/// Error returned by model configuration, construction, and forward passes.
#[derive(Debug)]
pub struct ModelError {
    inner: Box<ErrorInner>,
}

#[derive(Debug)]
struct ErrorInner {
    kind: ErrorKind,
    backtrace: Backtrace,
}

#[derive(Debug)]
enum ErrorKind {
    Backbone(BackboneError),
    #[cfg(feature = "train")]
    Loss(LossError),
    InvalidConfig {
        message: String,
    },
    InvalidInput {
        message: String,
    },
}

impl ModelError {
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

    pub(crate) fn invalid_input(message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::InvalidInput {
            message: message.into(),
        })
    }

    /// Returns `true` if this error originated from backbone construction.
    #[must_use]
    pub fn is_backbone(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::Backbone(_))
    }

    /// Returns `true` if this error originated from loss computation.
    #[must_use]
    pub fn is_loss(&self) -> bool {
        cfg_select! {
            feature = "train" => matches!(self.inner.kind, ErrorKind::Loss(_)),
            _ => false,
        }
    }

    /// Returns `true` if this error was caused by an invalid model configuration.
    #[must_use]
    pub fn is_invalid_config(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::InvalidConfig { .. })
    }

    /// Returns `true` if this error was caused by invalid model inputs.
    #[must_use]
    pub fn is_invalid_input(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::InvalidInput { .. })
    }

    /// Returns the captured [`Backtrace`] for this error.
    #[must_use]
    pub fn backtrace(&self) -> &Backtrace {
        &self.inner.backtrace
    }
}

impl From<BackboneError> for ModelError {
    fn from(source: BackboneError) -> Self {
        Self::from_kind(ErrorKind::Backbone(source))
    }
}

#[cfg(feature = "train")]
impl From<LossError> for ModelError {
    fn from(source: LossError) -> Self {
        Self::from_kind(ErrorKind::Loss(source))
    }
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner.kind {
            ErrorKind::Backbone(_) => f.write_str("backbone construction or forward pass failed"),
            #[cfg(feature = "train")]
            ErrorKind::Loss(_) => f.write_str("loss computation failed"),
            ErrorKind::InvalidConfig { message } => write!(f, "invalid configuration: {message}"),
            ErrorKind::InvalidInput { message } => write!(f, "invalid input: {message}"),
        }?;
        if self.inner.backtrace.status() == BacktraceStatus::Captured {
            write!(f, "\n\nstack backtrace:\n{}", self.inner.backtrace)?;
        }
        Ok(())
    }
}

impl Error for ModelError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.inner.kind {
            ErrorKind::Backbone(err) => Some(err),
            #[cfg(feature = "train")]
            ErrorKind::Loss(err) => Some(err),
            ErrorKind::InvalidConfig { .. } | ErrorKind::InvalidInput { .. } => None,
        }
    }
}

const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<ModelError>();
};
