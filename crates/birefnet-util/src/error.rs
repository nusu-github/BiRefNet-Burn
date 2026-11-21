//! Canonical error type for [`crate`].

use std::backtrace::Backtrace;
use std::error::Error;
use std::fmt;
use std::path::PathBuf;

/// Error returned by image I/O, normalization, and utility operations.
#[derive(Debug)]
pub struct UtilError {
    inner: Box<ErrorInner>,
}

#[derive(Debug)]
struct ErrorInner {
    kind: ErrorKind,
    backtrace: Backtrace,
}

#[derive(Debug)]
enum ErrorKind {
    Io {
        path: Option<PathBuf>,
        source: std::io::Error,
    },
    #[cfg(feature = "image-io")]
    Image {
        path: Option<PathBuf>,
        source: image::ImageError,
    },
    InvalidDimensions {
        message: String,
    },
    InvalidInput {
        message: String,
    },
}

impl UtilError {
    fn from_kind(kind: ErrorKind) -> Self {
        Self {
            inner: Box::new(ErrorInner {
                kind,
                backtrace: Backtrace::capture(),
            }),
        }
    }

    pub(crate) fn io(path: Option<PathBuf>, source: std::io::Error) -> Self {
        Self::from_kind(ErrorKind::Io { path, source })
    }

    #[cfg(feature = "image-io")]
    pub(crate) fn image(path: Option<PathBuf>, source: image::ImageError) -> Self {
        Self::from_kind(ErrorKind::Image { path, source })
    }

    pub(crate) fn invalid_dimensions(message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::InvalidDimensions {
            message: message.into(),
        })
    }

    #[cfg_attr(
        not(feature = "image-io"),
        expect(dead_code, reason = "constructed by image-io decoding paths")
    )]
    pub(crate) fn invalid_input(message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::InvalidInput {
            message: message.into(),
        })
    }

    /// Returns `true` if this error originated from a filesystem or reader I/O failure.
    #[must_use]
    pub fn is_io(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::Io { .. })
    }

    /// Returns `true` if this error originated from decoding or encoding an image.
    #[must_use]
    pub fn is_image(&self) -> bool {
        #[cfg(feature = "image-io")]
        {
            matches!(self.inner.kind, ErrorKind::Image { .. })
        }
        #[cfg(not(feature = "image-io"))]
        {
            false
        }
    }

    /// Returns `true` if this error was caused by invalid tensor or buffer dimensions.
    #[must_use]
    pub fn is_invalid_dimensions(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::InvalidDimensions { .. })
    }

    /// Returns `true` if this error was caused by invalid caller arguments or unsupported input data.
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

impl From<std::io::Error> for UtilError {
    fn from(source: std::io::Error) -> Self {
        Self::io(None, source)
    }
}

#[cfg(feature = "image-io")]
impl From<image::ImageError> for UtilError {
    fn from(source: image::ImageError) -> Self {
        Self::image(None, source)
    }
}

impl fmt::Display for UtilError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner.kind {
            ErrorKind::Io {
                path: Some(path),
                source,
            } => write!(f, "I/O error at '{}': {source}", path.display()),
            ErrorKind::Io { path: None, source } => write!(f, "I/O error: {source}"),
            #[cfg(feature = "image-io")]
            ErrorKind::Image {
                path: Some(path),
                source,
            } => write!(f, "image error at '{}': {source}", path.display()),
            #[cfg(feature = "image-io")]
            ErrorKind::Image { path: None, source } => write!(f, "image error: {source}"),
            ErrorKind::InvalidDimensions { message } => {
                write!(f, "invalid dimensions: {message}")
            }
            ErrorKind::InvalidInput { message } => write!(f, "invalid input: {message}"),
        }
    }
}

impl Error for UtilError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.inner.kind {
            ErrorKind::Io { source, .. } => Some(source),
            #[cfg(feature = "image-io")]
            ErrorKind::Image { source, .. } => Some(source),
            ErrorKind::InvalidDimensions { .. } | ErrorKind::InvalidInput { .. } => None,
        }
    }
}

const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<UtilError>();
};
