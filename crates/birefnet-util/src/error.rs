//! Canonical error type for [`crate`].

use std::{
    backtrace::{Backtrace, BacktraceStatus},
    error::Error,
    fmt,
    path::{Path, PathBuf},
};

use burn::tensor::TensorReadError;

/// Error returned by image I/O, tensor readback, normalization, and utility operations.
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
    TensorRead(TensorReadError),
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

    #[cfg_attr(
        not(feature = "image-io"),
        expect(dead_code, reason = "constructed by image-io file paths")
    )]
    pub(crate) fn io(path: Option<PathBuf>, source: std::io::Error) -> Self {
        Self::from_kind(ErrorKind::Io { path, source })
    }

    #[cfg(feature = "image-io")]
    pub(crate) fn image(path: Option<PathBuf>, source: image::ImageError) -> Self {
        Self::from_kind(ErrorKind::Image { path, source })
    }

    pub(crate) fn tensor_read(source: TensorReadError) -> Self {
        Self::from_kind(ErrorKind::TensorRead(source))
    }

    pub(crate) fn invalid_dimensions(message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::InvalidDimensions {
            message: message.into(),
        })
    }

    pub(crate) fn invalid_input(message: impl Into<String>) -> Self {
        Self::from_kind(ErrorKind::InvalidInput {
            message: message.into(),
        })
    }

    /// Returns `true` if this error originated from a filesystem or reader I/O failure,
    /// including a missing file.
    #[must_use]
    pub fn is_io(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::Io { .. })
    }

    /// Returns `true` if the file or stream does not exist.
    #[must_use]
    pub fn is_not_found(&self) -> bool {
        matches!(
            &self.inner.kind,
            ErrorKind::Io { source, .. } if source.kind() == std::io::ErrorKind::NotFound
        )
    }

    /// Returns `true` if this error originated from decoding or encoding an image.
    #[must_use]
    pub fn is_image(&self) -> bool {
        cfg_select! {
            feature = "image-io" => matches!(self.inner.kind, ErrorKind::Image { .. }),
            _ => false,
        }
    }

    /// Returns `true` if reading tensor data back from the device failed.
    #[must_use]
    pub fn is_tensor_read(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::TensorRead(_))
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

    /// Returns the file path involved in an I/O or image error, if known.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        match &self.inner.kind {
            ErrorKind::Io { path, .. } => path.as_deref(),
            #[cfg(feature = "image-io")]
            ErrorKind::Image { path, .. } => path.as_deref(),
            _ => None,
        }
    }

    /// Returns the captured [`Backtrace`] for this error.
    #[must_use]
    pub fn backtrace(&self) -> &Backtrace {
        &self.inner.backtrace
    }
}

impl From<std::io::Error> for UtilError {
    fn from(source: std::io::Error) -> Self {
        Self::from_kind(ErrorKind::Io { path: None, source })
    }
}

impl From<TensorReadError> for UtilError {
    fn from(source: TensorReadError) -> Self {
        Self::tensor_read(source)
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
        let at = |path: &Option<PathBuf>| {
            path.as_ref()
                .map_or_else(String::new, |path| format!(" at '{}'", path.display()))
        };
        match &self.inner.kind {
            ErrorKind::Io { path, .. } => write!(f, "I/O error{}", at(path)),
            #[cfg(feature = "image-io")]
            ErrorKind::Image { path, .. } => write!(f, "image codec error{}", at(path)),
            ErrorKind::TensorRead(_) => f.write_str("failed to read tensor data from the device"),
            ErrorKind::InvalidDimensions { message } => {
                write!(f, "invalid dimensions: {message}")
            }
            ErrorKind::InvalidInput { message } => write!(f, "invalid input: {message}"),
        }?;
        if self.inner.backtrace.status() == BacktraceStatus::Captured {
            write!(f, "\n\nstack backtrace:\n{}", self.inner.backtrace)?;
        }
        Ok(())
    }
}

impl Error for UtilError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.inner.kind {
            ErrorKind::Io { source, .. } => Some(source),
            #[cfg(feature = "image-io")]
            ErrorKind::Image { source, .. } => Some(source),
            ErrorKind::TensorRead(source) => Some(source),
            ErrorKind::InvalidDimensions { .. } | ErrorKind::InvalidInput { .. } => None,
        }
    }
}

const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<UtilError>();
};
