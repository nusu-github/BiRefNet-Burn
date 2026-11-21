//! Error types for training dataset discovery, image decoding, and preprocessing.

use std::{backtrace::Backtrace, fmt, path::Path};

type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ErrorKind {
    NotFound,
    EmptyDataset,
    Io,
    Image,
    InvalidPath,
}

#[derive(Debug)]
struct ErrorInner {
    kind: ErrorKind,
    message: String,
    source: Option<BoxError>,
    backtrace: Backtrace,
}

/// Error returned by dataset construction and item loading operations.
pub struct TrainDataError {
    inner: Box<ErrorInner>,
}

const _: () = {
    const fn assert_error_bounds<E: std::error::Error + Send + Sync + 'static>() {}
    assert_error_bounds::<TrainDataError>();
};

impl TrainDataError {
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

    pub(crate) fn not_found(path: &Path) -> Self {
        Self::new(
            ErrorKind::NotFound,
            format!("dataset directory not found: `{}`", path.display()),
            None,
        )
    }

    pub(crate) fn empty_dataset(path: &Path) -> Self {
        Self::new(
            ErrorKind::EmptyDataset,
            format!("no valid image/mask pairs found in `{}`", path.display()),
            None,
        )
    }

    pub(crate) fn invalid_path(path: &Path, reason: &str) -> Self {
        Self::new(
            ErrorKind::InvalidPath,
            format!("invalid dataset path `{}`: {reason}", path.display()),
            None,
        )
    }

    pub(crate) fn io(path: &Path, source: std::io::Error) -> Self {
        let kind = if source.kind() == std::io::ErrorKind::NotFound {
            ErrorKind::NotFound
        } else {
            ErrorKind::Io
        };
        let message = format!("failed to read `{}`: {source}", path.display());
        Self::new(kind, message, Some(Box::new(source)))
    }

    pub(crate) fn image(path: &Path, source: image::ImageError) -> Self {
        let message = format!("failed to decode image `{}`: {source}", path.display());
        Self::new(ErrorKind::Image, message, Some(Box::new(source)))
    }

    /// Returns `true` if a required dataset directory or file does not exist.
    #[must_use]
    pub fn is_not_found(&self) -> bool {
        self.inner.kind == ErrorKind::NotFound
    }

    /// Returns `true` if no matching image/mask pairs were discovered.
    #[must_use]
    pub fn is_empty_dataset(&self) -> bool {
        self.inner.kind == ErrorKind::EmptyDataset
    }

    /// Returns `true` if the failure originated from filesystem I/O.
    #[must_use]
    pub fn is_io(&self) -> bool {
        self.inner.kind == ErrorKind::Io
    }

    /// Returns `true` if decoding an image or mask failed.
    #[must_use]
    pub fn is_image(&self) -> bool {
        self.inner.kind == ErrorKind::Image
    }

    /// Returns `true` if a filesystem path lacked a valid UTF-8 filename or stem.
    #[must_use]
    pub fn is_invalid_path(&self) -> bool {
        self.inner.kind == ErrorKind::InvalidPath
    }

    /// Returns the backtrace captured when this error was constructed.
    #[must_use]
    pub fn backtrace(&self) -> &Backtrace {
        &self.inner.backtrace
    }
}

impl fmt::Debug for TrainDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TrainDataError")
            .field("kind", &self.inner.kind)
            .field("message", &self.inner.message)
            .field("source", &self.inner.source)
            .finish_non_exhaustive()
    }
}

impl fmt::Display for TrainDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.inner.message)
    }
}

impl std::error::Error for TrainDataError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self.inner.source.as_deref() {
            Some(err) => Some(err),
            None => None,
        }
    }
}
