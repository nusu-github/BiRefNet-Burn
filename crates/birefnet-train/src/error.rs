//! Canonical error type for [`crate`].

use std::{
    backtrace::{Backtrace, BacktraceStatus},
    error::Error,
    fmt,
    path::{Path, PathBuf},
};

/// Error returned by dataset discovery, sample loading, and data pipeline configuration.
#[derive(Debug)]
pub struct TrainDataError {
    inner: Box<ErrorInner>,
}

#[derive(Debug)]
struct ErrorInner {
    kind: ErrorKind,
    path: Option<PathBuf>,
    backtrace: Backtrace,
}

#[derive(Debug)]
enum ErrorKind {
    NotFound,
    EmptyDataset,
    InvalidPath { reason: &'static str },
    Io(std::io::Error),
    Image(image::ImageError),
    DimensionMismatch { image: (u32, u32), mask: (u32, u32) },
    IndexOutOfBounds { index: usize, len: usize },
    InvalidConfig { message: String },
}

impl TrainDataError {
    fn new(kind: ErrorKind, path: Option<&Path>) -> Self {
        Self {
            inner: Box::new(ErrorInner {
                kind,
                path: path.map(Path::to_path_buf),
                backtrace: Backtrace::capture(),
            }),
        }
    }

    pub(crate) fn not_found(path: &Path) -> Self {
        Self::new(ErrorKind::NotFound, Some(path))
    }

    pub(crate) fn empty_dataset(path: &Path) -> Self {
        Self::new(ErrorKind::EmptyDataset, Some(path))
    }

    pub(crate) fn invalid_path(path: &Path, reason: &'static str) -> Self {
        Self::new(ErrorKind::InvalidPath { reason }, Some(path))
    }

    pub(crate) fn io(path: &Path, source: std::io::Error) -> Self {
        Self::new(ErrorKind::Io(source), Some(path))
    }

    pub(crate) fn image(path: &Path, source: image::ImageError) -> Self {
        Self::new(ErrorKind::Image(source), Some(path))
    }

    pub(crate) fn dimension_mismatch(path: &Path, image: (u32, u32), mask: (u32, u32)) -> Self {
        Self::new(ErrorKind::DimensionMismatch { image, mask }, Some(path))
    }

    pub(crate) fn index_out_of_bounds(index: usize, len: usize) -> Self {
        Self::new(ErrorKind::IndexOutOfBounds { index, len }, None)
    }

    pub(crate) fn invalid_config(message: impl Into<String>) -> Self {
        Self::new(
            ErrorKind::InvalidConfig {
                message: message.into(),
            },
            None,
        )
    }

    /// Returns `true` if a required directory or file does not exist.
    #[must_use]
    pub fn is_not_found(&self) -> bool {
        match &self.inner.kind {
            ErrorKind::NotFound => true,
            ErrorKind::Io(source) => source.kind() == std::io::ErrorKind::NotFound,
            _ => false,
        }
    }

    /// Returns `true` if no image/mask pairs were discovered.
    #[must_use]
    pub fn is_empty_dataset(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::EmptyDataset)
    }

    /// Returns `true` if a filesystem operation failed (including a missing file).
    #[must_use]
    pub fn is_io(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::Io(_))
    }

    /// Returns `true` if decoding an image or mask failed.
    #[must_use]
    pub fn is_image(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::Image(_))
    }

    /// Returns `true` if a file name could not be interpreted.
    #[must_use]
    pub fn is_invalid_path(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::InvalidPath { .. })
    }

    /// Returns `true` if an image and its mask have different sizes.
    #[must_use]
    pub fn is_dimension_mismatch(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::DimensionMismatch { .. })
    }

    /// Returns `true` if a sample index was past the end of the dataset.
    #[must_use]
    pub fn is_index_out_of_bounds(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::IndexOutOfBounds { .. })
    }

    /// Returns `true` if a data pipeline parameter was invalid.
    #[must_use]
    pub fn is_invalid_config(&self) -> bool {
        matches!(self.inner.kind, ErrorKind::InvalidConfig { .. })
    }

    /// Returns the file or directory involved, if any.
    #[must_use]
    pub fn path(&self) -> Option<&Path> {
        self.inner.path.as_deref()
    }

    /// Returns the captured [`Backtrace`] for this error.
    #[must_use]
    pub fn backtrace(&self) -> &Backtrace {
        &self.inner.backtrace
    }
}

impl From<std::io::Error> for TrainDataError {
    fn from(source: std::io::Error) -> Self {
        Self::new(ErrorKind::Io(source), None)
    }
}

impl From<image::ImageError> for TrainDataError {
    fn from(source: image::ImageError) -> Self {
        Self::new(ErrorKind::Image(source), None)
    }
}

impl fmt::Display for TrainDataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner.kind {
            ErrorKind::NotFound => f.write_str("not found"),
            ErrorKind::EmptyDataset => f.write_str("no image/mask pairs found"),
            ErrorKind::InvalidPath { reason } => write!(f, "invalid path ({reason})"),
            ErrorKind::Io(_) => f.write_str("I/O error"),
            ErrorKind::Image(_) => f.write_str("image decoding failed"),
            ErrorKind::DimensionMismatch { image, mask } => write!(
                f,
                "image is {}x{} but its mask is {}x{}",
                image.0, image.1, mask.0, mask.1
            ),
            ErrorKind::IndexOutOfBounds { index, len } => {
                write!(f, "sample index {index} is out of bounds for {len} samples")
            }
            ErrorKind::InvalidConfig { message } => write!(f, "invalid configuration: {message}"),
        }?;
        if let Some(path) = &self.inner.path {
            write!(f, " at '{}'", path.display())?;
        }
        if self.inner.backtrace.status() == BacktraceStatus::Captured {
            write!(f, "\n\nstack backtrace:\n{}", self.inner.backtrace)?;
        }
        Ok(())
    }
}

impl Error for TrainDataError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.inner.kind {
            ErrorKind::Io(source) => Some(source),
            ErrorKind::Image(source) => Some(source),
            _ => None,
        }
    }
}

const _: fn() = || {
    fn assert_send_sync_static<T: Send + Sync + 'static>() {}
    assert_send_sync_static::<TrainDataError>();
};
