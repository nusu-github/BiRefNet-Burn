//! Error types specific to the birefnet-train crate.
//!
//! This module provides dataset-specific error handling for training operations,
//! separate from the general BiRefNet model errors.

use std::path::PathBuf;

use thiserror::Error;

/// Error type for dataset operations in birefnet-train.
///
/// This enum covers all possible errors that can occur during dataset loading,
/// preprocessing, and data access operations.
#[derive(Error, Debug)]
pub enum DatasetError {
    /// Error when a file path cannot be converted to a valid string.
    #[error("Invalid file path: {path} - cannot convert to string")]
    InvalidFilePath {
        /// The problematic file path.
        path: PathBuf,
    },

    /// Error when reading a directory fails.
    #[error("Failed to read directory: {path}")]
    DirectoryReadFailed {
        /// The directory path that failed to read.
        path: PathBuf,
        /// The underlying IO error.
        #[source]
        source: std::io::Error,
    },

    /// Error when the image directory is not found.
    #[error("Image directory not found: {path}")]
    ImageDirectoryNotFound {
        /// The expected image directory path.
        path: PathBuf,
    },

    /// Error when the mask directory is not found.
    #[error("Mask directory not found: {path}")]
    MaskDirectoryNotFound {
        /// The expected mask directory path.
        path: PathBuf,
    },

    /// Error when no valid image/mask pairs are found in the dataset.
    #[error("No valid image/mask pairs found in: {path}")]
    NoValidPairs {
        /// The directory where no pairs were found.
        path: PathBuf,
    },

    /// Error when opening or processing an image file fails.
    #[error("Failed to open image: {path}")]
    ImageOpenFailed {
        /// The image file path that failed to open.
        path: PathBuf,
        /// The underlying image processing error.
        #[source]
        source: image::ImageError,
    },

    /// Error when a file has no filename component.
    #[error("File has no filename: {path}")]
    NoFileName {
        /// The file path without a filename.
        path: PathBuf,
    },

    /// Error when a file has no stem (filename without extension).
    #[error("File has no stem: {path}")]
    NoFileStem {
        /// The file path without a stem.
        path: PathBuf,
    },

    /// Error when path components contain invalid UTF-8.
    #[error("Path contains invalid UTF-8: {path}")]
    InvalidUtf8Path {
        /// The path with invalid UTF-8.
        path: PathBuf,
    },
}

/// A specialized `Result` type for dataset operations.
pub type DatasetResult<T> = Result<T, DatasetError>;
