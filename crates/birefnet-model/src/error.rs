use thiserror::Error;

/// The error type for `BiRefNet-Burn` operations.
///
/// This enum encapsulates all possible errors that can occur within the BiRefNet-Burn library,
/// ranging from configuration issues to tensor manipulation failures.
#[derive(Error, Debug)]
pub enum BiRefNetError {
    /// Error for when an unsupported backbone network is specified.
    #[error("Unsupported backbone: {backbone}")]
    UnsupportedBackbone {
        /// The name of the unsupported backbone.
        backbone: String,
    },

    /// Error for when an unsupported Squeeze block type is specified.
    #[error("Unsupported squeeze block type: {block_type}")]
    UnsupportedSqueezeBlock {
        /// The name of the unsupported squeeze block.
        block_type: String,
    },

    /// Error for when an invalid model configuration is provided.
    /// This can happen if configuration parameters are logically inconsistent.
    #[error("Invalid model configuration: {reason}")]
    InvalidConfiguration {
        /// The reason why the configuration is invalid.
        reason: String,
    },

    /// Error for when a tensor operation fails.
    #[error("Tensor operation failed: {operation}")]
    TensorOperationFailed {
        /// A description of the failed tensor operation.
        operation: String,
    },

    /// Error for when an input tensor has an invalid shape.
    #[error("Invalid input tensor shape: expected {expected}, got {actual}")]
    InvalidTensorShape {
        /// The expected tensor shape.
        expected: String,
        /// The actual tensor shape.
        actual: String,
    },

    /// Error for when model initialization fails.
    #[error("Model initialization failed: {reason}")]
    ModelInitializationFailed {
        /// The reason for the initialization failure.
        reason: String,
    },

    /// Error for when loading model weights fails.
    #[error("Failed to load weights: {reason}")]
    WeightLoadingFailed {
        /// The reason for the weight loading failure.
        reason: String,
    },

    /// Error for when dataset operations fail.
    #[error("Dataset error: {message}")]
    DatasetError {
        /// The error message.
        message: String,
    },

    /// A general-purpose error for cases not covered by other variants.
    #[error("General error: {message}")]
    General {
        /// The error message.
        message: String,
    },
}

/// A specialized `Result` type for `BiRefNet-Burn` operations.
pub type BiRefNetResult<T> = Result<T, BiRefNetError>;
