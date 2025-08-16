pub mod dataset;
pub mod error;

// Re-export commonly used types
pub use dataset::{BiRefNetBatch, BiRefNetBatcher, BiRefNetDataset, BiRefNetItem};
pub use error::{DatasetError, DatasetResult};

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    pub type TestBackend = NdArray;
}
