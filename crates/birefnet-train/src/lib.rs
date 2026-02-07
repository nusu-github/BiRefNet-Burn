pub mod augmentation;
pub mod dataset;
pub mod error;

// Re-export commonly used types
#[doc(inline)]
pub use augmentation::{AugmentationConfig, AugmentationMethod, ImageAugmentor};
#[doc(inline)]
pub use dataset::{BiRefNetBatcher, BiRefNetDataset, BiRefNetItem};
#[doc(inline)]
pub use error::{DatasetError, DatasetResult};

#[cfg(test)]
mod tests {
    use burn::backend::Cpu;

    pub type TestBackend = Cpu;
}