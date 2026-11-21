//! Dataset loading, augmentations, and tensor batching for `BiRefNet` training.

mod augmentation;
mod dataset;
mod error;
mod transforms;

#[doc(inline)]
pub use augmentation::{
    AugmentationConfig, AugmentationMethod, BackgroundPattern, DynamicSizeConfig, ImageAugmentor,
};
#[doc(inline)]
pub use dataset::{BiRefNetBatcher, BiRefNetDataset, BiRefNetItem, DatasetSplit};
#[doc(inline)]
pub use error::TrainDataError;
