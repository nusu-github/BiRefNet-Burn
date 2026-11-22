#![doc = include_str!("../README.md")]

mod augmentation;
mod dataset;
mod error;
mod transforms;

#[doc(inline)]
pub use augmentation::{AugmentationConfig, AugmentationMethod, ImageAugmentor};
#[doc(inline)]
pub use dataset::{
    BiRefNetBatcher, BiRefNetDataset, BiRefNetItem, DatasetSplit, DynamicSize, Resolution,
};
#[doc(inline)]
pub use error::TrainDataError;
