//! Dataset discovery, training augmentations, and batching for `BiRefNet` on Burn.
//!
//! - [`BiRefNetDataset`] pairs `<root>/<task>/<split>/im/*` with same-stem masks in `gt/`,
//!   decodes them by content, and optionally augments each sample with an [`ImageAugmentor`].
//!   Items keep their source resolution.
//! - [`BiRefNetBatcher`] resizes every item of a batch to one [`Resolution`] — fixed
//!   `(width, height)`, or a fresh 32-aligned [`DynamicSize`] per batch as in upstream
//!   `dynamic_size` training — then uploads images and masks once and applies `ImageNet`
//!   normalization on the device.
//! - [`AugmentationConfig`] selects the upstream `preproc_methods` (flip, enhance, rotate,
//!   pepper, crop) and optional background color synthesis.
//!
//! ```text
//! <root>/
//! └── DIS5K/
//!     ├── DIS-TR/{im,gt}/   # training images and masks
//!     └── DIS-VD/{im,gt}/   # validation images and masks
//! ```
//!
//! # Examples
//!
//! ```rust,no_run
//! use birefnet_model::Task;
//! use birefnet_train::{
//!     AugmentationConfig, BiRefNetBatcher, BiRefNetDataset, DatasetSplit, DynamicSize,
//!     ImageAugmentor, Resolution,
//! };
//! use burn::{data::dataloader::DataLoaderBuilder, tensor::Device};
//!
//! let train = BiRefNetDataset::new("/data", Task::DIS5K, &DatasetSplit::Train)?
//!     .with_augmentor(ImageAugmentor::new(AugmentationConfig::new()));
//! let valid = BiRefNetDataset::new("/data", Task::DIS5K, &DatasetSplit::Val)?;
//!
//! let device = Device::flex().autodiff();
//! let train_loader = DataLoaderBuilder::new(BiRefNetBatcher::new(Resolution::Dynamic(
//!     DynamicSize::square(256..=2304)?,
//! )))
//! .batch_size(4)
//! .shuffle(42)
//! .set_device(device.clone())
//! .build(train);
//! let valid_loader = DataLoaderBuilder::new(BiRefNetBatcher::new(Resolution::fixed(1024, 1024)?))
//!     .batch_size(4)
//!     .set_device(device)
//!     .build(valid);
//! # let _ = (train_loader, valid_loader);
//! # Ok::<(), birefnet_train::TrainDataError>(())
//! ```
//!
//! Batches can also be built directly:
//!
//! ```rust
//! use birefnet_train::{BiRefNetBatcher, BiRefNetItem, Resolution};
//! use burn::{data::dataloader::batcher::Batcher, tensor::Device};
//! use image::{GrayImage, RgbImage};
//!
//! let items = vec![
//!     BiRefNetItem { image: RgbImage::new(40, 30), mask: GrayImage::new(40, 30) },
//!     BiRefNetItem { image: RgbImage::new(64, 64), mask: GrayImage::new(64, 64) },
//! ];
//! let batch = BiRefNetBatcher::new(Resolution::fixed(32, 32)?).batch(items, &Device::flex());
//! assert_eq!(batch.images.dims(), [2, 3, 32, 32]);
//! # Ok::<(), birefnet_train::TrainDataError>(())
//! ```

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
