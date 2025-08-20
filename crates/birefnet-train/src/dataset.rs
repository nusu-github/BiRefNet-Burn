//! Dataset implementation for BiRefNet training and inference.
//!
//! This module provides an efficient dataset implementation for BiRefNet,
//! handling data loading, preprocessing, and augmentation directly.
//! This approach avoids the inefficiencies of reading files multiple times and
//! provides a clear framework for adding data augmentation.

use std::{
    fs,
    marker::PhantomData,
    path::{Path, PathBuf},
};

use birefnet_model::{training::BiRefNetBatch, ModelConfig, Task};
use birefnet_util::ImageUtils;
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{backend::Backend, Tensor, TensorData},
};
use image::{DynamicImage, ImageFormat};

use crate::{
    augmentation::{AugmentationConfig, ImageAugmentor},
    error::{DatasetError, DatasetResult},
};

/// Represents a single preprocessed data item from the BiRefNet dataset.
///
/// This struct contains the raw image and mask data after preprocessing,
/// following Burn's convention where datasets return raw data and batchers
/// handle tensor creation and device placement.
#[derive(Debug, Clone)]
pub struct BiRefNetItem {
    /// Preprocessed RGB image data as 3D array [H, W, C] with normalized floats
    pub image: Vec<f32>,
    /// Preprocessed binary mask data as 2D array [H, W] with normalized floats  
    pub mask: Vec<f32>,
    /// Image height in pixels
    pub height: usize,
    /// Image width in pixels  
    pub width: usize,
}

/// Batcher implementation for converting vectors of BiRefNetItem into BiRefNetBatch.
///
/// This batcher handles the conversion from individual data items to batched tensors,
/// following the same pattern as Burn's official examples.
#[derive(Clone, Default)]
pub struct BiRefNetBatcher<B: Backend> {
    _phantom: PhantomData<B>,
}

impl<B: Backend> BiRefNetBatcher<B> {
    /// Create a new BiRefNet batcher.
    pub const fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend> Batcher<B, BiRefNetItem, BiRefNetBatch<B>> for BiRefNetBatcher<B> {
    fn batch(&self, items: Vec<BiRefNetItem>, device: &B::Device) -> BiRefNetBatch<B> {
        let batch_size = items.len();

        // Pre-allocate vectors with known capacity to avoid reallocations
        let mut images = Vec::with_capacity(batch_size);
        let mut masks = Vec::with_capacity(batch_size);

        // Convert raw data to tensors and collect
        for item in items {
            // Convert image data to tensor [C, H, W]
            let image_tensor = Tensor::<B, 3>::from_data(
                TensorData::new(item.image, [item.height, item.width, 3]),
                device,
            )
            .permute([2, 0, 1]); // HWC to CHW

            // Apply ImageNet normalization: add batch dimension, normalize, then remove batch dimension
            let image_tensor_with_batch = image_tensor.unsqueeze::<4>(); // [C, H, W] -> [1, C, H, W]
            let normalized_tensor =
                ImageUtils::apply_imagenet_normalization(image_tensor_with_batch)
                    .expect("Failed to apply ImageNet normalization")
                    .squeeze::<3>(0); // [1, C, H, W] -> [C, H, W]

            // Convert mask data to tensor [1, H, W]
            let mask_tensor = Tensor::<B, 2>::from_data(
                TensorData::new(item.mask, [item.height, item.width]),
                device,
            )
            .unsqueeze::<3>(); // Add channel dimension

            images.push(normalized_tensor);
            masks.push(mask_tensor);
        }

        // Stack tensors along the batch dimension (dim 0) to create [B, C, H, W] tensors
        let images = Tensor::stack(images, 0);
        let masks = Tensor::stack(masks, 0);

        BiRefNetBatch::new(images, masks)
    }
}

/// BiRefNet dataset for training and inference.
///
/// This dataset loads image/mask pairs from the DIS5K dataset structure and applies
/// preprocessing and augmentation, returning raw data following Burn's convention.
pub struct BiRefNetDataset {
    items: Vec<(PathBuf, PathBuf)>,
    is_train: bool,
    target_size: (u32, u32),
    augmentor: ImageAugmentor,
}

impl BiRefNetDataset {
    /// Create a new BiRefNet dataset.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration containing dataset paths and task settings
    /// * `split` - Dataset split ("train", "val", "test")
    ///
    /// # Returns
    ///
    /// A new dataset instance or an error if the dataset cannot be loaded.
    pub fn new(config: &ModelConfig, split: &str) -> DatasetResult<Self> {
        let items = Self::collect_dataset_items(config, split)?;
        let is_train = split == "train";

        // Use fixed target size for image preprocessing
        let target_size = (1024, 1024);

        // Create default reinforcement settings
        let augmentation_config = AugmentationConfig::default();
        let augmentor = ImageAugmentor::new(augmentation_config);

        Ok(Self {
            items,
            is_train,
            target_size,
            augmentor,
        })
    }

    /// Create a new BiRefNet dataset with custom augmentation configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration containing dataset paths and task settings
    /// * `split` - Dataset split ("train", "val", "test")
    /// * `augmentation_config` - Custom augmentation configuration
    ///
    /// # Returns
    ///
    /// A new dataset instance or an error if the dataset cannot be loaded.
    pub fn new_with_augmentation(
        config: &ModelConfig,
        split: &str,
        augmentation_config: AugmentationConfig,
    ) -> DatasetResult<Self> {
        let items = Self::collect_dataset_items(config, split)?;
        let is_train = split == "train";
        let target_size = augmentation_config.target_size;
        let augmentor = ImageAugmentor::new(augmentation_config);

        Ok(Self {
            items,
            is_train,
            target_size,
            augmentor,
        })
    }

    /// Collect image/mask path pairs from the dataset directory.
    ///
    /// This function replicates the logic from the original PyTorch dataset that
    /// finds all image files and their corresponding mask files.
    fn collect_dataset_items(
        config: &ModelConfig,
        split: &str,
    ) -> DatasetResult<Vec<(PathBuf, PathBuf)>> {
        let mut items = Vec::new();

        // Determine the dataset directory based on task and split
        let dataset_root = &config.path.data_root_dir;
        let task_name = match config.task.task {
            Task::DIS5K => "DIS5K",
            Task::COD => "COD",
            Task::HRSOD => "HRSOD",
            Task::General => "General",
            Task::General2k => "General2k",
            Task::Matting => "Matting",
        };

        // Map split names to directory names (following DIS5K convention)
        let split_dir = match split {
            "train" => "DIS-TR",
            "val" => "DIS-VD",
            "test1" => "DIS-TE1",
            "test2" => "DIS-TE2",
            "test3" => "DIS-TE3",
            "test4" => "DIS-TE4",
            _ => split, // Use as-is for custom splits
        };

        let image_root = Path::new(dataset_root)
            .join(task_name)
            .join(split_dir)
            .join("im");
        let mask_root = Path::new(dataset_root)
            .join(task_name)
            .join(split_dir)
            .join("gt");

        // Check if directories exist
        if !image_root.exists() {
            return Err(DatasetError::ImageDirectoryNotFound { path: image_root });
        }
        if !mask_root.exists() {
            return Err(DatasetError::MaskDirectoryNotFound { path: mask_root });
        }

        // Get valid image extensions from the image crate dynamically
        let valid_extensions = Self::get_supported_image_extensions();

        // Read image directory and find corresponding masks
        let image_dir =
            fs::read_dir(&image_root).map_err(|e| DatasetError::DirectoryReadFailed {
                path: image_root.clone(),
                source: e,
            })?;

        for entry in image_dir {
            let entry = entry.map_err(|e| DatasetError::DirectoryReadFailed {
                path: image_root.clone(),
                source: e,
            })?;

            let image_path = entry.path();
            if image_path.is_file() {
                // Safe handling of file name extraction
                let file_name = image_path
                    .file_name()
                    .ok_or_else(|| DatasetError::NoFileName {
                        path: image_path.clone(),
                    })?
                    .to_str()
                    .ok_or_else(|| DatasetError::InvalidUtf8Path {
                        path: image_path.clone(),
                    })?;

                // Check if file has valid extension
                if !valid_extensions.iter().any(|ext| file_name.ends_with(ext)) {
                    continue;
                }

                // Safe handling of file stem extraction
                let stem = image_path
                    .file_stem()
                    .ok_or_else(|| DatasetError::NoFileStem {
                        path: image_path.clone(),
                    })?
                    .to_str()
                    .ok_or_else(|| DatasetError::InvalidUtf8Path {
                        path: image_path.clone(),
                    })?;

                let mut mask_found = false;

                for ext in &valid_extensions {
                    let mask_path = mask_root.join(format!("{stem}{ext}"));
                    if mask_path.exists() {
                        items.push((image_path.clone(), mask_path));
                        mask_found = true;
                        break;
                    }
                }

                if !mask_found {
                    eprintln!("Warning: No mask found for image: {}", image_path.display());
                }
            }
        }

        if items.is_empty() {
            return Err(DatasetError::NoValidPairs { path: image_root });
        }

        println!(
            "Found {} image/mask pairs in {}",
            items.len(),
            image_root.display()
        );
        Ok(items)
    }

    /// Get supported image extensions from the image crate dynamically.
    ///
    /// This method uses the image crate's ImageFormat API to retrieve all supported
    /// file extensions, avoiding hardcoded magic numbers and ensuring compatibility
    /// with the image crate's actual capabilities.
    fn get_supported_image_extensions() -> Vec<String> {
        let mut extensions = Vec::new();

        // Iterate through all supported image formats
        for format in ImageFormat::all() {
            // Get extensions for this format and add them to our list
            for ext in format.extensions_str() {
                // Add both with and without leading dot for compatibility
                extensions.push(format!(".{}", ext));
                extensions.push(format!(".{}", ext.to_uppercase()));
            }
        }

        // Remove duplicates while preserving order
        extensions.sort();
        extensions.dedup();

        extensions
    }

    /// Convert an image to raw float array without normalization.
    /// Normalization will be applied later in the batcher for backend compatibility.
    fn image_to_array(&self, img: DynamicImage) -> Vec<f32> {
        // Convert to RGB and get raw pixel data
        let rgb_img = img.to_rgb32f();

        // The image crate already provides data in HWC format, so we can use it directly
        rgb_img.into_raw()
    }

    /// Convert a mask to normalized float array.
    fn mask_to_array(&self, mask: DynamicImage) -> Vec<f32> {
        let mask = mask.to_luma32f();
        mask.into_raw()
    }

    /// Apply augmentations to an image and its mask.
    /// Uses the comprehensive data augmentation system implemented in the augmentation module.
    fn augment(&self, image: DynamicImage, mask: DynamicImage) -> (DynamicImage, DynamicImage) {
        self.augmentor.augment(image, mask, self.is_train)
    }
}

impl Dataset<BiRefNetItem> for BiRefNetDataset {
    fn get(&self, index: usize) -> Option<BiRefNetItem> {
        let (image_path, mask_path) = self.items.get(index)?;

        // Load image with proper error logging
        let image = match image::open(image_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("Failed to open image {}: {}", image_path.display(), e);
                return None;
            }
        };

        // Load mask with proper error logging
        let mask = match image::open(mask_path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("Failed to open mask {}: {}", mask_path.display(), e);
                return None;
            }
        };

        // Apply data augmentation (full augmentation only during training, resizing only during validation)
        let (image, mask) = self.augment(image, mask);

        let height = image.height() as usize;
        let width = image.width() as usize;

        let image_data = self.image_to_array(image);
        let mask_data = self.mask_to_array(mask);

        Some(BiRefNetItem {
            image: image_data,
            mask: mask_data,
            height,
            width,
        })
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[cfg(test)]
mod tests {
    use burn::{data::dataloader::batcher::Batcher, prelude::*};

    use super::*;

    type TestBackend = burn::backend::ndarray::NdArray<f32>;

    #[test]
    fn birefnet_batcher_creates_correct_batch_dimensions() {
        let device = Default::default();
        let batcher = BiRefNetBatcher::<TestBackend>::new();

        let height = 32;
        let width = 32;

        // Create test items with raw data
        let item1 = BiRefNetItem {
            image: vec![0.5f32; height * width * 3], // RGB data
            mask: vec![1.0f32; height * width],      // Binary mask data
            height,
            width,
        };
        let item2 = BiRefNetItem {
            image: vec![0.3f32; height * width * 3], // RGB data
            mask: vec![0.0f32; height * width],      // Binary mask data
            height,
            width,
        };

        let items = vec![item1, item2];
        let batch = batcher.batch(items, &device);

        // Check batch dimensions
        assert_eq!(batch.images.shape().dims, [2, 3, 32, 32]); // [B, C, H, W]
        assert_eq!(batch.masks.shape().dims, [2, 1, 32, 32]); // [B, C, H, W]
    }

    #[test]
    fn birefnet_batch_creation_has_correct_tensor_shapes() {
        let device = Default::default();

        let images = Tensor::<TestBackend, 4>::zeros([4, 3, 64, 64], &device);
        let masks = Tensor::<TestBackend, 4>::zeros([4, 1, 64, 64], &device);

        let batch = BiRefNetBatch { images, masks };

        assert_eq!(batch.images.shape().dims, [4, 3, 64, 64]);
        assert_eq!(batch.masks.shape().dims, [4, 1, 64, 64]);
    }

    #[test]
    fn dataset_returns_raw_data() {
        // Test that BiRefNetItem contains raw data, not tensors
        let item = BiRefNetItem {
            image: vec![0.1, 0.2, 0.3],
            mask: vec![1.0],
            height: 1,
            width: 1,
        };

        assert_eq!(item.image.len(), 3); // RGB pixel
        assert_eq!(item.mask.len(), 1); // Single mask pixel
        assert_eq!(item.height, 1);
        assert_eq!(item.width, 1);
    }
}
