//! Dataset implementation for BiRefNet training and inference.
//!
//! This module provides an efficient dataset implementation for BiRefNet,
//! handling data loading, preprocessing, and augmentation directly.
//! This approach avoids the inefficiencies of reading files multiple times and
//! provides a clear framework for adding data augmentation.

use std::path::{Path, PathBuf};

use burn::data::{dataloader::batcher::Batcher, dataset::Dataset};
use burn::tensor::{backend::Backend, Tensor, TensorData};

use image::{self, imageops::FilterType, DynamicImage};

use crate::{
    config::{ModelConfig, Task},
    error::{BiRefNetError, BiRefNetResult},
};

/// Represents a single preprocessed data item from the BiRefNet dataset.
///
/// This struct contains the image and mask tensors after preprocessing,
/// equivalent to what the PyTorch `MyData.__getitem__` returns.
#[derive(Debug, Clone)]
pub struct BiRefNetItem<B: Backend> {
    /// Input image tensor with shape [C, H, W] where C=3 for RGB
    pub image: Tensor<B, 3>,
    /// Segmentation mask tensor with shape [C, H, W] where C=1 for binary masks
    pub mask: Tensor<B, 3>,
}

/// Represents a batch of preprocessed data items from the BiRefNet dataset.
///
/// This struct contains batched image and mask tensors suitable for training
/// and validation with the Burn framework.
#[derive(Debug, Clone)]
pub struct BiRefNetBatch<B: Backend> {
    /// Batched input image tensor with shape [B, C, H, W] where B=batch_size, C=3 for RGB
    pub images: Tensor<B, 4>,
    /// Batched segmentation mask tensor with shape [B, C, H, W] where B=batch_size, C=1 for binary masks
    pub masks: Tensor<B, 4>,
}

/// Batcher implementation for converting vectors of BiRefNetItem into BiRefNetBatch.
///
/// This batcher handles the conversion from individual data items to batched tensors,
/// following the same pattern as Burn's official examples.
#[derive(Clone, Default)]
pub struct BiRefNetBatcher<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> BiRefNetBatcher<B> {
    /// Create a new BiRefNet batcher.
    pub const fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend> Batcher<B, BiRefNetItem<B>, BiRefNetBatch<B>> for BiRefNetBatcher<B> {
    fn batch(&self, items: Vec<BiRefNetItem<B>>, _device: &B::Device) -> BiRefNetBatch<B> {
        let batch_size = items.len();

        // Pre-allocate vectors with known capacity to avoid reallocations
        let mut images = Vec::with_capacity(batch_size);
        let mut masks = Vec::with_capacity(batch_size);

        // Extract tensors directly into pre-allocated vectors
        // Using into_iter() to avoid cloning when possible
        for item in items {
            images.push(item.image);
            masks.push(item.mask);
        }

        // Stack tensors along the batch dimension (dim 0) to create [B, C, H, W] tensors
        let images = Tensor::stack(images, 0);
        let masks = Tensor::stack(masks, 0);

        BiRefNetBatch { images, masks }
    }
}

/// BiRefNet dataset for training and inference.
///
/// This dataset loads image/mask pairs from the DIS5K dataset structure and applies
/// preprocessing and augmentation.
pub struct BiRefNetDataset<B: Backend> {
    items: Vec<(PathBuf, PathBuf)>,
    device: B::Device,
    is_train: bool,
    target_size: (u32, u32),
    norm_mean: [f32; 3],
    norm_std: [f32; 3],
}

impl<B: Backend> BiRefNetDataset<B> {
    /// Create a new BiRefNet dataset.
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration containing dataset paths and task settings
    /// * `split` - Dataset split ("train", "val", "test")
    /// * `device` - Device to load tensors on
    ///
    /// # Returns
    ///
    /// A new dataset instance or an error if the dataset cannot be loaded.
    pub fn new(config: &ModelConfig, split: &str, device: &B::Device) -> BiRefNetResult<Self> {
        let items = Self::collect_dataset_items(config, split)?;
        let is_train = split == "train";

        // Use fixed target size for tensor preprocessing
        let target_size = (1024, 1024);

        // ImageNet normalization parameters (same as PyTorch implementation)
        let norm_mean = [0.485, 0.456, 0.406];
        let norm_std = [0.229, 0.224, 0.225];

        Ok(Self {
            items,
            device: device.clone(),
            is_train,
            target_size,
            norm_mean,
            norm_std,
        })
    }

    /// Collect image/mask path pairs from the dataset directory.
    ///
    /// This function replicates the logic from the original PyTorch dataset that
    /// finds all image files and their corresponding mask files.
    fn collect_dataset_items(
        config: &ModelConfig,
        split: &str,
    ) -> BiRefNetResult<Vec<(PathBuf, PathBuf)>> {
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
            return Err(BiRefNetError::DatasetError {
                message: format!("Image directory does not exist: {}", image_root.display()),
            });
        }
        if !mask_root.exists() {
            return Err(BiRefNetError::DatasetError {
                message: format!("Mask directory does not exist: {}", mask_root.display()),
            });
        }

        // Valid image extensions (same as PyTorch implementation)
        let valid_extensions = [".png", ".jpg", ".PNG", ".JPG", ".JPEG"];

        // Read image directory and find corresponding masks
        let image_dir =
            std::fs::read_dir(&image_root).map_err(|e| BiRefNetError::DatasetError {
                message: format!("Failed to read image directory: {e}"),
            })?;

        for entry in image_dir {
            let entry = entry.map_err(|e| BiRefNetError::DatasetError {
                message: format!("Failed to read directory entry: {e}"),
            })?;

            let image_path = entry.path();
            if image_path.is_file() {
                let file_name = image_path.file_name().unwrap().to_str().unwrap();

                // Check if file has valid extension
                if !valid_extensions.iter().any(|ext| file_name.ends_with(ext)) {
                    continue;
                }

                // Find corresponding mask file
                let stem = image_path.file_stem().unwrap().to_str().unwrap();
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
            return Err(BiRefNetError::DatasetError {
                message: format!(
                    "No valid image/mask pairs found in {}",
                    image_root.display()
                ),
            });
        }

        println!(
            "Found {} image/mask pairs in {}",
            items.len(),
            image_root.display()
        );
        Ok(items)
    }

    /// Convert an image to a tensor.
    fn image_to_tensor(&self, img: DynamicImage) -> Tensor<B, 3> {
        let img = img.to_rgb32f();
        let (width, height) = img.dimensions();
        let data = TensorData::new(img.into_raw(), [height as usize, width as usize, 3]);
        let tensor = Tensor::<B, 3>::from_data(data, &self.device);
        // HWC to CHW
        tensor.permute([2, 0, 1])
    }

    /// Convert a mask to a tensor.
    fn mask_to_tensor(&self, mask: DynamicImage) -> Tensor<B, 3> {
        let mask = mask.to_luma32f();
        let (width, height) = mask.dimensions();
        let data = TensorData::new(mask.into_raw(), [height as usize, width as usize]);
        let tensor = Tensor::<B, 3>::from_data(data, &self.device);
        // Add channel dimension and normalize
        tensor.permute([2, 0, 1])
    }

    /// Apply normalization to an image tensor.
    fn normalize_tensor(&self, tensor: Tensor<B, 3>) -> Tensor<B, 3> {
        let mean =
            Tensor::<B, 1>::from_data(TensorData::new(self.norm_mean.to_vec(), [3]), &self.device);
        let std =
            Tensor::<B, 1>::from_data(TensorData::new(self.norm_std.to_vec(), [3]), &self.device);

        // Reshape for broadcasting: [3, 1, 1]
        let mean = mean.reshape([3, 1, 1]);
        let std = std.reshape([3, 1, 1]);

        (tensor - mean) / std
    }

    /// Apply augmentations to an image and its mask.
    /// This is the place to add more complex data augmentation logic.
    fn augment(&self, image: DynamicImage, mask: DynamicImage) -> (DynamicImage, DynamicImage) {
        // Apply random data augmentations during training phase
        // For now, we just resize to the target size.
        let image =
            image.resize_exact(self.target_size.0, self.target_size.1, FilterType::Lanczos3);
        let mask = mask.resize_exact(self.target_size.0, self.target_size.1, FilterType::Nearest);

        (image, mask)
    }
}

impl<B: Backend> Dataset<BiRefNetItem<B>> for BiRefNetDataset<B> {
    fn get(&self, index: usize) -> Option<BiRefNetItem<B>> {
        let (image_path, mask_path) = self.items.get(index)?;

        let image = image::open(image_path).ok()?;
        let mask = image::open(mask_path).ok()?;

        let (image, mask) = if self.is_train {
            self.augment(image, mask)
        } else {
            // For validation/testing, just resize without other augmentations
            let image =
                image.resize_exact(self.target_size.0, self.target_size.1, FilterType::Lanczos3);
            let mask =
                mask.resize_exact(self.target_size.0, self.target_size.1, FilterType::Nearest);
            (image, mask)
        };

        let image_tensor = self.image_to_tensor(image);
        let mask_tensor = self.mask_to_tensor(mask);

        let image_tensor = self.normalize_tensor(image_tensor);

        Some(BiRefNetItem {
            image: image_tensor,
            mask: mask_tensor,
        })
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::{BiRefNetBatch, BiRefNetBatcher, BiRefNetItem, ModelConfig};
    use burn::backend::ndarray::NdArray;
    use burn::data::dataloader::batcher::Batcher;
    use burn::prelude::*;

    type TestBackend = NdArray;

    #[test]
    fn test_collect_dataset_items() {
        // TODO: This test would need a mock dataset structure
        // For now, we'll just test the basic functionality
        let _config = ModelConfig::new();
        // This would fail without actual dataset files, but shows the structure
        // let items = BiRefNetDataset::<TestBackend>::collect_dataset_items(&config, "train");
        // assert!(items.is_ok());
    }

    #[test]
    fn test_birefnet_batcher() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let batcher = BiRefNetBatcher::<TestBackend>::new();

        // Create test items
        let item1 = BiRefNetItem {
            image: Tensor::<TestBackend, 3>::random(
                [3, 32, 32],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            mask: Tensor::<TestBackend, 3>::random(
                [1, 32, 32],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
        };
        let item2 = BiRefNetItem {
            image: Tensor::<TestBackend, 3>::random(
                [3, 32, 32],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            mask: Tensor::<TestBackend, 3>::random(
                [1, 32, 32],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
        };

        let items = vec![item1, item2];
        let batch = batcher.batch(items, &device);

        // Check batch dimensions
        assert_eq!(batch.images.shape().dims, [2, 3, 32, 32]); // [B, C, H, W]
        assert_eq!(batch.masks.shape().dims, [2, 1, 32, 32]); // [B, C, H, W]
    }

    #[test]
    fn test_birefnet_batch_creation() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;

        let images = Tensor::<TestBackend, 4>::random(
            [4, 3, 64, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let masks = Tensor::<TestBackend, 4>::random(
            [4, 1, 64, 64],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );

        let batch = BiRefNetBatch { images, masks };

        assert_eq!(batch.images.shape().dims, [4, 3, 64, 64]);
        assert_eq!(batch.masks.shape().dims, [4, 1, 64, 64]);
    }
}
