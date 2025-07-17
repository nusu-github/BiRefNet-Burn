//! Dataset implementation for BiRefNet training and inference.
//!
//! This module provides a Rust implementation of the dataset loader that's equivalent
//! to the original PyTorch BiRefNet dataset functionality using the burn-dataset crate.
//!
//! The implementation follows the pattern:
//! 1. Collect image/mask path pairs from the dataset directories
//! 2. Use `ImageFolderDataset` to create a base dataset
//! 3. Wrap with `MapperDataset` to apply preprocessing transformations
//!
//! ## Key Components
//!
//! - `BiRefNetItem`: Represents a single preprocessed data item (image + mask)
//! - `BiRefNetDataset`: Main dataset struct that manages the entire dataset
//! - `BiRefNetMapper`: Handles preprocessing pipeline (resize, normalize, etc.)

use std::path::PathBuf;

use burn::data::{
    dataloader::batcher::Batcher,
    dataset::{
        transform::{Mapper, MapperDataset},
        vision::{Annotation, ImageDatasetItem, ImageFolderDataset, PixelDepth, SegmentationMask},
        Dataset,
    },
};
use burn::prelude::*;
use burn::tensor::{backend::Backend, Tensor};

#[cfg(feature = "train")]
use image::{self, GenericImageView};

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
        // Extract and collect image tensors
        let images: Vec<Tensor<B, 3>> = items.iter().map(|item| item.image.clone()).collect();

        // Extract and collect mask tensors
        let masks: Vec<Tensor<B, 3>> = items.iter().map(|item| item.mask.clone()).collect();

        // Stack tensors along the batch dimension (dim 0) to create [B, C, H, W] tensors
        let images = Tensor::stack(images, 0);
        let masks = Tensor::stack(masks, 0);

        BiRefNetBatch { images, masks }
    }
}

/// BiRefNet dataset for training and inference.
///
/// This dataset loads image/mask pairs from the DIS5K dataset structure and applies
/// preprocessing transformations similar to the original PyTorch implementation.
pub struct BiRefNetDataset<B: Backend> {
    dataset: MapperDataset<ImageFolderDataset, BiRefNetMapper<B>, ImageDatasetItem>,
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
        // Collect image/mask path pairs
        let items = Self::collect_dataset_items(config, split)?;

        // Create base ImageFolderDataset for segmentation
        let base_dataset = ImageFolderDataset::new_segmentation_with_items(items, &["mask"])
            .map_err(|e| BiRefNetError::DatasetError {
                message: format!("Failed to create ImageFolderDataset: {}", e),
            })?;

        // Create mapper for preprocessing
        let mapper = BiRefNetMapper::new(config, device.clone())?;

        // Wrap with MapperDataset
        let dataset = MapperDataset::new(base_dataset, mapper);

        Ok(Self { dataset })
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

        let image_root = dataset_root.join(task_name).join(split_dir).join("im");
        let mask_root = dataset_root.join(task_name).join(split_dir).join("gt");

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
                message: format!("Failed to read image directory: {}", e),
            })?;

        for entry in image_dir {
            let entry = entry.map_err(|e| BiRefNetError::DatasetError {
                message: format!("Failed to read directory entry: {}", e),
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
                    let mask_path = mask_root.join(format!("{}{}", stem, ext));
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
}

impl<B: Backend> Dataset<BiRefNetItem<B>> for BiRefNetDataset<B> {
    fn get(&self, index: usize) -> Option<BiRefNetItem<B>> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

/// Mapper for preprocessing BiRefNet dataset items.
///
/// This struct handles the preprocessing pipeline that converts raw image data
/// into tensors suitable for training/inference, similar to the transforms
/// applied in the original PyTorch implementation.
struct BiRefNetMapper<B: Backend> {
    device: B::Device,
    target_size: (usize, usize), // (height, width)
    // Image normalization parameters (ImageNet defaults)
    norm_mean: [f32; 3],
    norm_std: [f32; 3],
}

impl<B: Backend> BiRefNetMapper<B> {
    /// Create a new mapper with the given configuration.
    const fn new(_config: &ModelConfig, device: B::Device) -> BiRefNetResult<Self> {
        // Default target size (can be made configurable)
        let target_size = (1024, 1024);

        // ImageNet normalization parameters (same as PyTorch implementation)
        let norm_mean = [0.485, 0.456, 0.406];
        let norm_std = [0.229, 0.224, 0.225];

        Ok(Self {
            device,
            target_size,
            norm_mean,
            norm_std,
        })
    }

    /// Convert image pixels to tensor and apply preprocessing.
    fn preprocess_image(
        &self,
        pixels: &[u8],
        channels: usize,
        width: usize,
        height: usize,
    ) -> Tensor<B, 3> {
        // Ensure the pixel data matches the expected size
        let expected_size = height * width * channels;
        if pixels.len() != expected_size {
            panic!(
                "Pixel data size mismatch: expected {}, got {}",
                expected_size,
                pixels.len()
            );
        }

        // Convert pixels to tensor based on channels
        let tensor = if channels == 3 {
            // RGB image: [H, W, C]
            let tensor = Tensor::<B, 3>::from_data(
                TensorData::new(pixels.to_vec(), [height, width, channels]),
                &self.device,
            );
            tensor.int().float() / 255.0
        } else if channels == 1 {
            // Grayscale image: [H, W] -> [H, W, 1]
            let tensor = Tensor::<B, 2>::from_data(
                TensorData::new(pixels.to_vec(), [height, width]),
                &self.device,
            );
            let tensor = tensor.int().float() / 255.0;
            tensor.unsqueeze_dim(2)
        } else {
            panic!("Unsupported number of channels: {}", channels);
        };

        // Convert from [H, W, C] to [C, H, W]
        let tensor = tensor.permute([2, 0, 1]);

        // Apply normalization (only for RGB images)
        if channels == 3 {
            self.normalize_tensor(tensor)
        } else {
            tensor
        }
    }

    /// Convert mask pixels to tensor.
    fn preprocess_mask(
        &self,
        mask: &SegmentationMask,
        width: usize,
        height: usize,
    ) -> Tensor<B, 3> {
        // Convert mask to tensor [H, W]
        let mask_data: Vec<f32> = mask.mask.iter().map(|&x| x as f32 / 255.0).collect();
        let tensor =
            Tensor::<B, 2>::from_data(TensorData::new(mask_data, [height, width]), &self.device);

        // Add channel dimension to make it [C, H, W] where C=1
        tensor.unsqueeze_dim(0)
    }

    /// Get mask dimensions from the mask file.
    fn get_mask_dimensions(&self, mask_path: &str) -> (usize, usize) {
        let img = image::open(mask_path).expect("Failed to open mask image");
        let (width, height) = img.dimensions();
        (width as usize, height as usize)
    }

    /// Get mask file path from image path.
    fn get_mask_path_from_image_path(&self, image_path: &str) -> String {
        // Convert image path to mask path
        // Replace '/im/' with '/gt/' and change extension to .png
        let mask_path = image_path.replace("/im/", "/gt/");

        // Change extension to .png
        let path = std::path::Path::new(&mask_path);
        let stem = path.file_stem().unwrap().to_str().unwrap();
        let parent = path.parent().unwrap();
        let mask_path = parent.join(format!("{}.png", stem));

        mask_path.to_str().unwrap().to_string()
    }

    /// Apply normalization to image tensor.
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

    /// Get actual image dimensions from the image file.
    fn get_actual_image_info(&self, image_path: &str) -> (usize, usize, usize) {
        // Load image to get actual dimensions
        let img = image::open(image_path).expect("Failed to open image");
        let (width, height) = img.dimensions();

        // Determine the number of channels based on color type
        let channels = match img.color() {
            image::ColorType::L8 | image::ColorType::L16 => 1,
            image::ColorType::La8 | image::ColorType::La16 => 2,
            image::ColorType::Rgb8 | image::ColorType::Rgb16 | image::ColorType::Rgb32F => 3,
            image::ColorType::Rgba8 | image::ColorType::Rgba16 | image::ColorType::Rgba32F => 4,
            _ => 3, // Default to RGB
        };

        (channels, width as usize, height as usize)
    }
}

impl<B: Backend> Mapper<ImageDatasetItem, BiRefNetItem<B>> for BiRefNetMapper<B> {
    fn map(&self, item: &ImageDatasetItem) -> BiRefNetItem<B> {
        // Get actual image dimensions from the image file
        let (channels, width, height) = self.get_actual_image_info(&item.image_path);

        // Convert image pixels to bytes
        let image_bytes = Self::convert_pixels_to_rgb(&item.image);

        // Preprocess image
        let image = self.preprocess_image(&image_bytes, channels, width, height);

        // Preprocess mask
        let mask = match &item.annotation {
            Annotation::SegmentationMask(mask) => {
                // Get mask file path from image path
                let mask_path = self.get_mask_path_from_image_path(&item.image_path);
                let (mask_width, mask_height) = self.get_mask_dimensions(&mask_path);
                self.preprocess_mask(mask, mask_width, mask_height)
            }
            _ => {
                // If not a segmentation mask, create a dummy mask
                Tensor::zeros([1, self.target_size.0, self.target_size.1], &self.device)
            }
        };

        BiRefNetItem { image, mask }
    }
}

impl<B: Backend> BiRefNetMapper<B> {
    /// Convert pixel data to RGB bytes.
    fn convert_pixels_to_rgb(pixels: &[PixelDepth]) -> Vec<u8> {
        pixels
            .iter()
            .map(|pixel| match pixel {
                PixelDepth::U8(v) => *v,
                PixelDepth::U16(v) => (*v / 256) as u8,
                PixelDepth::F32(v) => (v * 255.0) as u8,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::ndarray::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_collect_dataset_items() {
        // This test would need a mock dataset structure
        // For now, we'll just test the basic functionality
        let config = ModelConfig::new();
        // This would fail without actual dataset files, but shows the structure
        // let items = BiRefNetDataset::<TestBackend>::collect_dataset_items(&config, "train");
        // assert!(items.is_ok());
    }

    #[test]
    fn test_mapper_creation() {
        let config = ModelConfig::new();
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let mapper = BiRefNetMapper::<TestBackend>::new(&config, device);
        assert!(mapper.is_ok());
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
