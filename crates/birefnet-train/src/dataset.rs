//! Dataset discovery, loading, and batching for `BiRefNet` training and validation.

use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use birefnet_model::{BiRefNetBatch, Task};
use birefnet_util::apply_imagenet_normalization;
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{Device, Tensor, TensorData},
};
use image::{DynamicImage, ImageFormat};

use crate::{
    augmentation::{AugmentationConfig, ImageAugmentor},
    error::TrainDataError,
};

/// Dataset split selecting subdirectory conventions under `<root>/<task>/`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DatasetSplit {
    /// Training split (`DIS-TR`).
    Train,
    /// Validation split (`DIS-VD`).
    Val,
    /// First test split (`DIS-TE1`).
    Test1,
    /// Second test split (`DIS-TE2`).
    Test2,
    /// Third test split (`DIS-TE3`).
    Test3,
    /// Fourth test split (`DIS-TE4`).
    Test4,
    /// Custom directory name relative to the task root.
    Custom(String),
}

impl DatasetSplit {
    /// Returns the subdirectory name expected under `<root>/<task>/`.
    #[must_use]
    pub fn dir_name(&self) -> &str {
        match self {
            Self::Train => "DIS-TR",
            Self::Val => "DIS-VD",
            Self::Test1 => "DIS-TE1",
            Self::Test2 => "DIS-TE2",
            Self::Test3 => "DIS-TE3",
            Self::Test4 => "DIS-TE4",
            Self::Custom(name) => name.as_str(),
        }
    }

    /// Returns `true` when full training augmentations should be enabled.
    #[must_use]
    pub const fn is_train(&self) -> bool {
        matches!(self, Self::Train)
    }
}

/// Single preprocessed image/mask pair emitted by [`BiRefNetDataset`].
#[derive(Debug, Clone)]
pub struct BiRefNetItem {
    /// Flattened `[H, W, 3]` RGB pixel values in `[0.0, 1.0]`.
    pub image: Vec<f32>,
    /// Flattened `[H, W]` grayscale mask values in `[0.0, 1.0]`.
    pub mask: Vec<f32>,
    /// Preprocessed image height in pixels.
    pub height: usize,
    /// Preprocessed image width in pixels.
    pub width: usize,
}

/// Converts a batch of [`BiRefNetItem`]s into normalized `[B, 3, H, W]` / `[B, 1, H, W]` tensors.
#[derive(Clone, Copy, Debug, Default)]
pub struct BiRefNetBatcher;

impl BiRefNetBatcher {
    /// Creates a new [`BiRefNetBatcher`].
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Batcher<BiRefNetItem, BiRefNetBatch> for BiRefNetBatcher {
    fn batch(&self, items: Vec<BiRefNetItem>, device: &Device) -> BiRefNetBatch {
        let batch_size = items.len();
        let mut images = Vec::with_capacity(batch_size);
        let mut masks = Vec::with_capacity(batch_size);

        for item in items {
            let image_tensor = Tensor::<3>::from_data(
                TensorData::new(item.image, [item.height, item.width, 3]),
                device,
            )
            .permute([2, 0, 1]);

            let normalized_tensor = apply_imagenet_normalization(image_tensor.unsqueeze::<4>())
                .expect("3-channel RGB tensor matches ImageNet normalization")
                .squeeze_dim::<3>(0);

            let mask_tensor = Tensor::<2>::from_data(
                TensorData::new(item.mask, [item.height, item.width]),
                device,
            )
            .unsqueeze::<3>();

            images.push(normalized_tensor);
            masks.push(mask_tensor);
        }

        BiRefNetBatch::new(Tensor::stack(images, 0), Tensor::stack(masks, 0))
    }
}

const fn task_dir_name(task: Task) -> &'static str {
    match task {
        Task::DIS5K => "DIS5K",
        Task::COD => "COD",
        Task::HRSOD => "HRSOD",
        Task::General => "General",
        Task::General2k => "General2k",
        Task::Matting => "Matting",
    }
}

fn supported_image_extensions() -> Vec<&'static str> {
    let mut extensions: Vec<&'static str> = ImageFormat::all()
        .filter(ImageFormat::reading_enabled)
        .flat_map(|format| format.extensions_str().iter().copied())
        .collect();
    extensions.sort_unstable();
    extensions.dedup();
    extensions
}

fn has_supported_extension(path: &Path, valid_extensions: &[&str]) -> bool {
    let Some(ext) = path.extension().and_then(|ext| ext.to_str()) else {
        return false;
    };
    valid_extensions
        .iter()
        .any(|valid| ext.eq_ignore_ascii_case(valid))
}

fn read_dir_entries(dir: &Path) -> Result<Vec<PathBuf>, TrainDataError> {
    let reader = fs::read_dir(dir).map_err(|err| TrainDataError::io(dir, err))?;
    let mut paths = Vec::new();
    for entry in reader {
        let entry = entry.map_err(|err| TrainDataError::io(dir, err))?;
        paths.push(entry.path());
    }
    paths.sort();
    Ok(paths)
}

/// Dataset loading paired segmentation images and masks for [`BiRefNetDataset`].
#[derive(Debug)]
pub struct BiRefNetDataset {
    items: Vec<(PathBuf, PathBuf)>,
    is_train: bool,
    augmentor: ImageAugmentor,
}

impl BiRefNetDataset {
    /// Discovers paired image/mask samples under `<root_dir>/<task>/<split>/{im,gt}`
    /// using default [`AugmentationConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`TrainDataError`] if expected directories are missing, contain invalid UTF-8
    /// filenames, or yield no matching pairs.
    pub fn new(
        root_dir: impl AsRef<Path>,
        task: Task,
        split: &DatasetSplit,
    ) -> Result<Self, TrainDataError> {
        Self::with_augmentation(root_dir, task, split, AugmentationConfig::default())
    }

    /// Discovers paired image/mask samples using an explicit `augmentation_config`.
    ///
    /// # Errors
    ///
    /// Returns [`TrainDataError`] if expected directories are missing, contain invalid UTF-8
    /// filenames, or yield no matching pairs.
    pub fn with_augmentation(
        root_dir: impl AsRef<Path>,
        task: Task,
        split: &DatasetSplit,
        augmentation_config: AugmentationConfig,
    ) -> Result<Self, TrainDataError> {
        let items = Self::collect_dataset_items(root_dir.as_ref(), task, split)?;
        let augmentor = ImageAugmentor::new(augmentation_config);

        Ok(Self {
            items,
            is_train: split.is_train(),
            augmentor,
        })
    }

    fn collect_dataset_items(
        root_dir: &Path,
        task: Task,
        split: &DatasetSplit,
    ) -> Result<Vec<(PathBuf, PathBuf)>, TrainDataError> {
        let split_root = root_dir.join(task_dir_name(task)).join(split.dir_name());
        let image_root = split_root.join("im");
        let mask_root = split_root.join("gt");

        if !image_root.exists() {
            return Err(TrainDataError::not_found(&image_root));
        }
        if !mask_root.exists() {
            return Err(TrainDataError::not_found(&mask_root));
        }

        let valid_extensions = supported_image_extensions();
        let mut masks_by_stem: BTreeMap<String, PathBuf> = BTreeMap::new();
        for mask_path in read_dir_entries(&mask_root)? {
            if mask_path.is_file() && has_supported_extension(&mask_path, &valid_extensions) {
                let stem = mask_path
                    .file_stem()
                    .ok_or_else(|| TrainDataError::invalid_path(&mask_path, "missing file stem"))?
                    .to_str()
                    .ok_or_else(|| {
                        TrainDataError::invalid_path(&mask_path, "non-UTF-8 file stem")
                    })?;
                masks_by_stem.entry(stem.to_owned()).or_insert(mask_path);
            }
        }

        let mut items = Vec::new();
        for image_path in read_dir_entries(&image_root)? {
            if !image_path.is_file() || !has_supported_extension(&image_path, &valid_extensions) {
                continue;
            }

            let stem = image_path
                .file_stem()
                .ok_or_else(|| TrainDataError::invalid_path(&image_path, "missing file stem"))?
                .to_str()
                .ok_or_else(|| TrainDataError::invalid_path(&image_path, "non-UTF-8 file stem"))?;

            if let Some(mask_path) = masks_by_stem.get(stem) {
                items.push((image_path, mask_path.clone()));
            } else {
                tracing::warn!(image = %image_path.display(), "no matching mask found in gt directory");
            }
        }

        if items.is_empty() {
            return Err(TrainDataError::empty_dataset(&image_root));
        }

        items.sort_by(|a, b| a.0.cmp(&b.0));
        tracing::info!(
            pairs = items.len(),
            directory = %image_root.display(),
            "discovered image/mask dataset pairs"
        );
        Ok(items)
    }

    fn open_image(path: &Path) -> Result<DynamicImage, burn::data::dataset::DatasetError> {
        image::open(path).map_err(|source| {
            burn::data::dataset::DatasetError::new(TrainDataError::image(path, source))
        })
    }
}

impl Dataset<BiRefNetItem> for BiRefNetDataset {
    fn get(&self, index: usize) -> Result<BiRefNetItem, burn::data::dataset::DatasetError> {
        let (image_path, mask_path) = &self.items[index];
        let image = Self::open_image(image_path)?;
        let mask = Self::open_image(mask_path)?;

        let (image, mask) = self.augmentor.augment(image, mask, self.is_train);
        let height = image.height() as usize;
        let width = image.width() as usize;

        Ok(BiRefNetItem {
            image: image.to_rgb32f().into_raw(),
            mask: mask.to_luma32f().into_raw(),
            height,
            width,
        })
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}
