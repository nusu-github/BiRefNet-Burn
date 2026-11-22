//! Dataset discovery, loading, and batching for BiRefNet training and validation.

use core::range::RangeInclusive;
use std::{
    collections::HashMap,
    fmt, fs,
    ops::{Bound, RangeBounds},
    path::{Path, PathBuf},
    sync::Arc,
};

use birefnet_model::{BiRefNetBatch, Task};
use birefnet_util::normalize::apply_imagenet_normalization;
use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, DatasetError},
    },
    tensor::{Device, Tensor, TensorData},
};
use image::{GrayImage, ImageFormat, ImageReader, RgbImage, imageops::FilterType};
use rand::RngExt;

use crate::{
    augmentation::{ImageAugmentor, SeedSequence},
    error::TrainDataError,
};

/// Dataset split selecting the directory under `<root>/<task>/`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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
            Self::Custom(name) => name,
        }
    }
}

impl fmt::Display for DatasetSplit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.dir_name())
    }
}

/// One loaded (and possibly augmented) image / mask pair of matching size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BiRefNetItem {
    /// RGB image.
    pub image: RgbImage,
    /// Grayscale ground-truth mask.
    pub mask: GrayImage,
}

/// Side length alignment required by the backbones' 32x total stride.
const SIZE_ALIGN: u32 = 32;

/// Per-batch random resolution, drawn independently for width and height from the multiples
/// of 32 inside the given ranges (upstream `dynamic_size`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DynamicSize {
    width: RangeInclusive<u32>,
    height: RangeInclusive<u32>,
}

fn inclusive(range: &impl RangeBounds<u32>) -> Option<RangeInclusive<u32>> {
    let start = match range.start_bound() {
        Bound::Included(&s) => s,
        Bound::Excluded(&s) => s.checked_add(1)?,
        Bound::Unbounded => 0,
    };
    let last = match range.end_bound() {
        Bound::Included(&e) => e,
        Bound::Excluded(&e) => e.checked_sub(1)?,
        Bound::Unbounded => u32::MAX,
    };
    Some(RangeInclusive { start, last })
}

/// Multiples of `SIZE_ALIGN` (at least one) inside `range`, as `(first, last)` multipliers.
fn aligned_steps(range: RangeInclusive<u32>) -> Option<(u32, u32)> {
    let first = range.start.div_ceil(SIZE_ALIGN).max(1);
    let last = range.last / SIZE_ALIGN;
    (first <= last).then_some((first, last))
}

impl DynamicSize {
    /// Creates a dynamic size from width and height ranges in pixels.
    ///
    /// # Errors
    ///
    /// Returns [`TrainDataError`] if either range contains no positive multiple of 32.
    pub fn new(
        width: impl RangeBounds<u32>,
        height: impl RangeBounds<u32>,
    ) -> Result<Self, TrainDataError> {
        let checked = |range: Option<RangeInclusive<u32>>, axis: &str| {
            range
                .filter(|&r| aligned_steps(r).is_some())
                .ok_or_else(|| {
                    TrainDataError::invalid_config(format!(
                        "dynamic {axis} range must contain a positive multiple of {SIZE_ALIGN}"
                    ))
                })
        };
        Ok(Self {
            width: checked(inclusive(&width), "width")?,
            height: checked(inclusive(&height), "height")?,
        })
    }

    /// Creates a dynamic size using the same range for width and height.
    ///
    /// # Errors
    ///
    /// Returns [`TrainDataError`] if the range contains no positive multiple of 32.
    pub fn square(range: impl RangeBounds<u32>) -> Result<Self, TrainDataError> {
        let range = inclusive(&range).unwrap_or(RangeInclusive { start: 1, last: 0 });
        Self::new(range, range)
    }

    /// Returns the width range in pixels.
    #[must_use]
    pub const fn width(&self) -> RangeInclusive<u32> {
        self.width
    }

    /// Returns the height range in pixels.
    #[must_use]
    pub const fn height(&self) -> RangeInclusive<u32> {
        self.height
    }

    fn sample(&self, rng: &mut impl rand::Rng) -> [u32; 2] {
        let mut draw = |range: RangeInclusive<u32>| {
            let (first, last) = aligned_steps(range)
                .unwrap_or_else(|| panic!("DynamicSize range {range:?} was validated"));
            rng.random_range(first..=last) * SIZE_ALIGN
        };
        [draw(self.width), draw(self.height)]
    }
}

/// Spatial size every batch is resized to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Resolution {
    /// The same `[width, height]` for every batch.
    Fixed([u32; 2]),
    /// A new random size per batch.
    Dynamic(DynamicSize),
}

impl Resolution {
    /// A fixed `[width, height]` resolution.
    ///
    /// # Errors
    ///
    /// Returns [`TrainDataError`] if either dimension is zero.
    pub fn fixed(width: u32, height: u32) -> Result<Self, TrainDataError> {
        if width == 0 || height == 0 {
            return Err(TrainDataError::invalid_config(format!(
                "resolution must be positive, got {width}x{height}"
            )));
        }
        Ok(Self::Fixed([width, height]))
    }
}

/// Resizes, normalizes, and uploads [`BiRefNetItem`]s as one `[B, 3, H, W]` / `[B, 1, H, W]`
/// batch.
///
/// Images and masks are resized with bilinear filtering (as `torchvision.transforms.Resize`
/// does upstream), scaled to `[0, 1]`, and uploaded in a single transfer each; images are then
/// ImageNet-normalized on the device.
#[derive(Debug, Clone)]
pub struct BiRefNetBatcher {
    resolution: Resolution,
    seeds: Arc<SeedSequence>,
}

impl BiRefNetBatcher {
    /// Creates a batcher; dynamic resolutions are seeded from the operating system's entropy.
    #[must_use]
    pub fn new(resolution: Resolution) -> Self {
        Self::with_seed(resolution, rand::random())
    }

    /// Creates a batcher with a reproducible `seed` for dynamic resolutions.
    #[must_use]
    pub fn with_seed(resolution: Resolution, seed: u64) -> Self {
        Self {
            resolution,
            seeds: Arc::new(SeedSequence::new(seed)),
        }
    }

    /// Returns the batch resolution.
    #[must_use]
    pub const fn resolution(&self) -> Resolution {
        self.resolution
    }

    fn batch_size(&self) -> [u32; 2] {
        match self.resolution {
            Resolution::Fixed(size) => size,
            Resolution::Dynamic(dynamic) => dynamic.sample(&mut self.seeds.next_rng()),
        }
    }
}

impl Batcher<BiRefNetItem, BiRefNetBatch> for BiRefNetBatcher {
    /// # Panics
    ///
    /// Panics if `items` is empty (Burn never requests an empty batch).
    fn batch(&self, items: Vec<BiRefNetItem>, device: &Device) -> BiRefNetBatch {
        assert!(!items.is_empty(), "BiRefNetBatcher received an empty batch");
        let [width, height] = self.batch_size();
        let (w, h) = (width as usize, height as usize);
        let batch = items.len();

        let mut images = Vec::with_capacity(batch * h * w * 3);
        let mut masks = Vec::with_capacity(batch * h * w);
        for item in items {
            let image = resized(item.image, width, height);
            let mask = resized(item.mask, width, height);
            images.extend(image.as_raw().iter().map(|&v| f32::from(v) / 255.0));
            masks.extend(mask.as_raw().iter().map(|&v| f32::from(v) / 255.0));
        }

        let images = Tensor::<4>::from_data(TensorData::new(images, [batch, h, w, 3]), device)
            .permute([0, 3, 1, 2]);
        let images = apply_imagenet_normalization(images)
            .unwrap_or_else(|err| panic!("RGB batch has 3 channels: {err}"));
        let masks = Tensor::<4>::from_data(TensorData::new(masks, [batch, 1, h, w]), device);
        BiRefNetBatch::new(images, masks)
    }
}

fn resized<P>(
    image: image::ImageBuffer<P, Vec<u8>>,
    width: u32,
    height: u32,
) -> image::ImageBuffer<P, Vec<u8>>
where
    P: image::Pixel<Subpixel = u8> + 'static,
{
    if image.dimensions() == (width, height) {
        image
    } else {
        image::imageops::resize(&image, width, height, FilterType::Triangle)
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

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .and_then(ImageFormat::from_extension)
        .is_some_and(|format| format.reading_enabled())
}

fn image_files(dir: &Path) -> Result<Vec<PathBuf>, TrainDataError> {
    let reader = fs::read_dir(dir).map_err(|err| TrainDataError::io(dir, err))?;
    let mut paths = Vec::new();
    for entry in reader {
        let path = entry.map_err(|err| TrainDataError::io(dir, err))?.path();
        if path.is_file() && is_supported_image(&path) {
            paths.push(path);
        }
    }
    Ok(paths)
}

fn file_stem(path: &Path) -> Result<&str, TrainDataError> {
    path.file_stem()
        .ok_or_else(|| TrainDataError::invalid_path(path, "missing file stem"))?
        .to_str()
        .ok_or_else(|| TrainDataError::invalid_path(path, "non-UTF-8 file stem"))
}

/// Pairs every image under `<root>/<task>/<split>/im` with the same-stem mask under `gt`.
fn discover_pairs(
    root_dir: &Path,
    task: Task,
    split: &DatasetSplit,
) -> Result<Vec<(PathBuf, PathBuf)>, TrainDataError> {
    let split_root = root_dir.join(task_dir_name(task)).join(split.dir_name());
    let image_root = split_root.join("im");
    let mask_root = split_root.join("gt");
    for dir in [&image_root, &mask_root] {
        if !dir.is_dir() {
            return Err(TrainDataError::not_found(dir));
        }
    }

    let mask_paths = image_files(&mask_root)?;
    let mut masks_by_stem = HashMap::with_capacity(mask_paths.len());
    for mask_path in mask_paths {
        let stem = file_stem(&mask_path)?.to_owned();
        masks_by_stem.entry(stem).or_insert(mask_path);
    }

    let mut pairs = Vec::new();
    let mut unmatched = 0_usize;
    for image_path in image_files(&image_root)? {
        match masks_by_stem.get(file_stem(&image_path)?) {
            Some(mask_path) => pairs.push((image_path, mask_path.clone())),
            None => unmatched += 1,
        }
    }
    if unmatched > 0 {
        tracing::warn!(
            name: "birefnet.train.dataset.discover.unmatched",
            unmatched,
            file.path = %image_root.display(),
            "{unmatched} images have no mask with the same file stem and were skipped"
        );
    }
    if pairs.is_empty() {
        return Err(TrainDataError::empty_dataset(&image_root));
    }

    pairs.sort_unstable();
    tracing::info!(
        name: "birefnet.train.dataset.discover.done",
        pairs = pairs.len(),
        file.path = %image_root.display(),
        "discovered {} image/mask pairs",
        pairs.len()
    );
    Ok(pairs)
}

fn open_image(path: &Path) -> Result<image::DynamicImage, TrainDataError> {
    let file = fs::File::open(path).map_err(|err| TrainDataError::io(path, err))?;
    ImageReader::new(std::io::BufReader::new(file))
        .with_guessed_format()
        .map_err(|err| TrainDataError::io(path, err))?
        .decode()
        .map_err(|err| TrainDataError::image(path, err))
}

/// Paired segmentation images and masks loaded from disk, optionally augmented.
///
/// Items keep their source resolution (after augmentation); [`BiRefNetBatcher`] resizes each
/// batch to a common size.
#[derive(Debug)]
pub struct BiRefNetDataset {
    items: Box<[(PathBuf, PathBuf)]>,
    augmentor: Option<ImageAugmentor>,
}

impl BiRefNetDataset {
    /// Discovers image / mask pairs under `<root_dir>/<task>/<split>/{im,gt}` without
    /// augmentation.
    ///
    /// # Errors
    ///
    /// Returns [`TrainDataError`] if a directory is missing or unreadable, a file name is not
    /// UTF-8, or no pairs are found.
    pub fn new(
        root_dir: impl AsRef<Path>,
        task: Task,
        split: &DatasetSplit,
    ) -> Result<Self, TrainDataError> {
        Ok(Self {
            items: discover_pairs(root_dir.as_ref(), task, split)?.into_boxed_slice(),
            augmentor: None,
        })
    }

    /// Applies `augmentor` to every loaded sample (use for training splits only).
    #[must_use]
    pub fn with_augmentor(self, augmentor: ImageAugmentor) -> Self {
        Self {
            augmentor: Some(augmentor),
            ..self
        }
    }

    /// Returns the augmentor, if any.
    #[must_use]
    pub const fn augmentor(&self) -> Option<&ImageAugmentor> {
        self.augmentor.as_ref()
    }

    /// Returns the `(image, mask)` paths of every sample.
    #[must_use]
    pub fn pairs(&self) -> &[(PathBuf, PathBuf)] {
        &self.items
    }

    /// Returns the number of samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` if the dataset has no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Loads (and augments) sample `index`.
    ///
    /// # Errors
    ///
    /// Returns [`TrainDataError`] if `index` is out of bounds, a file cannot be read or decoded,
    /// or the image and mask sizes differ.
    pub fn get(&self, index: usize) -> Result<BiRefNetItem, TrainDataError> {
        let (image_path, mask_path) = self
            .items
            .get(index)
            .ok_or_else(|| TrainDataError::index_out_of_bounds(index, self.items.len()))?;
        let image = open_image(image_path)?.into_rgb8();
        let mask = open_image(mask_path)?.into_luma8();
        if image.dimensions() != mask.dimensions() {
            return Err(TrainDataError::dimension_mismatch(
                mask_path,
                image.dimensions(),
                mask.dimensions(),
            ));
        }
        let (image, mask) = match &self.augmentor {
            Some(augmentor) => augmentor.augment(image, mask),
            None => (image, mask),
        };
        Ok(BiRefNetItem { image, mask })
    }
}

impl Dataset<BiRefNetItem> for BiRefNetDataset {
    fn get(&self, index: usize) -> Result<BiRefNetItem, DatasetError> {
        Self::get(self, index).map_err(DatasetError::new)
    }

    fn len(&self) -> usize {
        Self::len(self)
    }
}
