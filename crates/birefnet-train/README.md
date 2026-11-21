# birefnet-train

[![Crates.io](https://img.shields.io/crates/v/birefnet-train.svg)](https://crates.io/crates/birefnet-train)
[![Documentation](https://docs.rs/birefnet-train/badge.svg)](https://docs.rs/birefnet-train)

**Dataset discovery, image augmentations, and tensor batching for `BiRefNet` training on Burn**

## Core Components

### Dataset Management

- **`BiRefNetDataset`**: Discovers sorted paired images and ground-truth masks under `<root>/<task>/<split>/{im,gt}` (`DatasetSplit::Train`, `Val`, `Test1`–`Test4`, or `Custom(String)`), returning `TrainDataError` for missing directories or empty datasets.
- **`BiRefNetItem`**: Preprocessed raw RGB `[H, W, 3]` and Luma `[H, W]` pixel buffers ready for Burn `DataLoaderBuilder`.
- **`BiRefNetBatcher`**: Stacks `BiRefNetItem` batches into normalized `[B, 3, H, W]` / `[B, 1, H, W]` `BiRefNetBatch` tensors.
- **`TrainDataError`**: Typed error with `is_not_found`, `is_empty_dataset`, `is_io`, `is_image`, and `is_invalid_path` predicates.

### Data Augmentation

- **`ImageAugmentor` & `AugmentationConfig`**: Thread-safe, atomic-seed-advancing pipeline (`new`, `with_seed`) implementing horizontal flip, border crop, center rotation (`Rotate`), color enhancement (brightness, contrast, hue, Laplacian sharpen / Gaussian blur), salt-and-pepper noise, synthetic background compositing (`BackgroundPattern`), and 32-pixel-aligned dynamic input sizing (`DynamicSizeConfig`).
- **Self-contained image transforms**: Rotation and 3x3 sharpening implemented directly with `image` without `imageproc`.

## Usage

```rust,no_run
use birefnet_model::Task;
use birefnet_train::{AugmentationConfig, BiRefNetBatcher, BiRefNetDataset, DatasetSplit};
use burn::{data::dataloader::DataLoaderBuilder, tensor::Device};

let dataset = BiRefNetDataset::with_augmentation(
    "/path/to/datasets",
    Task::DIS5K,
    &DatasetSplit::Train,
    AugmentationConfig::default(),
)?;

let device = Device::flex().autodiff();
let dataloader = DataLoaderBuilder::new(BiRefNetBatcher::new())
    .batch_size(4)
    .shuffle(42)
    .set_device(device)
    .build(dataset);
# let _ = dataloader;
# Ok::<(), birefnet_train::TrainDataError>(())
```

## Dataset Layout

```text
<root_dir>/
└── DIS5K/
    ├── DIS-TR/
    │   ├── im/   # Training RGB images
    │   └── gt/   # Training binary masks
    └── DIS-VD/
        ├── im/   # Validation RGB images
        └── gt/   # Validation binary masks
```

## License

MIT OR Apache-2.0
