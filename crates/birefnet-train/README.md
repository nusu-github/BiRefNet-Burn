# birefnet-train

Dataset discovery, training augmentations, and batching for BiRefNet on Burn.

- `BiRefNetDataset` pairs `<root>/<task>/<split>/im/*` with same-stem masks in `gt/`, decodes
  them by file content, and optionally augments each sample with an `ImageAugmentor`. Items keep
  their source resolution until batching.
- `BiRefNetBatcher` resizes every item in a batch to one `Resolution` — fixed `(width, height)`,
  or a fresh 32-aligned `DynamicSize` per batch matching upstream `dynamic_size` training — then
  uploads images and masks once and applies ImageNet normalization on the device.
- `AugmentationConfig` selects the upstream `preproc_methods` (flip, enhance, rotate, pepper,
  crop) and optional background color synthesis.

```text
<root>/
└── DIS5K/
    ├── DIS-TR/{im,gt}/   # training images and masks
    └── DIS-VD/{im,gt}/   # validation images and masks
```

> Note: `Resolution::fixed(width, height)` and `DynamicSize` use `[width, height]` order, and
> `Task::General2k` maps to the directory name `General2k`.

## Examples

Building Burn data loaders for training and validation:

```rust
use birefnet_model::Task;
use birefnet_train::{
    AugmentationConfig, BiRefNetBatcher, BiRefNetDataset, DatasetSplit, DynamicSize,
    ImageAugmentor, Resolution, TrainDataError,
};
use burn::{data::dataloader::DataLoaderBuilder, tensor::Device};

fn main() -> Result<(), TrainDataError> {
    let train = BiRefNetDataset::new("/data", Task::DIS5K, &DatasetSplit::Train)?
        .with_augmentor(ImageAugmentor::new(AugmentationConfig::new()));
    let valid = BiRefNetDataset::new("/data", Task::DIS5K, &DatasetSplit::Val)?;

    let device = Device::flex().autodiff();
    let _train_loader = DataLoaderBuilder::new(BiRefNetBatcher::new(Resolution::Dynamic(
        DynamicSize::square(256..=2304)?,
    )))
    .batch_size(4)
    .shuffle(42)
    .set_device(device.clone())
    .build(train);
    let _valid_loader =
        DataLoaderBuilder::new(BiRefNetBatcher::new(Resolution::fixed(1024, 1024)?))
            .batch_size(4)
            .set_device(device)
            .build(valid);
    Ok(())
}
```

Building a batch directly from in-memory images:

```rust
use birefnet_train::{BiRefNetBatcher, BiRefNetItem, Resolution, TrainDataError};
use burn::{data::dataloader::batcher::Batcher, tensor::Device};
use image::{GrayImage, RgbImage};

fn main() -> Result<(), TrainDataError> {
    let items = vec![
        BiRefNetItem { image: RgbImage::new(40, 30), mask: GrayImage::new(40, 30) },
        BiRefNetItem { image: RgbImage::new(64, 64), mask: GrayImage::new(64, 64) },
    ];
    let batch = BiRefNetBatcher::new(Resolution::fixed(32, 32)?).batch(items, &Device::flex());
    assert_eq!(batch.images.dims(), [2, 3, 32, 32]);
    Ok(())
}
```

## License

MIT OR Apache-2.0
