//! Integration tests for dataset discovery, augmentations, and tensor batching.

use std::{collections::HashSet, path::Path};

use birefnet_model::Task;
use birefnet_train::{
    AugmentationConfig, AugmentationMethod, BiRefNetBatcher, BiRefNetDataset, BiRefNetItem,
    DatasetSplit, DynamicSize, ImageAugmentor, Resolution,
};
use burn::{data::dataloader::batcher::Batcher, prelude::*};
use image::{GrayImage, Luma, Rgb, RgbImage};
use tempfile::TempDir;

fn write_sample_pair(im_dir: &Path, gt_dir: &Path, stem: &str, size: (u32, u32)) {
    let (w, h) = size;
    let img = RgbImage::from_fn(w, h, |x, y| {
        if x + y > 18 {
            Rgb([200, 100, 50])
        } else {
            Rgb([20, 30, 40])
        }
    });
    let mask = GrayImage::from_fn(w, h, |x, y| Luma([if x + y > 18 { 255 } else { 0 }]));
    img.save(im_dir.join(format!("{stem}.png"))).unwrap();
    mask.save(gt_dir.join(format!("{stem}.png"))).unwrap();
}

fn split_dirs(root: &Path, task: &str, split: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    let split_dir = root.join(task).join(split);
    let (im, gt) = (split_dir.join("im"), split_dir.join("gt"));
    std::fs::create_dir_all(&im).unwrap();
    std::fs::create_dir_all(&gt).unwrap();
    (im, gt)
}

fn item(width: u32, height: u32, value: u8) -> BiRefNetItem {
    BiRefNetItem {
        image: RgbImage::from_pixel(width, height, Rgb([value; 3])),
        mask: GrayImage::from_pixel(width, height, Luma([255])),
    }
}

#[test]
fn dataset_discovers_sorted_pairs_and_keeps_source_sizes() {
    let dir = TempDir::new().unwrap();
    let (im_dir, gt_dir) = split_dirs(dir.path(), "DIS5K", "DIS-TR");
    write_sample_pair(&im_dir, &gt_dir, "b_second", (24, 20));
    write_sample_pair(&im_dir, &gt_dir, "a_first", (30, 30));
    // An image without a mask is skipped; a non-image file is ignored.
    RgbImage::from_pixel(16, 16, Rgb([1, 2, 3]))
        .save(im_dir.join("unpaired.png"))
        .unwrap();
    std::fs::write(im_dir.join("notes.txt"), "not an image").unwrap();

    let dataset = BiRefNetDataset::new(dir.path(), Task::DIS5K, &DatasetSplit::Train).unwrap();
    assert_eq!(dataset.len(), 2);
    assert!(dataset.pairs()[0].0.ends_with("a_first.png"));

    let first = dataset.get(0).unwrap();
    assert_eq!(first.image.dimensions(), (30, 30));
    assert_eq!(dataset.get(1).unwrap().mask.dimensions(), (24, 20));
    assert!(dataset.get(2).unwrap_err().is_index_out_of_bounds());
}

#[test]
fn dataset_reports_missing_empty_and_mismatched_inputs() {
    let dir = TempDir::new().unwrap();
    let missing = BiRefNetDataset::new(dir.path(), Task::General, &DatasetSplit::Val).unwrap_err();
    assert!(missing.is_not_found());
    assert!(missing.path().unwrap().ends_with("im"));

    let (im_dir, gt_dir) = split_dirs(dir.path(), "General", "DIS-VD");
    let empty = BiRefNetDataset::new(dir.path(), Task::General, &DatasetSplit::Val).unwrap_err();
    assert!(empty.is_empty_dataset());

    RgbImage::new(10, 10).save(im_dir.join("x.png")).unwrap();
    GrayImage::new(10, 12).save(gt_dir.join("x.png")).unwrap();
    let dataset = BiRefNetDataset::new(dir.path(), Task::General, &DatasetSplit::Val).unwrap();
    let err = dataset.get(0).unwrap_err();
    assert!(err.is_dimension_mismatch(), "{err}");
}

#[test]
fn fixed_resolution_batches_resize_and_normalize() {
    let device = Device::flex();
    let batcher = BiRefNetBatcher::new(Resolution::fixed(16, 8).unwrap());
    let batch = batcher.batch(vec![item(32, 32, 0), item(20, 10, 255)], &device);
    assert_eq!(batch.images.dims(), [2, 3, 8, 16]);
    assert_eq!(batch.masks.dims(), [2, 1, 8, 16]);

    // Black and white pixels map to (0 - mean) / std and (1 - mean) / std on channel 0.
    let red: Vec<f32> = batch
        .images
        .slice(s![.., 0..1, 0..1, 0..1])
        .into_data()
        .try_to_vec()
        .unwrap();
    assert!((red[0] - (-0.485 / 0.229)).abs() < 1e-4);
    assert!((red[1] - (0.515 / 0.229)).abs() < 1e-4);
    assert!(Resolution::fixed(0, 8).unwrap_err().is_invalid_config());
}

#[test]
fn dynamic_resolution_resizes_whole_batches_of_mixed_sizes() {
    let device = Device::flex();
    let dynamic = DynamicSize::new(64..=256, 96..=128).unwrap();
    let batcher = BiRefNetBatcher::with_seed(Resolution::Dynamic(dynamic), 7);

    let mut seen = HashSet::new();
    for _ in 0..12 {
        // Items of different source sizes in one batch (previously a stack panic).
        let batch = batcher.batch(vec![item(50, 40, 10), item(300, 120, 20)], &device);
        let [b, _, h, w] = batch.images.dims();
        assert_eq!(b, 2);
        assert_eq!(batch.masks.dims(), [2, 1, h, w]);
        assert!((64..=256).contains(&w) && w % 32 == 0, "width {w}");
        assert!((96..=128).contains(&h) && h % 32 == 0, "height {h}");
        seen.insert((w, h));
    }
    assert!(seen.len() > 1, "dynamic sizes never changed: {seen:?}");

    // Sizes are aligned up into the range, never below its minimum.
    let unaligned = DynamicSize::square(250..=300).unwrap();
    let batch = BiRefNetBatcher::with_seed(Resolution::Dynamic(unaligned), 1)
        .batch(vec![item(10, 10, 0)], &device);
    assert_eq!(batch.images.dims()[2..], [256, 256]);

    assert!(
        DynamicSize::square(33..=63)
            .unwrap_err()
            .is_invalid_config()
    );
    assert!(
        DynamicSize::new(64..64, 64..=64)
            .unwrap_err()
            .is_invalid_config()
    );
}

#[test]
fn seeded_augmentor_reproduces_and_advances_across_calls() {
    let config = AugmentationConfig::new()
        .with_methods([
            AugmentationMethod::Flip,
            AugmentationMethod::Crop,
            AugmentationMethod::Rotate,
            AugmentationMethod::Enhance,
            AugmentationMethod::Pepper,
        ])
        .with_background_synthesis(true);

    let image = RgbImage::from_fn(40, 40, |x, y| {
        Rgb([
            u8::try_from(x * 6).unwrap(),
            u8::try_from(y * 6).unwrap(),
            140,
        ])
    });
    let mask = GrayImage::from_fn(40, 40, |x, y| {
        Luma([if x > 10 && y > 10 { 255 } else { 0 }])
    });

    let primary = ImageAugmentor::with_seed(config.clone(), 12345);
    let replay = ImageAugmentor::with_seed(config, 12345);
    let first = primary.augment(image.clone(), mask.clone());
    assert_eq!(first, replay.augment(image.clone(), mask.clone()));
    assert_eq!(first.0.dimensions(), first.1.dimensions());

    let outputs: HashSet<_> = (0..4)
        .map(|_| primary.augment(image.clone(), mask.clone()).0.into_raw())
        .collect();
    assert!(
        outputs.len() > 1,
        "sequential calls must advance the generator"
    );
}

#[test]
fn flip_only_augmentation_mirrors_image_and_mask_together() {
    let config = AugmentationConfig::new().with_methods([AugmentationMethod::Flip]);
    let image = RgbImage::from_fn(6, 2, |x, _| Rgb([u8::try_from(x).unwrap(); 3]));
    let mask = GrayImage::from_fn(6, 2, |x, _| Luma([u8::try_from(x * 10).unwrap()]));
    let augmentor = ImageAugmentor::with_seed(config, 0);
    for _ in 0..8 {
        let (out_image, out_mask) = augmentor.augment(image.clone(), mask.clone());
        let flipped = out_image.get_pixel(0, 0).0[0] == 5;
        assert_eq!(out_mask.get_pixel(0, 0).0[0] == 50, flipped);
        if !flipped {
            assert_eq!((out_image, out_mask), (image.clone(), mask.clone()));
        }
    }
}
