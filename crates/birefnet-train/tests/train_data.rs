//! Integration tests for dataset discovery, augmentations, and tensor batching.

use std::path::Path;

use birefnet_model::Task;
use birefnet_train::{
    AugmentationConfig, AugmentationMethod, BiRefNetBatcher, BiRefNetDataset, BiRefNetItem,
    DatasetSplit, ImageAugmentor,
};
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};
use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use tempfile::TempDir;

fn write_sample_pair(im_dir: &Path, gt_dir: &Path, stem: &str, rgb: [u8; 3]) {
    let img = RgbImage::from_fn(24, 24, |x, y| {
        if x + y > 18 {
            Rgb([rgb[0], rgb[1], rgb[2]])
        } else {
            Rgb([20, 30, 40])
        }
    });
    let mask = GrayImage::from_fn(24, 24, |x, y| Luma([if x + y > 18 { 255 } else { 0 }]));
    img.save(im_dir.join(format!("{stem}.png"))).unwrap();
    mask.save(gt_dir.join(format!("{stem}.png"))).unwrap();
}

#[test]
fn dataset_discovers_sorted_pairs_and_batches_items() {
    let dir = TempDir::new().unwrap();
    let split_dir = dir.path().join("DIS5K").join("DIS-TR");
    let im_dir = split_dir.join("im");
    let gt_dir = split_dir.join("gt");
    std::fs::create_dir_all(&im_dir).unwrap();
    std::fs::create_dir_all(&gt_dir).unwrap();

    write_sample_pair(&im_dir, &gt_dir, "b_second", [200, 100, 50]);
    write_sample_pair(&im_dir, &gt_dir, "a_first", [50, 150, 220]);
    // Image without a matching mask is skipped rather than failing the whole dataset.
    RgbImage::from_pixel(16, 16, Rgb([1, 2, 3]))
        .save(im_dir.join("unpaired.png"))
        .unwrap();

    let aug_cfg = AugmentationConfig {
        target_size: (32, 32),
        ..AugmentationConfig::default()
    };
    let dataset =
        BiRefNetDataset::with_augmentation(dir.path(), Task::DIS5K, &DatasetSplit::Train, aug_cfg)
            .unwrap();
    assert_eq!(dataset.len(), 2);

    let item0: BiRefNetItem = dataset.get(0).unwrap();
    let item1: BiRefNetItem = dataset.get(1).unwrap();
    assert_eq!((item0.height, item0.width), (32, 32));
    assert_eq!(item0.image.len(), 32 * 32 * 3);
    assert_eq!(item0.mask.len(), 32 * 32);

    let device = Device::flex();
    let batch = BiRefNetBatcher::new().batch(vec![item0, item1], &device);
    assert_eq!(batch.images.shape().dims(), [2, 3, 32, 32]);
    assert_eq!(batch.masks.shape().dims(), [2, 1, 32, 32]);
}

#[test]
fn dataset_reports_not_found_and_empty_directory_errors() {
    let dir = TempDir::new().unwrap();
    let missing = BiRefNetDataset::new(dir.path(), Task::General, &DatasetSplit::Val).unwrap_err();
    assert!(missing.is_not_found());

    let split_dir = dir.path().join("General").join("DIS-VD");
    std::fs::create_dir_all(split_dir.join("im")).unwrap();
    std::fs::create_dir_all(split_dir.join("gt")).unwrap();

    let empty = BiRefNetDataset::new(dir.path(), Task::General, &DatasetSplit::Val).unwrap_err();
    assert!(empty.is_empty_dataset());
}

#[test]
fn seeded_augmentor_reproduces_and_advances_across_calls() {
    let cfg = AugmentationConfig {
        enabled_methods: vec![
            AugmentationMethod::Flip,
            AugmentationMethod::Crop,
            AugmentationMethod::Rotate,
            AugmentationMethod::Enhance,
            AugmentationMethod::Pepper,
        ],
        background_color_synthesis: true,
        dynamic_size_config: None,
        target_size: (32, 32),
    };

    let image = DynamicImage::ImageRgb8(ImageBuffer::from_fn(40, 40, |x, y| {
        Rgb([((x * 6) % 255) as u8, ((y * 6) % 255) as u8, 140])
    }));
    let mask = DynamicImage::ImageLuma8(ImageBuffer::from_fn(40, 40, |x, y| {
        Luma([if x > 10 && y > 10 { 255 } else { 0 }])
    }));

    let primary_aug = ImageAugmentor::with_seed(cfg.clone(), 12345);
    let replay_aug = ImageAugmentor::with_seed(cfg, 12345);

    let (first_img, first_mask) = primary_aug.augment(image.clone(), mask.clone(), true);
    let (replay_img, replay_mask) = replay_aug.augment(image.clone(), mask.clone(), true);
    assert_eq!(first_img.to_rgb8().as_raw(), replay_img.to_rgb8().as_raw());
    assert_eq!(
        first_mask.to_luma8().as_raw(),
        replay_mask.to_luma8().as_raw()
    );

    let (next_img, _) = primary_aug.augment(image.clone(), mask.clone(), true);
    assert_ne!(
        first_img.to_rgb8().as_raw(),
        next_img.to_rgb8().as_raw(),
        "sequential augment calls should advance the RNG state instead of repeating"
    );

    // Validation mode always performs deterministic resize-only without random perturbations.
    let (val_img_one, val_mask_one) = primary_aug.augment(image.clone(), mask.clone(), false);
    let (val_img_two, val_mask_two) = primary_aug.augment(image, mask, false);
    assert_eq!(
        val_img_one.to_rgb8().as_raw(),
        val_img_two.to_rgb8().as_raw()
    );
    assert_eq!(
        val_mask_one.to_luma8().as_raw(),
        val_mask_two.to_luma8().as_raw()
    );
}
