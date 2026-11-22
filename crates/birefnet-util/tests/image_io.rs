//! Integration tests for image decoding/encoding APIs in `birefnet-util`.

#![cfg(feature = "image-io")]

use std::io::Cursor;

use approx::assert_abs_diff_eq;
use birefnet_util::image::{
    decode_image_bytes, dynamic_image_to_tensor, from_raw_pixels, is_extension_supported,
    is_supported_image_format, load_image, load_image_batch, load_image_from_reader,
    resize_image_file, supported_extensions, supported_formats, tensor_to_dynamic_image,
};
use burn::prelude::*;
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use tempfile::TempDir;

#[test]
fn round_trips_png_through_disk_reader_and_bytes() {
    let device = Device::flex();
    let tmp = TempDir::new().unwrap();
    let img_path = tmp.path().join("sample.png");

    let mut rgb = RgbImage::new(4, 3);
    rgb.put_pixel(1, 2, Rgb([255, 128, 0]));
    DynamicImage::ImageRgb8(rgb.clone())
        .save(&img_path)
        .unwrap();

    let from_disk = load_image(&img_path, &device).unwrap();
    assert_eq!(from_disk.dims(), [1, 3, 3, 4]);

    let bytes = std::fs::read(&img_path).unwrap();
    let from_bytes = decode_image_bytes(&bytes, &device).unwrap();
    let from_reader =
        load_image_from_reader(Cursor::new(&bytes), Some(ImageFormat::Png), &device).unwrap();

    assert_eq!(
        from_disk.clone().into_data().try_to_vec::<f32>().unwrap(),
        from_bytes.into_data().try_to_vec::<f32>().unwrap()
    );
    assert_eq!(
        from_disk.clone().into_data().try_to_vec::<f32>().unwrap(),
        from_reader.into_data().try_to_vec::<f32>().unwrap()
    );

    let reencoded = tensor_to_dynamic_image(from_disk, false).unwrap();
    let round_tripped = dynamic_image_to_tensor(reencoded, &device)
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
    assert_eq!(round_tripped.len(), 3 * 3 * 4);
    assert_abs_diff_eq!(round_tripped[2 * 4 + 1], 1.0, epsilon = 1e-2);
}

#[test]
fn format_support_matches_compiled_codecs() {
    assert!(is_extension_supported("png"));
    assert!(is_extension_supported(".jpg"));
    assert!(!is_extension_supported("exr"));
    assert!(!is_extension_supported("avif"));
    assert!(is_supported_image_format("photo.webp"));
    assert!(!is_supported_image_format("render.exr"));
    assert!(is_supported_image_format("anim.gif"));
    // Only one leading dot is stripped.
    assert!(!is_extension_supported("..png"));

    let formats = supported_formats();
    assert!(formats.contains(&ImageFormat::Png));
    assert!(!formats.contains(&ImageFormat::OpenExr));
    assert!(supported_extensions().contains(&"png"));
}

#[test]
fn raw_pixels_batch_and_overflow_guards() {
    let device = Device::flex();
    let raw = [255_u8, 0, 128, 0, 255, 128];
    let tensor = from_raw_pixels(&raw, (2, 1), 3, &device).unwrap();
    assert_eq!(tensor.dims(), [1, 3, 1, 2]);
    // Channel-first layout: channel 0 holds pixels 0 and 1.
    let vals = tensor.into_data().try_to_vec::<f32>().unwrap();
    assert_abs_diff_eq!(vals[0], 1.0);
    assert_abs_diff_eq!(vals[1], 0.0);

    let overflow_err = from_raw_pixels(&[], (u32::MAX, u32::MAX), 4, &device).unwrap_err();
    assert!(overflow_err.is_invalid_dimensions());

    // A missing file is an I/O error for both plain and resizing loaders.
    let tmp = TempDir::new().unwrap();
    let missing = tmp.path().join("missing.png");
    for err in [
        load_image(&missing, &device).unwrap_err(),
        resize_image_file(
            &missing,
            (2, 2),
            image::imageops::FilterType::Nearest,
            &device,
        )
        .unwrap_err(),
    ] {
        assert!(
            err.is_io() && err.is_not_found() && !err.is_image(),
            "{err}"
        );
        assert_eq!(err.path(), Some(missing.as_path()));
    }

    // A file with an image extension but garbage content is an image (codec) error.
    let garbage = tmp.path().join("garbage.png");
    std::fs::write(&garbage, b"not an image").unwrap();
    let err = load_image(&garbage, &device).unwrap_err();
    assert!(err.is_image() && !err.is_io(), "{err}");

    // Batch load + ImageNet norm on two matching images.
    let p1 = tmp.path().join("a.png");
    let p2 = tmp.path().join("b.png");
    DynamicImage::ImageRgb8(RgbImage::new(4, 4))
        .save(&p1)
        .unwrap();
    DynamicImage::ImageRgb8(RgbImage::new(4, 4))
        .save(&p2)
        .unwrap();

    let batch = load_image_batch(&[&p1, &p2], &device).unwrap();
    assert_eq!(batch.dims(), [2, 3, 4, 4]);

    let resized =
        resize_image_file(&p1, (2, 3), image::imageops::FilterType::Nearest, &device).unwrap();
    assert_eq!(resized.dims(), [1, 3, 3, 2]);

    let empty: [&std::path::Path; 0] = [];
    assert!(
        load_image_batch(&empty, &device)
            .unwrap_err()
            .is_invalid_input()
    );
}
