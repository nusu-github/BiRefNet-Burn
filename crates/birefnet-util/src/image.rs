//! Image file/byte I/O, decoding, encoding, and tensor conversion helpers.

use std::{
    fs::File,
    io::{BufRead, BufReader, Cursor, Seek},
    path::Path,
};

use burn::tensor::{Device, Tensor, TensorData};
use image::{
    DynamicImage, GenericImageView, ImageBuffer, ImageFormat, ImageReader, Luma, Rgb, Rgba,
    buffer::ConvertBuffer, imageops::FilterType,
};

use crate::{error::UtilError, normalize::apply_imagenet_normalization};

/// Loads and decodes an image file on disk into `[1, 3, H, W]` normalized to `[0.0, 1.0]`.
///
/// # Errors
///
/// Returns [`UtilError`] if opening or decoding `path` fails.
pub fn load_image(path: impl AsRef<Path>, device: &Device) -> Result<Tensor<4>, UtilError> {
    let path_ref = path.as_ref();
    let file =
        File::open(path_ref).map_err(|err| UtilError::io(Some(path_ref.to_path_buf()), err))?;
    let reader = BufReader::new(file);
    let guessed = ImageReader::new(reader)
        .with_guessed_format()
        .map_err(|err| UtilError::io(Some(path_ref.to_path_buf()), err))?;
    let img = guessed
        .decode()
        .map_err(|err| UtilError::image(Some(path_ref.to_path_buf()), err))?;
    dynamic_image_to_tensor(img, device)
}

/// Decodes an image from any seekable buffered reader into `[1, 3, H, W]` in `[0.0, 1.0]`.
///
/// If `format` is `None`, the format is guessed from stream magic bytes.
///
/// # Errors
///
/// Returns [`UtilError`] if reading or decoding the stream fails.
pub fn load_image_from_reader<R: BufRead + Seek>(
    reader: R,
    format: Option<ImageFormat>,
    device: &Device,
) -> Result<Tensor<4>, UtilError> {
    let image_reader = match format {
        Some(fmt) => ImageReader::with_format(reader, fmt),
        None => ImageReader::new(reader).with_guessed_format()?,
    };
    let img = image_reader.decode()?;
    dynamic_image_to_tensor(img, device)
}

/// Decodes encoded image bytes in memory into `[1, 3, H, W]` in `[0.0, 1.0]`.
///
/// # Errors
///
/// Returns [`UtilError`] if format guessing or decoding fails.
pub fn decode_image_bytes(bytes: &[u8], device: &Device) -> Result<Tensor<4>, UtilError> {
    load_image_from_reader(Cursor::new(bytes), None, device)
}

/// Converts a [`DynamicImage`] into an RGB float tensor `[1, 3, H, W]` with values in `[0.0, 1.0]`.
///
/// # Errors
///
/// Currently infallible; returns [`Result`] for symmetry with decoding APIs.
pub fn dynamic_image_to_tensor(img: DynamicImage, device: &Device) -> Result<Tensor<4>, UtilError> {
    let (width, height) = img.dimensions();
    let raw = img.into_rgb32f().into_raw();
    let data = TensorData::new(raw, [height as usize, width as usize, 3]);
    Ok(Tensor::from_data(data, device)
        .permute([2, 0, 1])
        .unsqueeze::<4>())
}

/// Converts a `[1, C, H, W]` tensor (`C = 1` if `is_mask`, else `C = 3 | 4`) into a [`DynamicImage`].
///
/// # Errors
///
/// Returns [`UtilError`] if `batch != 1`, channel count disagrees with `is_mask`, or host extraction fails.
pub fn tensor_to_dynamic_image(
    tensor: Tensor<4>,
    is_mask: bool,
) -> Result<DynamicImage, UtilError> {
    let [batch, channels, height, width] = tensor.dims();
    if batch != 1 {
        return Err(UtilError::invalid_dimensions(format!(
            "tensor_to_dynamic_image expects batch size 1, got {batch}"
        )));
    }
    if !matches!((is_mask, channels), (true, 1) | (false, 3 | 4)) {
        return Err(UtilError::invalid_input(format!(
            "unsupported channel count {channels} for is_mask={is_mask}"
        )));
    }

    let width_u32 = u32::try_from(width).map_err(|_err| {
        UtilError::invalid_dimensions(format!("image width {width} exceeds u32::MAX"))
    })?;
    let height_u32 = u32::try_from(height).map_err(|_err| {
        UtilError::invalid_dimensions(format!("image height {height} exceeds u32::MAX"))
    })?;

    let hwc = tensor.squeeze_dim::<3>(0).permute([1, 2, 0]);
    let data = hwc.into_data().try_into_vec_as::<f32>().map_err(|err| {
        UtilError::invalid_input(format!("failed to read tensor as f32: {err:?}"))
    })?;

    match (is_mask, channels) {
        (true, 1) => {
            let f32_buf = ImageBuffer::<Luma<f32>, _>::from_raw(width_u32, height_u32, data)
                .ok_or_else(|| {
                    UtilError::invalid_dimensions(
                        "failed to construct Luma<f32> buffer from tensor",
                    )
                })?;
            let u8_buf: ImageBuffer<Luma<u8>, Vec<u8>> = f32_buf.convert();
            Ok(DynamicImage::ImageLuma8(u8_buf))
        }
        (false, 3) => {
            let f32_buf = ImageBuffer::<Rgb<f32>, _>::from_raw(width_u32, height_u32, data)
                .ok_or_else(|| {
                    UtilError::invalid_dimensions("failed to construct Rgb<f32> buffer from tensor")
                })?;
            let u8_buf: ImageBuffer<Rgb<u8>, Vec<u8>> = f32_buf.convert();
            Ok(DynamicImage::ImageRgb8(u8_buf))
        }
        (false, 4) => {
            let f32_buf = ImageBuffer::<Rgba<f32>, _>::from_raw(width_u32, height_u32, data)
                .ok_or_else(|| {
                    UtilError::invalid_dimensions(
                        "failed to construct Rgba<f32> buffer from tensor",
                    )
                })?;
            let u8_buf: ImageBuffer<Rgba<u8>, Vec<u8>> = f32_buf.convert();
            Ok(DynamicImage::ImageRgba8(u8_buf))
        }
        _ => unreachable!("channel/mask combination already validated"),
    }
}

/// Loads an image file, resizes it to `(width, height)`, and converts to `[1, 3, H, W]` in `[0.0, 1.0]`.
///
/// # Errors
///
/// Returns [`UtilError`] if file reading or image decoding fails.
pub fn resize_image_file(
    path: impl AsRef<Path>,
    target_size: (u32, u32),
    filter: FilterType,
    device: &Device,
) -> Result<Tensor<4>, UtilError> {
    let path_ref = path.as_ref();
    let img =
        image::open(path_ref).map_err(|err| UtilError::image(Some(path_ref.to_path_buf()), err))?;
    let resized = img.resize_exact(target_size.0, target_size.1, filter);
    dynamic_image_to_tensor(resized, device)
}

/// Converts interleaved `u8` pixel data `[H, W, C]` (`C` in `1 | 3 | 4`) to `[1, C, H, W]` in `[0.0, 1.0]`.
///
/// # Errors
///
/// Returns [`UtilError`] on dimension overflow, buffer length mismatch, or unsupported channel count.
pub fn from_raw_pixels(
    data: Vec<u8>,
    width: u32,
    height: u32,
    channels: usize,
    device: &Device,
) -> Result<Tensor<4>, UtilError> {
    from_raw_pixels_with_normalization(data, width, height, channels, device, true)
}

/// Converts interleaved `u8` pixels `[H, W, C]` (`C` in `1 | 3 | 4`) to `[1, C, H, W]`, optionally dividing by `255.0`.
///
/// # Errors
///
/// Returns [`UtilError`] on dimension overflow, buffer length mismatch, or unsupported channel count.
pub fn from_raw_pixels_with_normalization(
    data: Vec<u8>,
    width: u32,
    height: u32,
    channels: usize,
    device: &Device,
    normalize: bool,
) -> Result<Tensor<4>, UtilError> {
    const INV_255: f32 = 1.0 / 255.0;

    if !matches!(channels, 1 | 3 | 4) {
        return Err(UtilError::invalid_input(format!(
            "unsupported channel count {channels}; expected 1, 3, or 4"
        )));
    }

    let width_usize = usize::try_from(width)
        .map_err(|_err| UtilError::invalid_dimensions("width exceeds addressable usize range"))?;
    let height_usize = usize::try_from(height)
        .map_err(|_err| UtilError::invalid_dimensions("height exceeds addressable usize range"))?;
    let expected_len = width_usize
        .checked_mul(height_usize)
        .and_then(|pixels| pixels.checked_mul(channels))
        .ok_or_else(|| {
            UtilError::invalid_dimensions(format!(
                "pixel buffer dimensions {width}x{height}x{channels} overflow usize"
            ))
        })?;

    if data.len() != expected_len {
        return Err(UtilError::invalid_dimensions(format!(
            "pixel buffer length {} does not match expected {expected_len}",
            data.len()
        )));
    }

    let floats: Vec<f32> = if normalize {
        data.into_iter()
            .map(|byte| f32::from(byte) * INV_255)
            .collect()
    } else {
        data.into_iter().map(f32::from).collect()
    };

    let tensor_data = TensorData::new(floats, [height_usize, width_usize, channels]);
    Ok(Tensor::from_data(tensor_data, device)
        .permute([2, 0, 1])
        .unsqueeze::<4>())
}

/// Loads all image paths into a single `[B, 3, H, W]` batch in `[0.0, 1.0]`.
///
/// # Errors
///
/// Returns [`UtilError`] if `paths` is empty, any file fails to load, or spatial sizes differ across images.
pub fn load_image_batch<P: AsRef<Path>>(
    paths: &[P],
    device: &Device,
) -> Result<Tensor<4>, UtilError> {
    if paths.is_empty() {
        return Err(UtilError::invalid_input(
            "cannot load batch from an empty path list",
        ));
    }

    let first = load_image(&paths[0], device)?;
    let [_, _, expected_h, expected_w] = first.dims();

    let mut items = Vec::with_capacity(paths.len());
    items.push(first.squeeze_dim::<3>(0));

    for (idx, path) in paths.iter().enumerate().skip(1) {
        let tensor = load_image(path, device)?;
        let [_, _, height, width] = tensor.dims();
        if height != expected_h || width != expected_w {
            return Err(UtilError::invalid_dimensions(format!(
                "image[{idx}] has size {height}x{width}, expected {expected_h}x{expected_w}"
            )));
        }
        items.push(tensor.squeeze_dim::<3>(0));
    }

    Ok(Tensor::stack(items, 0))
}

/// Loads an image file and applies ImageNet channel normalization `(x - mean) / std`.
///
/// # Errors
///
/// Returns [`UtilError`] if loading or decoding fails.
pub fn load_image_with_imagenet_normalization(
    path: impl AsRef<Path>,
    device: &Device,
) -> Result<Tensor<4>, UtilError> {
    apply_imagenet_normalization(load_image(path, device)?)
}

/// Converts a [`DynamicImage`] to `[1, 3, H, W]` and applies ImageNet channel normalization.
///
/// # Errors
///
/// Returns [`UtilError`] if conversion or normalization fails.
pub fn dynamic_image_to_tensor_with_imagenet_normalization(
    img: DynamicImage,
    device: &Device,
) -> Result<Tensor<4>, UtilError> {
    apply_imagenet_normalization(dynamic_image_to_tensor(img, device)?)
}

/// Loads a batch of equally sized images and applies ImageNet channel normalization.
///
/// # Errors
///
/// Returns [`UtilError`] if `paths` is empty, loading fails, or dimensions differ across the batch.
pub fn load_image_batch_with_imagenet_normalization<P: AsRef<Path>>(
    paths: &[P],
    device: &Device,
) -> Result<Tensor<4>, UtilError> {
    apply_imagenet_normalization(load_image_batch(paths, device)?)
}

/// Returns `true` if `extension` (with or without leading `.`) is enabled for decoding in this build.
#[must_use]
pub fn is_extension_supported(extension: &str) -> bool {
    let ext = extension.trim_start_matches('.');
    ImageFormat::from_extension(ext)
        .as_ref()
        .is_some_and(ImageFormat::reading_enabled)
}

/// Returns `true` if `path` has a file extension enabled for decoding in this build.
#[must_use]
pub fn is_supported_image_format(path: impl AsRef<Path>) -> bool {
    path.as_ref()
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(is_extension_supported)
}

/// Returns file extensions associated with image formats enabled for reading in this build.
#[must_use]
pub fn common_image_extensions() -> Vec<&'static str> {
    ImageFormat::all()
        .filter(ImageFormat::reading_enabled)
        .flat_map(|fmt| fmt.extensions_str().iter().copied())
        .collect()
}

/// Returns every [`ImageFormat`] enabled for reading in this build.
#[must_use]
pub fn supported_formats() -> Vec<ImageFormat> {
    ImageFormat::all()
        .filter(ImageFormat::reading_enabled)
        .collect()
}
