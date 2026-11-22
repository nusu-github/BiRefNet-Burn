//! Image decoding, encoding, and tensor conversion (`image-io` feature).
//!
//! Decoded images become `[1, 3, H, W]` RGB tensors in `[0, 1]`; formats are detected from the
//! stream's magic bytes, never from the file extension.

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

use crate::error::UtilError;

fn decode_file(path: &Path) -> Result<DynamicImage, UtilError> {
    let file = File::open(path).map_err(|err| UtilError::io(Some(path.to_path_buf()), err))?;
    ImageReader::new(BufReader::new(file))
        .with_guessed_format()
        .map_err(|err| UtilError::io(Some(path.to_path_buf()), err))?
        .decode()
        .map_err(|err| UtilError::image(Some(path.to_path_buf()), err))
}

/// Loads and decodes an image file into `[1, 3, H, W]` with values in `[0, 1]`.
///
/// # Errors
///
/// Returns [`UtilError`] if opening (see [`UtilError::is_io`]) or decoding
/// ([`UtilError::is_image`]) `path` fails.
pub fn load_image(path: impl AsRef<Path>, device: &Device) -> Result<Tensor<4>, UtilError> {
    Ok(dynamic_image_to_tensor(decode_file(path.as_ref())?, device))
}

/// Decodes an image from a seekable buffered reader into `[1, 3, H, W]` in `[0, 1]`.
///
/// If `format` is `None`, the format is guessed from the stream's magic bytes.
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
    Ok(dynamic_image_to_tensor(image_reader.decode()?, device))
}

/// Decodes encoded image bytes into `[1, 3, H, W]` in `[0, 1]`.
///
/// # Errors
///
/// Returns [`UtilError`] if format detection or decoding fails.
pub fn decode_image_bytes(bytes: &[u8], device: &Device) -> Result<Tensor<4>, UtilError> {
    load_image_from_reader(Cursor::new(bytes), None, device)
}

/// Loads an image file and resizes it to `(width, height)` before conversion to `[1, 3, H, W]`.
///
/// # Errors
///
/// Returns [`UtilError`] if opening or decoding `path` fails.
pub fn resize_image_file(
    path: impl AsRef<Path>,
    target_size: (u32, u32),
    filter: FilterType,
    device: &Device,
) -> Result<Tensor<4>, UtilError> {
    let resized = decode_file(path.as_ref())?.resize_exact(target_size.0, target_size.1, filter);
    Ok(dynamic_image_to_tensor(resized, device))
}

/// Converts a [`DynamicImage`] into an RGB float tensor `[1, 3, H, W]` in `[0, 1]`.
#[must_use]
pub fn dynamic_image_to_tensor(img: DynamicImage, device: &Device) -> Tensor<4> {
    let (width, height) = img.dimensions();
    let raw = img.into_rgb32f().into_raw();
    let data = TensorData::new(raw, [height as usize, width as usize, 3]);
    Tensor::from_data(data, device)
        .permute([2, 0, 1])
        .unsqueeze::<4>()
}

/// Converts a `[1, C, H, W]` tensor in `[0, 1]` into an 8-bit [`DynamicImage`].
///
/// `C` must be `1` when `is_mask` is set (grayscale) and `3` or `4` otherwise (RGB / RGBA).
///
/// # Errors
///
/// Returns [`UtilError`] if `batch != 1`, the channel count disagrees with `is_mask`, or the
/// tensor cannot be read back from its device.
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
    let dims_u32 = |value: usize, name: &str| {
        u32::try_from(value).map_err(|_overflow| {
            UtilError::invalid_dimensions(format!("image {name} {value} exceeds u32::MAX"))
        })
    };
    let (width_u32, height_u32) = (dims_u32(width, "width")?, dims_u32(height, "height")?);
    if !matches!((is_mask, channels), (true, 1) | (false, 3 | 4)) {
        return Err(UtilError::invalid_input(format!(
            "unsupported channel count {channels} for is_mask={is_mask}"
        )));
    }

    let pixels = tensor
        .squeeze_dim::<3>(0)
        .permute([1, 2, 0])
        .try_into_vec_as::<f32>()?;
    let size_mismatch = || UtilError::invalid_dimensions("pixel buffer does not match image size");

    Ok(match channels {
        1 => DynamicImage::ImageLuma8(
            ImageBuffer::<Luma<f32>, _>::from_raw(width_u32, height_u32, pixels)
                .ok_or_else(size_mismatch)?
                .convert(),
        ),
        3 => DynamicImage::ImageRgb8(
            ImageBuffer::<Rgb<f32>, _>::from_raw(width_u32, height_u32, pixels)
                .ok_or_else(size_mismatch)?
                .convert(),
        ),
        _ => DynamicImage::ImageRgba8(
            ImageBuffer::<Rgba<f32>, _>::from_raw(width_u32, height_u32, pixels)
                .ok_or_else(size_mismatch)?
                .convert(),
        ),
    })
}

/// Converts interleaved 8-bit pixels `[H, W, C]` (`C` in `1 | 3 | 4`) to `[1, C, H, W]` in `[0, 1]`.
///
/// # Errors
///
/// Returns [`UtilError`] on dimension overflow, buffer length mismatch, or unsupported channel count.
pub fn from_raw_pixels(
    data: &[u8],
    (width, height): (u32, u32),
    channels: usize,
    device: &Device,
) -> Result<Tensor<4>, UtilError> {
    const INV_255: f32 = 1.0 / 255.0;

    if !matches!(channels, 1 | 3 | 4) {
        return Err(UtilError::invalid_input(format!(
            "unsupported channel count {channels}; expected 1, 3, or 4"
        )));
    }

    let width_usize = width as usize;
    let height_usize = height as usize;
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

    let floats: Vec<f32> = data.iter().map(|&byte| f32::from(byte) * INV_255).collect();
    let tensor_data = TensorData::new(floats, [height_usize, width_usize, channels]);
    Ok(Tensor::from_data(tensor_data, device)
        .permute([2, 0, 1])
        .unsqueeze::<4>())
}

/// Loads equally sized image files into a single `[B, 3, H, W]` batch in `[0, 1]`.
///
/// # Errors
///
/// Returns [`UtilError`] if `paths` is empty, any file fails to load, or sizes differ.
pub fn load_image_batch<P: AsRef<Path>>(
    paths: &[P],
    device: &Device,
) -> Result<Tensor<4>, UtilError> {
    let mut items = Vec::with_capacity(paths.len());
    let mut expected: Option<[usize; 2]> = None;
    for (idx, path) in paths.iter().enumerate() {
        let tensor = load_image(path, device)?;
        let [_, _, height, width] = tensor.dims();
        match expected {
            None => expected = Some([height, width]),
            Some([eh, ew]) if (eh, ew) != (height, width) => {
                return Err(UtilError::invalid_dimensions(format!(
                    "image[{idx}] has size {height}x{width}, expected {eh}x{ew}"
                )));
            }
            Some(_) => {}
        }
        items.push(tensor.squeeze_dim::<3>(0));
    }
    if items.is_empty() {
        return Err(UtilError::invalid_input(
            "cannot load batch from an empty path list",
        ));
    }
    Ok(Tensor::stack(items, 0))
}

/// Returns `true` if `extension` (with or without one leading `.`) is decodable in this build.
#[must_use]
pub fn is_extension_supported(extension: &str) -> bool {
    ImageFormat::from_extension(extension.trim_prefix('.'))
        .as_ref()
        .is_some_and(ImageFormat::reading_enabled)
}

/// Returns `true` if `path` has a file extension decodable in this build.
#[must_use]
pub fn is_supported_image_format(path: impl AsRef<Path>) -> bool {
    path.as_ref()
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(is_extension_supported)
}

/// Returns the file extensions of every image format decodable in this build.
#[must_use]
pub fn supported_extensions() -> Vec<&'static str> {
    ImageFormat::all()
        .filter(ImageFormat::reading_enabled)
        .flat_map(|fmt| fmt.extensions_str().iter().copied())
        .collect()
}

/// Returns every [`ImageFormat`] decodable in this build.
#[must_use]
pub fn supported_formats() -> Vec<ImageFormat> {
    ImageFormat::all()
        .filter(ImageFormat::reading_enabled)
        .collect()
}
