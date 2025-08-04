//! Improved Image processing utilities with better error handling and performance

use anyhow::{Context, Result};
use burn::tensor::{backend::Backend, DType, Tensor, TensorData};
use image::{
    imageops::FilterType, ColorType, DynamicImage, GenericImageView, ImageBuffer, Luma, Rgb, Rgba,
};
use std::path::Path;

/// Improved image processing utilities
pub struct ImageUtils;

impl ImageUtils {
    /// Load image from file and convert to tensor with improved error handling
    ///
    /// # Arguments
    /// * `path` - Path to the image file
    /// * `device` - Device to create tensor on
    ///
    /// # Returns
    /// Tensor of shape [1, 3, height, width] with values in range [0, 1]
    pub fn load_image<B: Backend, P: AsRef<Path>>(
        path: P,
        device: &B::Device,
    ) -> Result<Tensor<B, 4>> {
        let img = image::open(&path)
            .with_context(|| format!("Failed to open image at {}", path.as_ref().display()))?;

        Self::dynamic_image_to_tensor(img, device)
    }

    /// Convert DynamicImage to tensor - separated for reusability
    pub fn dynamic_image_to_tensor<B: Backend>(
        img: DynamicImage,
        device: &B::Device,
    ) -> Result<Tensor<B, 4>> {
        let (width, height) = img.dimensions();

        // Use more efficient conversion based on image format
        let tensor = match img.color() {
            ColorType::Rgb8 => {
                let rgb_img = img.into_rgb8();
                let buf: Vec<f32> = rgb_img
                    .into_raw()
                    .into_iter()
                    .map(|byte| byte as f32 / 255.0)
                    .collect();

                let data = TensorData::new(buf, [height as usize, width as usize, 3])
                    .convert::<B::FloatElem>();
                Tensor::from_data(data, device)
            }
            ColorType::Rgba8 => {
                let rgba_img = img.into_rgba8();
                let buf: Vec<f32> = rgba_img
                    .into_raw()
                    .into_iter()
                    .map(|byte| byte as f32 / 255.0)
                    .collect();

                let data = TensorData::new(buf, [height as usize, width as usize, 4])
                    .convert::<B::FloatElem>();
                let tensor = Tensor::from_data(data, device);
                // Extract RGB channels only
                tensor.slice([0..height as usize, 0..width as usize, 0..3])
            }
            _ => {
                // Convert to RGB32F for other formats
                let rgb_img = img.into_rgb32f();
                let buf = rgb_img.into_raw();

                let data = TensorData::new(buf, [height as usize, width as usize, 3])
                    .convert::<B::FloatElem>();
                Tensor::from_data(data, device)
            }
        };

        // Permute to [channels, height, width] and add batch dimension
        Ok(tensor.permute([2, 0, 1]).unsqueeze::<4>())
    }

    /// Convert tensor to DynamicImage with improved type safety
    ///
    /// # Arguments
    /// * `tensor` - Tensor of shape [batch, channels, height, width]
    /// * `is_mask` - Whether the tensor is a mask (single channel)
    pub fn tensor_to_dynamic_image<B: Backend>(
        tensor: Tensor<B, 4>,
        is_mask: bool,
    ) -> Result<DynamicImage> {
        let [batch, channels, height, width] = tensor.dims();

        if batch != 1 {
            anyhow::bail!("Expected batch size of 1, got {}", batch);
        }

        // Validate channel count early
        match (is_mask, channels) {
            (true, 1) | (false, 3) | (false, 4) => {}
            _ => anyhow::bail!(
                "Unsupported image format: is_mask={}, channels={}. \
                Expected: (true, 1), (false, 3), or (false, 4)",
                is_mask,
                channels
            ),
        }

        // Remove batch dimension and permute to HWC
        let tensor = tensor.squeeze::<3>(0).permute([1, 2, 0]);

        // Convert to f32 with better error handling
        let data = tensor
            .into_data()
            .convert_dtype(DType::F32)
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("Failed to convert tensor to f32: {:#?}", e))?;

        // Clamp values to valid range [0, 1] and convert to u8
        let data_u8: Vec<u8> = data
            .into_iter()
            .map(|val| (val.clamp(0.0, 1.0) * 255.0) as u8)
            .collect();

        // Create ImageBuffer directly with u8 data for better performance
        let img = match (is_mask, channels) {
            (true, 1) => {
                let img_buffer =
                    ImageBuffer::<Luma<u8>, _>::from_raw(width as u32, height as u32, data_u8)
                        .context("Failed to create grayscale image buffer")?;
                DynamicImage::ImageLuma8(img_buffer)
            }
            (false, 3) => {
                let img_buffer =
                    ImageBuffer::<Rgb<u8>, _>::from_raw(width as u32, height as u32, data_u8)
                        .context("Failed to create RGB image buffer")?;
                DynamicImage::ImageRgb8(img_buffer)
            }
            (false, 4) => {
                let img_buffer =
                    ImageBuffer::<Rgba<u8>, _>::from_raw(width as u32, height as u32, data_u8)
                        .context("Failed to create RGBA image buffer")?;
                DynamicImage::ImageRgba8(img_buffer)
            }
            _ => unreachable!("Already validated above"),
        };

        Ok(img)
    }

    /// Save tensor as image file (backward compatibility)
    ///
    /// # Arguments
    /// * `tensor` - Tensor of shape [batch, channels, height, width] or [channels, height, width]
    /// * `is_mask` - Whether the tensor is a mask (single channel)
    pub fn to_image<B: Backend>(tensor: Tensor<B, 4>, is_mask: bool) -> Result<DynamicImage> {
        Self::tensor_to_dynamic_image(tensor, is_mask)
    }

    /// Apply segmentation mask to image with improved validation
    ///
    /// # Arguments
    /// * `image` - Original image tensor [batch, 3, height, width]
    /// * `mask` - Segmentation mask tensor [batch, 1, height, width]
    ///
    /// # Returns
    /// RGBA image tensor [batch, 4, height, width] with alpha channel from mask
    pub fn apply_mask<B: Backend>(image: Tensor<B, 4>, mask: Tensor<B, 4>) -> Result<Tensor<B, 4>> {
        let image_dims = image.dims();
        let mask_dims = mask.dims();
        let [batch, channels, height, width] = image_dims;
        let [mask_batch, mask_channels, mask_height, mask_width] = mask_dims;

        // Comprehensive dimension validation
        if batch != mask_batch {
            anyhow::bail!("Batch size mismatch: image={}, mask={}", batch, mask_batch);
        }
        if height != mask_height || width != mask_width {
            anyhow::bail!(
                "Spatial dimensions mismatch: image={}x{}, mask={}x{}",
                height,
                width,
                mask_height,
                mask_width
            );
        }
        if channels != 3 {
            anyhow::bail!("Expected image with 3 channels, got {}", channels);
        }
        if mask_channels != 1 {
            anyhow::bail!("Expected mask with 1 channel, got {}", mask_channels);
        }

        // Concatenate image and mask to create RGBA
        Ok(Tensor::cat(vec![image, mask], 1))
    }

    /// Resize image with improved error handling and performance
    ///
    /// # Arguments
    /// * `path` - Path to input image
    /// * `target_size` - Target size (width, height)
    /// * `filter` - Resize filter type
    /// * `device` - Device to create tensor on
    ///
    /// # Returns
    /// Resized tensor of shape [1, 3, height, width]
    pub fn resize_image_file<B: Backend, P: AsRef<Path>>(
        path: P,
        target_size: (u32, u32),
        filter: FilterType,
        device: &B::Device,
    ) -> Result<Tensor<B, 4>> {
        let img = image::open(&path)
            .with_context(|| format!("Failed to open image at {}", path.as_ref().display()))?;

        let resized = img.resize_exact(target_size.0, target_size.1, filter);
        Self::dynamic_image_to_tensor(resized, device)
    }

    /// Create tensor from raw pixel data with validation
    ///
    /// # Arguments
    /// * `data` - Raw pixel data (RGB or RGBA)
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `channels` - Number of channels (3 for RGB, 4 for RGBA)
    /// * `device` - Device to create tensor on
    ///
    /// # Returns
    /// Tensor of shape [1, channels, height, width]
    pub fn from_raw_pixels<B: Backend>(
        data: Vec<u8>,
        width: u32,
        height: u32,
        channels: usize,
        device: &B::Device,
    ) -> Result<Tensor<B, 4>> {
        let expected_len = (width * height) as usize * channels;
        if data.len() != expected_len {
            anyhow::bail!(
                "Data length mismatch: expected {}, got {}",
                expected_len,
                data.len()
            );
        }

        if !matches!(channels, 1 | 3 | 4) {
            anyhow::bail!("Unsupported channel count: {}", channels);
        }

        // Convert u8 to f32 and normalize to [0,1]
        let normalized_data: Vec<f32> = data.into_iter().map(|byte| byte as f32 / 255.0).collect();

        let tensor_data =
            TensorData::new(normalized_data, [height as usize, width as usize, channels])
                .convert::<B::FloatElem>();

        let tensor = Tensor::from_data(tensor_data, device);
        Ok(tensor.permute([2, 0, 1]).unsqueeze::<4>())
    }

    /// Batch process multiple images efficiently
    ///
    /// # Arguments
    /// * `paths` - Vector of image paths
    /// * `device` - Device to create tensors on
    ///
    /// # Returns
    /// Tensor of shape [batch_size, 3, height, width]
    pub fn load_image_batch<B: Backend, P: AsRef<Path>>(
        paths: Vec<P>,
        device: &B::Device,
    ) -> Result<Tensor<B, 4>> {
        if paths.is_empty() {
            anyhow::bail!("Empty path list provided");
        }

        let mut tensors = Vec::with_capacity(paths.len());
        let mut expected_dims: Option<[usize; 2]> = None;

        for (i, path) in paths.iter().enumerate() {
            let tensor = Self::load_image(path, device).with_context(|| {
                format!("Failed to load image {} at {}", i, path.as_ref().display())
            })?;

            let [_, _channels, height, width] = tensor.dims();

            // Validate that all images have the same dimensions
            let current_dims = [height, width];
            match expected_dims {
                None => expected_dims = Some(current_dims),
                Some(expected) => {
                    if current_dims != expected {
                        anyhow::bail!(
                            "Image {} has dimensions {}x{}, expected {}x{}",
                            i,
                            height,
                            width,
                            expected[0],
                            expected[1]
                        );
                    }
                }
            }

            tensors.push(tensor.squeeze::<3>(0)); // Remove individual batch dimension
        }

        // Concatenate along batch dimension
        Ok(Tensor::stack(tensors, 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn from_raw_pixels_invalid_data_returns_error() {
        let device = Default::default();

        // Test invalid data length
        let result = ImageUtils::from_raw_pixels::<TestBackend>(
            vec![255u8; 10], // Invalid length
            2,
            2,
            3, // 2x2 RGB should need 12 bytes
            &device,
        );
        assert!(result.is_err());

        // Test invalid channel count
        let result = ImageUtils::from_raw_pixels::<TestBackend>(
            vec![255u8; 8],
            2,
            2,
            2, // Invalid channel count
            &device,
        );
        assert!(result.is_err());
    }

    #[test]
    fn apply_mask_mismatched_dimensions_returns_error() {
        let device = Default::default();

        // Create test tensors with mismatched dimensions
        let image = Tensor::<TestBackend, 4>::zeros([1, 3, 10, 10], &device);
        let mask = Tensor::<TestBackend, 4>::zeros([1, 1, 5, 5], &device); // Wrong size

        let result = ImageUtils::apply_mask(image, mask);
        assert!(result.is_err());
    }
}
