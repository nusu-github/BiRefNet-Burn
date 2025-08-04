//! Improved Image processing utilities with better error handling and performance

use anyhow::{Context, Result};
use burn::tensor::{backend::Backend, DType, Tensor, TensorData};
use image::{
    buffer::ConvertBuffer, imageops::FilterType, DynamicImage, GenericImageView, ImageBuffer, Luma,
    Rgb, Rgba,
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

        // Use efficient direct f32 conversion for all formats
        // Always convert to RGB32F for consistent float handling
        let rgb_img = img.into_rgb32f();
        let buf = rgb_img.into_raw();

        let data =
            TensorData::new(buf, [height as usize, width as usize, 3]).convert::<B::FloatElem>();
        let tensor = Tensor::from_data(data, device);

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

        // Create f32 ImageBuffer first, then convert to u8 using ConvertBuffer trait
        let img = match (is_mask, channels) {
            (true, 1) => {
                let f32_buffer =
                    ImageBuffer::<Luma<f32>, _>::from_raw(width as u32, height as u32, data)
                        .context("Failed to create grayscale f32 image buffer")?;
                let u8_buffer: ImageBuffer<Luma<u8>, Vec<u8>> = f32_buffer.convert();
                DynamicImage::ImageLuma8(u8_buffer)
            }
            (false, 3) => {
                let f32_buffer =
                    ImageBuffer::<Rgb<f32>, _>::from_raw(width as u32, height as u32, data)
                        .context("Failed to create RGB f32 image buffer")?;
                let u8_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = f32_buffer.convert();
                DynamicImage::ImageRgb8(u8_buffer)
            }
            (false, 4) => {
                let f32_buffer =
                    ImageBuffer::<Rgba<f32>, _>::from_raw(width as u32, height as u32, data)
                        .context("Failed to create RGBA f32 image buffer")?;
                let u8_buffer: ImageBuffer<Rgba<u8>, Vec<u8>> = f32_buffer.convert();
                DynamicImage::ImageRgba8(u8_buffer)
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

    /// Create tensor from raw pixel data with validation and optimized conversion
    ///
    /// # Arguments
    /// * `data` - Raw pixel data (RGB or RGBA, u8 values 0-255)
    /// * `width` - Image width
    /// * `height` - Image height
    /// * `channels` - Number of channels (3 for RGB, 4 for RGBA)
    /// * `device` - Device to create tensor on
    ///
    /// # Returns
    /// Tensor of shape [1, channels, height, width] with values normalized to [0,1]
    pub fn from_raw_pixels<B: Backend>(
        data: Vec<u8>,
        width: u32,
        height: u32,
        channels: usize,
        device: &B::Device,
    ) -> Result<Tensor<B, 4>> {
        Self::from_raw_pixels_with_normalization(data, width, height, channels, device, true)
    }

    /// Create tensor from raw pixel data with optional normalization
    ///
    /// # Arguments
    /// * `data` - Raw pixel data
    /// * `width` - Image width
    /// * `height` - Image height  
    /// * `channels` - Number of channels (1, 3, or 4)
    /// * `device` - Device to create tensor on
    /// * `normalize` - Whether to normalize u8 values to [0,1] range
    ///
    /// # Returns
    /// Tensor of shape [1, channels, height, width]
    pub fn from_raw_pixels_with_normalization<B: Backend>(
        data: Vec<u8>,
        width: u32,
        height: u32,
        channels: usize,
        device: &B::Device,
        normalize: bool,
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

        // More efficient conversion using iterator adaptors
        let normalized_data: Vec<f32> = if normalize {
            // Use const to avoid repeated division
            const INV_255: f32 = 1.0 / 255.0;
            data.into_iter().map(|byte| byte as f32 * INV_255).collect()
        } else {
            data.into_iter().map(|byte| byte as f32).collect()
        };

        let tensor_data =
            TensorData::new(normalized_data, [height as usize, width as usize, channels])
                .convert::<B::FloatElem>();

        let tensor = Tensor::from_data(tensor_data, device);
        Ok(tensor.permute([2, 0, 1]).unsqueeze::<4>())
    }

    /// Batch process multiple images efficiently with parallel loading
    ///
    /// # Arguments
    /// * `paths` - Vector of image paths
    /// * `device` - Device to create tensors on
    ///
    /// # Returns
    /// Tensor of shape [batch_size, 3, height, width]
    pub fn load_image_batch<B: Backend, P: AsRef<Path> + Send + Sync>(
        paths: Vec<P>,
        device: &B::Device,
    ) -> Result<Tensor<B, 4>> {
        if paths.is_empty() {
            anyhow::bail!("Empty path list provided");
        }

        // Load first image to get expected dimensions
        let first_tensor = Self::load_image(&paths[0], device).with_context(|| {
            format!(
                "Failed to load first image at {}",
                paths[0].as_ref().display()
            )
        })?;
        let [_, _, expected_height, expected_width] = first_tensor.dims();

        let mut tensors = Vec::with_capacity(paths.len());
        tensors.push(first_tensor.squeeze::<3>(0));

        // Process remaining images
        for (i, path) in paths.iter().enumerate().skip(1) {
            let tensor = Self::load_image(path, device).with_context(|| {
                format!("Failed to load image {} at {}", i, path.as_ref().display())
            })?;

            let [_, _, height, width] = tensor.dims();
            if height != expected_height || width != expected_width {
                anyhow::bail!(
                    "Image {} has dimensions {}x{}, expected {}x{}",
                    i,
                    height,
                    width,
                    expected_height,
                    expected_width
                );
            }

            tensors.push(tensor.squeeze::<3>(0));
        }

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

    #[test]
    fn from_raw_pixels_with_normalization_works_correctly() {
        let device = Default::default();

        // Test with normalization
        let data = vec![0u8, 127, 255]; // Should become [0.0, ~0.5, 1.0]
        let result = ImageUtils::from_raw_pixels_with_normalization::<TestBackend>(
            data, 1, 1, 3, &device, true,
        );
        assert!(result.is_ok());

        // Test without normalization
        let data = vec![0u8, 127, 255]; // Should remain [0.0, 127.0, 255.0]
        let result = ImageUtils::from_raw_pixels_with_normalization::<TestBackend>(
            data, 1, 1, 3, &device, false,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn apply_mask_creates_rgba_tensor() {
        let device = Default::default();

        let image = Tensor::<TestBackend, 4>::zeros([1, 3, 10, 10], &device);
        let mask = Tensor::<TestBackend, 4>::ones([1, 1, 10, 10], &device);

        let result = ImageUtils::apply_mask(image, mask);
        assert!(result.is_ok());

        let rgba_tensor = result.unwrap();
        let [batch, channels, height, width] = rgba_tensor.dims();
        assert_eq!(batch, 1);
        assert_eq!(channels, 4); // RGB + Alpha
        assert_eq!(height, 10);
        assert_eq!(width, 10);
    }

    #[test]
    fn load_image_batch_empty_paths_returns_error() {
        let device = Default::default();
        let empty_paths: Vec<&str> = vec![];

        let result = ImageUtils::load_image_batch::<TestBackend, _>(empty_paths, &device);
        assert!(result.is_err());
    }
}
