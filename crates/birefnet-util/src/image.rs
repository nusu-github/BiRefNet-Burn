//! Improved Image processing utilities with better error handling and performance

use std::path::Path;

use burn::tensor::{backend::Backend, DType, Tensor, TensorData};
use image::{
    buffer::ConvertBuffer, imageops::FilterType, DynamicImage, GenericImageView, ImageBuffer,
    ImageFormat, Luma, Rgb, Rgba,
};
use thiserror::Error;

/// ImageError covers all possible errors in image processing operations
#[derive(Debug, Error)]
pub enum ImageError {
    #[error("failed to open image at '{path}': {source}")]
    ImageLoadError {
        path: String,
        #[source]
        source: image::ImageError,
    },

    #[error("failed to convert tensor to data: {reason}")]
    TensorConversionError { reason: String },

    #[error("failed to create image buffer: {reason}")]
    BufferCreationError { reason: String },

    #[error("batch size mismatch: expected 1, got {actual}")]
    InvalidBatchSize { actual: usize },

    #[error("unsupported image format: is_mask={is_mask}, channels={channels}. Expected: (true, 1), (false, 3), or (false, 4)")]
    UnsupportedImageFormat { is_mask: bool, channels: usize },

    #[error(
        "dimension mismatch between image and mask: image={}x{}, mask={}x{}",
        image_height,
        image_width,
        mask_height,
        mask_width
    )]
    DimensionMismatch {
        image_height: usize,
        image_width: usize,
        mask_height: usize,
        mask_width: usize,
    },

    #[error("batch size mismatch between tensors: image={image_batch}, mask={mask_batch}")]
    BatchSizeMismatch {
        image_batch: usize,
        mask_batch: usize,
    },

    #[error("invalid channel count: expected 3 for image, got {actual}")]
    InvalidImageChannels { actual: usize },

    #[error("invalid channel count: expected 1 for mask, got {actual}")]
    InvalidMaskChannels { actual: usize },

    #[error("data length mismatch: expected {expected}, got {actual}")]
    DataLengthMismatch { expected: usize, actual: usize },

    #[error("unsupported channel count: {channels} (supported: 1, 3, 4)")]
    UnsupportedChannelCount { channels: usize },

    #[error("empty path list provided for batch processing")]
    EmptyPathList,

    #[error("image {index} has dimensions {actual_height}x{actual_width}, expected {expected_height}x{expected_width}")]
    InconsistentImageDimensions {
        index: usize,
        actual_height: usize,
        actual_width: usize,
        expected_height: usize,
        expected_width: usize,
    },

    #[error("failed to load image {index} at '{path}': {source}")]
    BatchLoadError {
        index: usize,
        path: String,
        #[source]
        source: Box<ImageError>,
    },
}

/// Result type alias for ImageError
pub type ImageResult<T> = Result<T, ImageError>;

/// Improved image processing utilities
pub struct ImageUtils;

impl ImageUtils {
    /// ImageNet normalization constants
    pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

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
    ) -> ImageResult<Tensor<B, 4>> {
        let path_str = path.as_ref().display().to_string();
        let img = image::open(&path).map_err(|source| ImageError::ImageLoadError {
            path: path_str,
            source,
        })?;

        Self::dynamic_image_to_tensor(img, device)
    }

    /// Convert DynamicImage to tensor - separated for reusability
    pub fn dynamic_image_to_tensor<B: Backend>(
        img: DynamicImage,
        device: &B::Device,
    ) -> ImageResult<Tensor<B, 4>> {
        let (width, height) = img.dimensions();

        // Use efficient direct f64 conversion for all formats
        // Always convert to RGB32F for consistent float handling
        let rgb_img = img.into_rgb32f();
        let buf = rgb_img.into_raw();

        let data = TensorData::new(buf, [height as usize, width as usize, 3]);
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
    ) -> ImageResult<DynamicImage> {
        let [batch, channels, height, width] = tensor.dims();

        if batch != 1 {
            return Err(ImageError::InvalidBatchSize { actual: batch });
        }

        // Validate channel count early
        match (is_mask, channels) {
            (true, 1) | (false, 3 | 4) => {}
            _ => {
                return Err(ImageError::UnsupportedImageFormat { is_mask, channels });
            }
        }

        // Remove batch dimension and permute to HWC
        let tensor = tensor.squeeze::<3>(0).permute([1, 2, 0]);

        // Convert to f64 with better error handling
        let data = tensor
            .into_data()
            .convert_dtype(DType::F32)
            .to_vec::<f32>()
            .map_err(|e| ImageError::TensorConversionError {
                reason: format!("{:#?}", e),
            })?;

        // Create f64 ImageBuffer first, then convert to u8 using ConvertBuffer trait
        let img = match (is_mask, channels) {
            (true, 1) => {
                let f64_buffer =
                    ImageBuffer::<Luma<f32>, _>::from_raw(width as u32, height as u32, data)
                        .ok_or_else(|| ImageError::BufferCreationError {
                            reason: "Failed to create grayscale f32 image buffer".to_string(),
                        })?;
                let u8_buffer: ImageBuffer<Luma<u8>, Vec<u8>> = f64_buffer.convert();
                DynamicImage::ImageLuma8(u8_buffer)
            }
            (false, 3) => {
                let f64_buffer =
                    ImageBuffer::<Rgb<f32>, _>::from_raw(width as u32, height as u32, data)
                        .ok_or_else(|| ImageError::BufferCreationError {
                            reason: "Failed to create RGB f32 image buffer".to_string(),
                        })?;
                let u8_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = f64_buffer.convert();
                DynamicImage::ImageRgb8(u8_buffer)
            }
            (false, 4) => {
                let f64_buffer =
                    ImageBuffer::<Rgba<f32>, _>::from_raw(width as u32, height as u32, data)
                        .ok_or_else(|| ImageError::BufferCreationError {
                            reason: "Failed to create RGBA f32 image buffer".to_string(),
                        })?;
                let u8_buffer: ImageBuffer<Rgba<u8>, Vec<u8>> = f64_buffer.convert();
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
    pub fn to_image<B: Backend>(tensor: Tensor<B, 4>, is_mask: bool) -> ImageResult<DynamicImage> {
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
    pub fn apply_mask<B: Backend>(
        image: Tensor<B, 4>,
        mask: Tensor<B, 4>,
    ) -> ImageResult<Tensor<B, 4>> {
        let image_dims = image.dims();
        let mask_dims = mask.dims();
        let [batch, channels, height, width] = image_dims;
        let [mask_batch, mask_channels, mask_height, mask_width] = mask_dims;

        // Comprehensive dimension validation
        if batch != mask_batch {
            return Err(ImageError::BatchSizeMismatch {
                image_batch: batch,
                mask_batch,
            });
        }
        if height != mask_height || width != mask_width {
            return Err(ImageError::DimensionMismatch {
                image_height: height,
                image_width: width,
                mask_height,
                mask_width,
            });
        }
        if channels != 3 {
            return Err(ImageError::InvalidImageChannels { actual: channels });
        }
        if mask_channels != 1 {
            return Err(ImageError::InvalidMaskChannels {
                actual: mask_channels,
            });
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
    ) -> ImageResult<Tensor<B, 4>> {
        let path_str = path.as_ref().display().to_string();
        let img = image::open(&path).map_err(|source| ImageError::ImageLoadError {
            path: path_str,
            source,
        })?;

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
    ) -> ImageResult<Tensor<B, 4>> {
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
    ) -> ImageResult<Tensor<B, 4>> {
        let expected_len = (width * height) as usize * channels;
        if data.len() != expected_len {
            return Err(ImageError::DataLengthMismatch {
                expected: expected_len,
                actual: data.len(),
            });
        }

        if !matches!(channels, 1 | 3 | 4) {
            return Err(ImageError::UnsupportedChannelCount { channels });
        }

        // More efficient conversion using iterator adaptors
        let normalized_data: Vec<f64> = if normalize {
            // Use const to avoid repeated division
            const INV_255: f64 = 1.0 / 255.0;
            data.into_iter()
                .map(|byte| f64::from(byte) * INV_255)
                .collect()
        } else {
            data.into_iter().map(f64::from).collect()
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
        paths: &[P],
        device: &B::Device,
    ) -> ImageResult<Tensor<B, 4>> {
        if paths.is_empty() {
            return Err(ImageError::EmptyPathList);
        }

        // Load first image to get expected dimensions
        let first_tensor =
            Self::load_image(&paths[0], device).map_err(|e| ImageError::BatchLoadError {
                index: 0,
                path: paths[0].as_ref().display().to_string(),
                source: Box::new(e),
            })?;
        let [_, _, expected_height, expected_width] = first_tensor.dims();

        let mut tensors = Vec::with_capacity(paths.len());
        tensors.push(first_tensor.squeeze::<3>(0));

        // Process remaining images
        for (i, path) in paths.iter().enumerate().skip(1) {
            let tensor =
                Self::load_image(path, device).map_err(|e| ImageError::BatchLoadError {
                    index: i,
                    path: path.as_ref().display().to_string(),
                    source: Box::new(e),
                })?;

            let [_, _, height, width] = tensor.dims();
            if height != expected_height || width != expected_width {
                return Err(ImageError::InconsistentImageDimensions {
                    index: i,
                    actual_height: height,
                    actual_width: width,
                    expected_height,
                    expected_width,
                });
            }

            tensors.push(tensor.squeeze::<3>(0));
        }

        Ok(Tensor::stack(tensors, 0))
    }

    /// Apply ImageNet normalization to a tensor
    ///
    /// # Arguments
    /// * `tensor` - Input tensor of shape [batch, 3, height, width] with values in [0, 1]
    ///
    /// # Returns
    /// Tensor with ImageNet normalization applied: (pixel - mean) / std
    pub fn apply_imagenet_normalization<B: Backend>(
        tensor: Tensor<B, 4>,
    ) -> ImageResult<Tensor<B, 4>> {
        let [_, channels, ..] = tensor.dims();

        if channels != 3 {
            return Err(ImageError::InvalidImageChannels { actual: channels });
        }

        let device = tensor.device();

        // Create mean and std tensors with shape [1, 3, 1, 1] for broadcasting
        let mean_data =
            TensorData::new(Self::IMAGENET_MEAN.to_vec(), [1, 3, 1, 1]).convert::<B::FloatElem>();
        let mean_tensor = Tensor::from_data(mean_data, &device);

        let std_data =
            TensorData::new(Self::IMAGENET_STD.to_vec(), [1, 3, 1, 1]).convert::<B::FloatElem>();
        let std_tensor = Tensor::from_data(std_data, &device);

        // Apply normalization: (tensor - mean) / std
        let normalized = tensor.sub(mean_tensor).div(std_tensor);

        Ok(normalized)
    }

    /// Load image from file and convert to tensor with ImageNet normalization
    ///
    /// # Arguments
    /// * `path` - Path to the image file
    /// * `device` - Device to create tensor on
    ///
    /// # Returns
    /// Tensor of shape [1, 3, height, width] with ImageNet normalization applied
    pub fn load_image_with_imagenet_normalization<B: Backend, P: AsRef<Path>>(
        path: P,
        device: &B::Device,
    ) -> ImageResult<Tensor<B, 4>> {
        // First load image with standard [0, 1] normalization
        let tensor = Self::load_image(path, device)?;

        // Then apply ImageNet normalization
        Self::apply_imagenet_normalization(tensor)
    }

    /// Convert DynamicImage to tensor with ImageNet normalization
    ///
    /// # Arguments
    /// * `img` - Input image
    /// * `device` - Device to create tensor on
    ///
    /// # Returns
    /// Tensor of shape [1, 3, height, width] with ImageNet normalization applied
    pub fn dynamic_image_to_tensor_with_imagenet_normalization<B: Backend>(
        img: DynamicImage,
        device: &B::Device,
    ) -> ImageResult<Tensor<B, 4>> {
        // First convert to standard [0, 1] tensor
        let tensor = Self::dynamic_image_to_tensor(img, device)?;

        // Then apply ImageNet normalization
        Self::apply_imagenet_normalization(tensor)
    }

    /// Batch process multiple images with ImageNet normalization
    ///
    /// # Arguments
    /// * `paths` - Vector of image paths
    /// * `device` - Device to create tensors on
    ///
    /// # Returns
    /// Tensor of shape [batch_size, 3, height, width] with ImageNet normalization applied
    pub fn load_image_batch_with_imagenet_normalization<
        B: Backend,
        P: AsRef<Path> + Send + Sync,
    >(
        paths: &[P],
        device: &B::Device,
    ) -> ImageResult<Tensor<B, 4>> {
        // First load batch with standard [0, 1] normalization
        let tensor = Self::load_image_batch(paths, device)?;

        // Then apply ImageNet normalization
        Self::apply_imagenet_normalization(tensor)
    }

    /// Check if a file extension is supported by the image crate
    ///
    /// # Arguments
    /// * `extension` - File extension to check (with or without leading dot)
    ///
    /// # Returns
    /// * `bool` - true if the extension is supported for reading
    pub fn is_extension_supported(extension: &str) -> bool {
        let ext = extension.trim_start_matches('.');
        ImageFormat::from_extension(ext).is_some()
    }

    /// Check if a file path has a supported image format extension
    ///
    /// # Arguments
    /// * `path` - Path to check
    ///
    /// # Returns
    /// * `bool` - true if the file extension is supported
    pub fn is_supported_image_format<P: AsRef<Path>>(path: P) -> bool {
        path.as_ref()
            .extension()
            .and_then(|ext| ext.to_str())
            .map(Self::is_extension_supported)
            .unwrap_or(false)
    }

    /// Get a list of commonly supported image extensions
    ///
    /// This returns extensions that are typically supported by the image crate.
    /// For definitive support checking, use `is_extension_supported()`.
    ///
    /// # Returns
    /// * `Vec<&'static str>` - Vector of supported file extensions
    pub fn get_common_image_extensions() -> Vec<&'static str> {
        const COMMON_EXTENSIONS: &[&str] = &[
            "jpg", "jpeg", "png", "gif", "bmp", "ico", "tiff", "tif", "webp", "pnm", "pbm", "pgm",
            "ppm", "pam", "dds", "tga", "hdr", "exr", "ff", "qoi", "avif",
        ];

        COMMON_EXTENSIONS
            .iter()
            .filter(|&ext| Self::is_extension_supported(ext))
            .copied()
            .collect()
    }

    /// Get all supported image formats from the image crate
    ///
    /// # Returns
    /// * `Vec<ImageFormat>` - Vector of all supported ImageFormat variants
    pub fn get_supported_formats() -> Vec<ImageFormat> {
        // Note: ImageFormat::all() method may not be available in all versions
        // This is a fallback implementation that tests common formats
        let formats_to_test = [
            ImageFormat::Png,
            ImageFormat::Jpeg,
            ImageFormat::Gif,
            ImageFormat::WebP,
            ImageFormat::Bmp,
            ImageFormat::Ico,
            ImageFormat::Tiff,
            ImageFormat::Tga,
            ImageFormat::Dds,
            ImageFormat::Hdr,
            ImageFormat::Farbfeld,
            ImageFormat::Avif,
            ImageFormat::Qoi,
        ];

        formats_to_test
            .into_iter()
            .filter(|format| format.reading_enabled())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    use super::*;

    type TestBackend = NdArray;

    #[test]
    fn from_raw_pixels_invalid_data_returns_error() {
        let device = Default::default();

        // Test invalid data length
        let result = ImageUtils::from_raw_pixels::<TestBackend>(
            vec![255_u8; 10], // Invalid length
            2,
            2,
            3, // 2x2 RGB should need 12 bytes
            &device,
        );
        assert!(matches!(
            result.unwrap_err(),
            ImageError::DataLengthMismatch { .. }
        ));

        // Test invalid channel count
        let result = ImageUtils::from_raw_pixels::<TestBackend>(
            vec![255_u8; 8],
            2,
            2,
            2, // Invalid channel count
            &device,
        );
        assert!(matches!(
            result.unwrap_err(),
            ImageError::UnsupportedChannelCount { .. }
        ));
    }

    #[test]
    fn apply_mask_mismatched_dimensions_returns_error() {
        let device = Default::default();

        // Create test tensors with mismatched dimensions
        let image = Tensor::<TestBackend, 4>::zeros([1, 3, 10, 10], &device);
        let mask = Tensor::<TestBackend, 4>::zeros([1, 1, 5, 5], &device); // Wrong size

        let result = ImageUtils::apply_mask(image, mask);
        assert!(matches!(
            result.unwrap_err(),
            ImageError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn from_raw_pixels_with_normalization_works_correctly() {
        let device = Default::default();

        // Test with normalization
        let data = vec![0_u8, 127, 255]; // Should become [0.0, ~0.5, 1.0]
        let result = ImageUtils::from_raw_pixels_with_normalization::<TestBackend>(
            data, 1, 1, 3, &device, true,
        );
        assert!(result.is_ok());

        // Test without normalization
        let data = vec![0_u8, 127, 255]; // Should remain [0.0, 127.0, 255.0]
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

        let result = ImageUtils::load_image_batch::<TestBackend, _>(&empty_paths, &device);
        assert!(matches!(result.unwrap_err(), ImageError::EmptyPathList));
    }

    #[test]
    fn apply_imagenet_normalization_works_correctly() {
        let device = Default::default();

        // Create a test tensor with shape [1, 3, 2, 2]
        // Channel 0: ImageNet mean for R (0.485), then 1.0, 0.0, 0.5
        // Channel 1: ImageNet mean for G (0.456), then 1.0, 0.0, 0.5
        // Channel 2: ImageNet mean for B (0.406), then 1.0, 0.0, 0.5
        let test_data: Vec<f32> = vec![
            // Channel 0 (Red): should use mean=0.485, std=0.229
            0.485, 1.0, 0.0, 0.5, // Channel 1 (Green): should use mean=0.456, std=0.224
            0.456, 1.0, 0.0, 0.5, // Channel 2 (Blue): should use mean=0.406, std=0.225
            0.406, 1.0, 0.0, 0.5,
        ];

        let tensor_data = TensorData::new(test_data, [1, 3, 2, 2]);
        let tensor = Tensor::<TestBackend, 4>::from_data(tensor_data, &device);

        let normalized = ImageUtils::apply_imagenet_normalization(tensor).unwrap();
        let result_data = normalized.into_data().to_vec::<f32>().unwrap();

        // Check that normalization was applied correctly for each channel
        // Channel 0 (Red): mean=0.485, std=0.229
        assert!((result_data[0] - 0.0).abs() < 1e-6); // (0.485 - 0.485) / 0.229 ≈ 0
        assert!((result_data[1] - 2.2489).abs() < 1e-3); // (1.0 - 0.485) / 0.229 ≈ 2.2489
        assert!((result_data[2] - (-2.1179)).abs() < 1e-3); // (0.0 - 0.485) / 0.229 ≈ -2.1179
        assert!((result_data[3] - 0.0655).abs() < 1e-3); // (0.5 - 0.485) / 0.229 ≈ 0.0655

        // Channel 1 (Green): mean=0.456, std=0.224
        assert!((result_data[4] - 0.0).abs() < 1e-6); // (0.456 - 0.456) / 0.224 ≈ 0
        assert!((result_data[5] - 2.4286).abs() < 1e-3); // (1.0 - 0.456) / 0.224 ≈ 2.4286
        assert!((result_data[6] - (-2.0357)).abs() < 1e-3); // (0.0 - 0.456) / 0.224 ≈ -2.0357
        assert!((result_data[7] - 0.1964).abs() < 1e-3); // (0.5 - 0.456) / 0.224 ≈ 0.1964

        // Channel 2 (Blue): mean=0.406, std=0.225
        assert!((result_data[8] - 0.0).abs() < 1e-6); // (0.406 - 0.406) / 0.225 ≈ 0
        assert!((result_data[9] - 2.6400).abs() < 1e-3); // (1.0 - 0.406) / 0.225 ≈ 2.6400
        assert!((result_data[10] - (-1.8044)).abs() < 1e-3); // (0.0 - 0.406) / 0.225 ≈ -1.8044
        assert!((result_data[11] - 0.4178).abs() < 1e-3); // (0.5 - 0.406) / 0.225 ≈ 0.4178
    }

    #[test]
    fn imagenet_normalization_constants_are_correct() {
        // Verify that ImageNet constants match PyTorch values
        assert_eq!(ImageUtils::IMAGENET_MEAN, [0.485, 0.456, 0.406]);
        assert_eq!(ImageUtils::IMAGENET_STD, [0.229, 0.224, 0.225]);
    }

    #[test]
    fn apply_imagenet_normalization_invalid_channels_returns_error() {
        let device = Default::default();

        // Create tensor with wrong number of channels (4 instead of 3)
        let tensor = Tensor::<TestBackend, 4>::zeros([1, 4, 10, 10], &device);

        let result = ImageUtils::apply_imagenet_normalization(tensor);
        assert!(matches!(
            result.unwrap_err(),
            ImageError::InvalidImageChannels { actual: 4 }
        ));
    }

    #[test]
    fn is_extension_supported_works_correctly() {
        // Test common supported extensions
        assert!(ImageUtils::is_extension_supported("png"));
        assert!(ImageUtils::is_extension_supported("jpg"));
        assert!(ImageUtils::is_extension_supported("jpeg"));
        assert!(ImageUtils::is_extension_supported("gif"));

        // Test with leading dot
        assert!(ImageUtils::is_extension_supported(".png"));
        assert!(ImageUtils::is_extension_supported(".jpg"));

        // Test unsupported extensions
        assert!(!ImageUtils::is_extension_supported("txt"));
        assert!(!ImageUtils::is_extension_supported("pdf"));
        assert!(!ImageUtils::is_extension_supported("doc"));
    }

    #[test]
    fn is_supported_image_format_works_correctly() {
        // Test supported formats
        assert!(ImageUtils::is_supported_image_format("test.png"));
        assert!(ImageUtils::is_supported_image_format("test.jpg"));
        assert!(ImageUtils::is_supported_image_format("test.jpeg"));
        assert!(ImageUtils::is_supported_image_format("test.gif"));

        // Test case insensitive
        assert!(ImageUtils::is_supported_image_format("test.PNG"));
        assert!(ImageUtils::is_supported_image_format("test.JPG"));

        // Test unsupported formats
        assert!(!ImageUtils::is_supported_image_format("test.txt"));
        assert!(!ImageUtils::is_supported_image_format("test.pdf"));

        // Test no extension
        assert!(!ImageUtils::is_supported_image_format("test"));

        // Test complex paths
        assert!(ImageUtils::is_supported_image_format("/path/to/image.png"));
        assert!(ImageUtils::is_supported_image_format(
            "./relative/path/image.jpg"
        ));
    }

    #[test]
    fn get_common_image_extensions_returns_valid_extensions() {
        let extensions = ImageUtils::get_common_image_extensions();

        // Should contain at least some common formats
        assert!(!extensions.is_empty());
        assert!(extensions.contains(&"png"));
        assert!(extensions.contains(&"jpg") || extensions.contains(&"jpeg"));

        // All returned extensions should be supported
        for ext in &extensions {
            assert!(
                ImageUtils::is_extension_supported(ext),
                "Extension '{}' should be supported",
                ext
            );
        }
    }

    #[test]
    fn get_supported_formats_returns_valid_formats() {
        let formats = ImageUtils::get_supported_formats();

        // Should contain at least some formats
        assert!(!formats.is_empty());

        // All returned formats should have reading enabled
        for format in &formats {
            assert!(
                format.reading_enabled(),
                "Format {:?} should have reading enabled",
                format
            );
        }
    }
}
