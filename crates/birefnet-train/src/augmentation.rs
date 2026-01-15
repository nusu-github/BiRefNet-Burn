//! Data augmentation processing implementation
//!
//! This module ports all data augmentation processes used in BiRefNet's Python training
//! implementation for the Rust/Burn framework.
//!
//! Implemented augmentation processes:
//! 1. Horizontal flip (cv_random_flip) - 50% probability
//! 2. Random crop (random_crop) - 10% border cropping
//! 3. Random rotation (random_rotate) - 20% probability, ±15 degrees
//! 4. Color enhancement (color_enhance) - brightness, contrast, saturation, sharpness adjustment
//! 5. Pepper noise (random_pepper) - 0.15% density
//! 6. Background color synthesis augmentation - solid, similar, random color patterns
//! 7. Dynamic input size augmentation - per-batch size changes

use image::{DynamicImage, GenericImageView, Rgb, buffer::ConvertBuffer, imageops::FilterType};
use imageproc::geometric_transformations::{Interpolation, rotate_about_center};
use rand::{Rng, SeedableRng};

/// Data augmentation configuration parameters
#[derive(Debug, Clone)]
pub struct AugmentationConfig {
    /// List of enabled augmentation methods
    pub enabled_methods: Vec<AugmentationMethod>,
    /// Whether to use background color synthesis augmentation
    pub background_color_synthesis: bool,
    /// Dynamic input size augmentation configuration
    pub dynamic_size_config: Option<DynamicSizeConfig>,
    /// Target output size
    pub target_size: (u32, u32),
}

/// Available data augmentation methods
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AugmentationMethod {
    /// Horizontal flip (50% probability)
    Flip,
    /// Color enhancement (brightness, contrast, saturation, sharpness)
    Enhance,
    /// Random rotation (20% probability, ±15 degrees)
    Rotate,
    /// Pepper noise (0.15% density)
    Pepper,
    /// Random crop (10% border)
    Crop,
}

/// Dynamic input size augmentation configuration
#[derive(Debug, Clone)]
pub struct DynamicSizeConfig {
    /// Minimum size
    pub min_size: u32,
    /// Maximum size
    pub max_size: u32,
    /// Size selection step (32x multiple constraint)
    pub size_step: u32,
}

/// Background color synthesis patterns
#[derive(Debug, Clone)]
pub enum BackgroundPattern {
    /// Solid background (40% probability) - black/gray/white
    Solid,
    /// Similar color background (40% probability) - colors similar to foreground objects
    Similar,
    /// Random color background (20% probability) - completely random
    Random,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            // Default configuration when background color synthesis is disabled: ['flip', 'enhance', 'rotate', 'pepper']
            enabled_methods: vec![
                AugmentationMethod::Flip,
                AugmentationMethod::Enhance,
                AugmentationMethod::Rotate,
                AugmentationMethod::Pepper,
            ],
            background_color_synthesis: false,
            dynamic_size_config: None,
            target_size: (1024, 1024),
        }
    }
}

impl AugmentationConfig {
    /// Create configuration for when background color synthesis augmentation is enabled
    pub fn with_background_synthesis(target_size: (u32, u32)) -> Self {
        Self {
            // When background color synthesis is enabled: flip only
            enabled_methods: vec![AugmentationMethod::Flip],
            background_color_synthesis: true,
            dynamic_size_config: None,
            target_size,
        }
    }

    /// Create configuration for when dynamic input size augmentation is enabled
    pub fn with_dynamic_size(min_size: u32, max_size: u32, target_size: (u32, u32)) -> Self {
        let mut config = Self::default();
        config.dynamic_size_config = Some(DynamicSizeConfig {
            min_size,
            max_size,
            size_step: 32, // 32x multiple constraint
        });
        config.target_size = target_size;
        config
    }
}

/// Main structure for executing data augmentation processing
pub struct ImageAugmentor {
    config: AugmentationConfig,
    // Store seed for thread safety and create new RNGs as needed
    seed: u64,
}

impl ImageAugmentor {
    /// Create a new ImageAugmentor instance
    pub fn new(config: AugmentationConfig) -> Self {
        Self {
            config,
            seed: rand::random(),
        }
    }

    /// Helper method to create RNG internally
    fn create_rng(&self, additional_entropy: Option<usize>) -> rand::rngs::StdRng {
        let final_seed = match additional_entropy {
            Some(entropy) => self.seed.wrapping_add(entropy as u64),
            None => self.seed,
        };
        rand::rngs::StdRng::seed_from_u64(final_seed)
    }

    /// Apply data augmentation to images and masks
    ///
    /// # Arguments
    /// * `image` - Input image
    /// * `mask` - Corresponding segmentation mask
    /// * `is_training` - Whether it's training time (augmentation is limited during validation)
    ///
    /// # Returns
    /// Tuple of augmented (image, mask)
    pub fn augment(
        &self,
        mut image: DynamicImage,
        mut mask: DynamicImage,
        is_training: bool,
    ) -> (DynamicImage, DynamicImage) {
        if !is_training {
            // Only resize during validation
            return self.resize_only(image, mask);
        }

        // Apply dynamic size augmentation
        let current_target_size = if let Some(ref dynamic_config) = self.config.dynamic_size_config
        {
            self.select_dynamic_size(dynamic_config)
        } else {
            self.config.target_size
        };

        // Apply background color synthesis augmentation
        if self.config.background_color_synthesis {
            (image, mask) = self.apply_background_synthesis(image, mask);
        }

        // Apply standard data augmentation processes
        for method in &self.config.enabled_methods {
            (image, mask) = match method {
                AugmentationMethod::Flip => self.apply_horizontal_flip(image, mask),
                AugmentationMethod::Crop => self.apply_random_crop(image, mask),
                AugmentationMethod::Rotate => self.apply_random_rotation(image, mask),
                AugmentationMethod::Enhance => self.apply_color_enhancement(image, mask),
                AugmentationMethod::Pepper => self.apply_pepper_noise(image, mask),
            };
        }

        // Final resize
        let image = image.resize_exact(
            current_target_size.0,
            current_target_size.1,
            FilterType::Lanczos3,
        );
        let mask = mask.resize_exact(
            current_target_size.0,
            current_target_size.1,
            FilterType::Nearest,
        );

        (image, mask)
    }

    /// Apply resize only (for validation)
    fn resize_only(&self, image: DynamicImage, mask: DynamicImage) -> (DynamicImage, DynamicImage) {
        let image = image.resize_exact(
            self.config.target_size.0,
            self.config.target_size.1,
            FilterType::Lanczos3,
        );
        let mask = mask.resize_exact(
            self.config.target_size.0,
            self.config.target_size.1,
            FilterType::Nearest,
        );
        (image, mask)
    }

    /// Select dynamic input size
    fn select_dynamic_size(&self, config: &DynamicSizeConfig) -> (u32, u32) {
        let mut rng = self.create_rng(None);
        let size_range = (config.max_size - config.min_size) / config.size_step;
        let selected_steps = rng.random_range(0..=size_range);
        let size = config.min_size + (selected_steps * config.size_step);

        // Adjust to multiple of 32
        let adjusted_size = (size / 32) * 32;
        (adjusted_size, adjusted_size)
    }

    /// Apply horizontal flip (50% probability)
    fn apply_horizontal_flip(
        &self,
        image: DynamicImage,
        mask: DynamicImage,
    ) -> (DynamicImage, DynamicImage) {
        let mut rng = self.create_rng(Some(1));
        if rng.random_bool(0.5) {
            let flipped_image = image.fliph();
            let flipped_mask = mask.fliph();
            (flipped_image, flipped_mask)
        } else {
            (image, mask)
        }
    }

    /// Apply random crop (10% border setting)
    fn apply_random_crop(
        &self,
        image: DynamicImage,
        mask: DynamicImage,
    ) -> (DynamicImage, DynamicImage) {
        let mut rng = self.create_rng(Some(2));
        let (width, height) = image.dimensions();

        // Set 10% border
        let border_x = (width as f32 * 0.1) as u32;
        let border_y = (height as f32 * 0.1) as u32;

        if border_x >= width || border_y >= height {
            return (image, mask);
        }

        // Randomly determine crop region
        let min_crop_width = width - 2 * border_x;
        let min_crop_height = height - 2 * border_y;

        let crop_width = rng.random_range(min_crop_width..=width);
        let crop_height = rng.random_range(min_crop_height..=height);

        let max_x = width.saturating_sub(crop_width);
        let max_y = height.saturating_sub(crop_height);

        let crop_x = if max_x > 0 {
            rng.random_range(0..max_x)
        } else {
            0
        };
        let crop_y = if max_y > 0 {
            rng.random_range(0..max_y)
        } else {
            0
        };

        let cropped_image = image.crop_imm(crop_x, crop_y, crop_width, crop_height);
        let cropped_mask = mask.crop_imm(crop_x, crop_y, crop_width, crop_height);

        (cropped_image, cropped_mask)
    }

    /// Apply random rotation (20% probability, ±15 degrees)
    fn apply_random_rotation(
        &self,
        image: DynamicImage,
        mask: DynamicImage,
    ) -> (DynamicImage, DynamicImage) {
        let mut rng = self.create_rng(Some(3));

        // Apply rotation with 20% probability
        if !rng.random_bool(0.2) {
            return (image, mask);
        }

        // Generate random angle of ±15 degrees
        let angle_degrees: f32 = rng.random_range(-15.0..=15.0);
        let angle_radians = angle_degrees.to_radians();

        // Rotate image and mask
        let rotated_image = self.rotate_image(&image, angle_radians, false);
        let rotated_mask = self.rotate_image(&mask, angle_radians, true);

        (rotated_image, rotated_mask)
    }

    /// Helper function for image rotation processing
    fn rotate_image(&self, image: &DynamicImage, angle: f32, is_mask: bool) -> DynamicImage {
        let (_width, _height) = image.dimensions();

        if is_mask {
            // Use nearest neighbor interpolation for masks
            let gray_image = image.to_luma8();
            let rotated = rotate_about_center(
                &gray_image,
                angle,
                Interpolation::Nearest,
                image::Luma([0u8]),
            );
            DynamicImage::ImageLuma8(rotated)
        } else {
            // Use bicubic interpolation for regular images
            let rgb_image = image.to_rgb8();
            let rotated = rotate_about_center(
                &rgb_image,
                angle,
                Interpolation::Bicubic,
                image::Rgb([0u8, 0u8, 0u8]),
            );
            DynamicImage::ImageRgb8(rotated)
        }
    }

    /// Apply color enhancement (brightness, contrast, saturation, sharpness)
    fn apply_color_enhancement(
        &self,
        mut image: DynamicImage,
        mask: DynamicImage,
    ) -> (DynamicImage, DynamicImage) {
        let mut rng = self.create_rng(Some(4));

        // Brightness adjustment: 0.5 to 1.5x
        let brightness_factor: f32 = rng.random_range(0.5..=1.5);
        let brightness_value = ((brightness_factor - 1.0) * 255.0) as i32;
        image = image.brighten(brightness_value);

        // Contrast adjustment: 0.5 to 1.5x
        let contrast_factor: f32 = rng.random_range(0.5..=1.5);
        let contrast_value = ((contrast_factor - 1.0) * 100.0) as i32;
        image = image.adjust_contrast(contrast_value as f32);

        // Hue adjustment (as alternative to saturation adjustment): 0.0 to 2.0x change
        let hue_factor: f32 = rng.random_range(0.0..=2.0);
        let hue_value = ((hue_factor - 1.0) * 180.0) as i32;
        image = image.huerotate(hue_value);

        // Sharpness adjustment: 0.0 to 3.0x
        let sharpness_factor: f32 = rng.random_range(0.0..=3.0);
        if sharpness_factor > 1.0 {
            // Apply sharpening
            image = self.apply_sharpening(&image, sharpness_factor);
        } else if sharpness_factor < 1.0 {
            // Apply blur (reduce sharpness)
            let blur_sigma = (1.0 - sharpness_factor) * 2.0;
            image = image.blur(blur_sigma);
        }

        (image, mask)
    }

    /// Apply sharpening filter
    fn apply_sharpening(&self, image: &DynamicImage, factor: f32) -> DynamicImage {
        // Laplacian-based sharpening
        let rgb_image = image.to_rgb8();
        let gray_image: image::ImageBuffer<image::Luma<u8>, Vec<u8>> = rgb_image.convert();
        let sharpened = imageproc::filter::sharpen3x3(&gray_image);

        // Adjust sharpening intensity
        let original = &rgb_image;
        let mut result = original.clone();

        for ((original_pixel, sharpened_pixel), result_pixel) in original
            .pixels()
            .zip(sharpened.pixels())
            .zip(result.pixels_mut())
        {
            // Apply grayscale sharpening result to RGB
            for ((&o, s), r) in original_pixel
                .0
                .iter()
                .zip(std::iter::repeat(sharpened_pixel.0[0]))
                .zip(&mut result_pixel.0)
            {
                let diff = s as f32 - o as f32;
                let enhanced = diff.mul_add(factor - 1.0, o as f32);
                *r = enhanced.clamp(0.0, 255.0) as u8;
            }
        }

        DynamicImage::ImageRgb8(result)
    }

    /// Apply pepper noise (0.15% density)
    fn apply_pepper_noise(
        &self,
        image: DynamicImage,
        mask: DynamicImage,
    ) -> (DynamicImage, DynamicImage) {
        let mut rng = self.create_rng(Some(5));
        let noise_rate = 0.0015; // 0.15%
        let seed = rng.random::<u64>();

        // Apply pepper noise (black dots only)
        let rgb_image = image.to_rgb8();
        let noisy_image = self.apply_pepper_only(&rgb_image, noise_rate, seed);

        (DynamicImage::ImageRgb8(noisy_image), mask)
    }

    /// Implementation of pepper noise (black dots only)
    fn apply_pepper_only(
        &self,
        image: &image::ImageBuffer<image::Rgb<u8>, Vec<u8>>,
        rate: f64,
        seed: u64,
    ) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut result = image.clone();

        for pixel in result.pixels_mut() {
            if rng.random_bool(rate) {
                // Randomly select 0 or 255 (pepper noise in original)
                if rng.random_bool(0.5) {
                    *pixel = image::Rgb([0u8, 0u8, 0u8]); // Black
                } else {
                    *pixel = image::Rgb([255u8, 255u8, 255u8]); // White
                }
            }
        }

        result
    }

    /// Apply background color synthesis augmentation
    fn apply_background_synthesis(
        &self,
        image: DynamicImage,
        mask: DynamicImage,
    ) -> (DynamicImage, DynamicImage) {
        let mut rng = self.create_rng(Some(6));

        // Randomly select background pattern
        let pattern = match rng.random_range(0..10) {
            0..=3 => BackgroundPattern::Solid,   // 40%
            4..=7 => BackgroundPattern::Similar, // 40%
            _ => BackgroundPattern::Random,      // 20%
        };

        let background_color = match pattern {
            BackgroundPattern::Solid => {
                // Random selection from black/gray/white
                match rng.random_range(0..3) {
                    0 => Rgb([0u8, 0u8, 0u8]),       // Black
                    1 => Rgb([128u8, 128u8, 128u8]), // Gray
                    _ => Rgb([255u8, 255u8, 255u8]), // White
                }
            }
            BackgroundPattern::Similar => {
                // Generate colors similar to foreground objects
                self.extract_similar_background_color(&image, &mask)
            }
            BackgroundPattern::Random => {
                // Completely random color
                Rgb([rng.random::<u8>(), rng.random::<u8>(), rng.random::<u8>()])
            }
        };

        // Composite background
        let composed_image = self.compose_background(&image, &mask, background_color);
        (composed_image, mask)
    }

    /// Extract background colors similar to foreground objects
    fn extract_similar_background_color(
        &self,
        image: &DynamicImage,
        mask: &DynamicImage,
    ) -> Rgb<u8> {
        let mut rng = self.create_rng(Some(7));
        let rgb_image = image.to_rgb8();
        let gray_mask = mask.to_luma8();

        let mut foreground_colors = Vec::new();

        // Collect colors in foreground region
        for (rgb_pixel, mask_pixel) in rgb_image.pixels().zip(gray_mask.pixels()) {
            if mask_pixel.0[0] > 128 {
                // Foreground determination
                foreground_colors.push(*rgb_pixel);
            }
        }

        if foreground_colors.is_empty() {
            // Random color if no foreground is found
            return Rgb([rng.random::<u8>(), rng.random::<u8>(), rng.random::<u8>()]);
        }

        // Randomly select foreground color and slightly modify it
        let base_color = foreground_colors[rng.random_range(0..foreground_colors.len())];

        Rgb([
            (base_color.0[0] as i32 + rng.random_range(-30..=30)).clamp(0, 255) as u8,
            (base_color.0[1] as i32 + rng.random_range(-30..=30)).clamp(0, 255) as u8,
            (base_color.0[2] as i32 + rng.random_range(-30..=30)).clamp(0, 255) as u8,
        ])
    }

    /// Compose background color
    fn compose_background(
        &self,
        image: &DynamicImage,
        mask: &DynamicImage,
        background_color: Rgb<u8>,
    ) -> DynamicImage {
        let rgb_image = image.to_rgb8();
        let gray_mask = mask.to_luma8();
        let mut result = rgb_image.clone();

        // Composite background based on mask
        for ((result_pixel, _), mask_pixel) in result
            .pixels_mut()
            .zip(rgb_image.pixels())
            .zip(gray_mask.pixels())
        {
            if mask_pixel.0[0] <= 128 {
                // Background region
                *result_pixel = background_color;
            }
        }

        DynamicImage::ImageRgb8(result)
    }
}

#[cfg(test)]
mod tests {
    use image::ImageBuffer;

    use super::*;

    #[test]
    fn augmentation_config_default_methods() {
        let config = AugmentationConfig::default();
        assert_eq!(config.enabled_methods.len(), 4);
        assert!(config.enabled_methods.contains(&AugmentationMethod::Flip));
        assert!(
            config
                .enabled_methods
                .contains(&AugmentationMethod::Enhance)
        );
        assert!(config.enabled_methods.contains(&AugmentationMethod::Rotate));
        assert!(config.enabled_methods.contains(&AugmentationMethod::Pepper));
        assert!(!config.background_color_synthesis);
    }

    #[test]
    fn augmentation_config_with_background_synthesis() {
        let config = AugmentationConfig::with_background_synthesis((512, 512));
        assert_eq!(config.enabled_methods.len(), 1);
        assert!(config.enabled_methods.contains(&AugmentationMethod::Flip));
        assert!(config.background_color_synthesis);
    }

    #[test]
    fn horizontal_flip_deterministic() {
        let config = AugmentationConfig::default();
        let augmentor = ImageAugmentor::new(config);

        // Create small test image
        let image = DynamicImage::ImageRgb8(ImageBuffer::from_fn(4, 4, |x, y| {
            image::Rgb([x as u8 * 60, y as u8 * 60, 128])
        }));
        let mask = DynamicImage::ImageLuma8(ImageBuffer::from_fn(4, 4, |x, y| {
            image::Luma([if x + y > 4 { 255 } else { 0 }])
        }));

        // Test flip processing (result is probabilistic, but confirm function works)
        let (result_image, result_mask) =
            augmentor.apply_horizontal_flip(image.clone(), mask.clone());

        // Confirm that result image size is maintained
        assert_eq!(result_image.dimensions(), image.dimensions());
        assert_eq!(result_mask.dimensions(), mask.dimensions());
    }

    #[test]
    fn dynamic_size_selection() {
        let dynamic_config = DynamicSizeConfig {
            min_size: 320,
            max_size: 640,
            size_step: 32,
        };

        let config = AugmentationConfig::with_dynamic_size(320, 640, (512, 512));
        let augmentor = ImageAugmentor::new(config);

        let selected_size = augmentor.select_dynamic_size(&dynamic_config);

        // Confirm selected size is within range and multiple of 32
        assert!(selected_size.0 >= 320);
        assert!(selected_size.0 <= 640);
        assert_eq!(selected_size.0 % 32, 0);
        assert_eq!(selected_size.0, selected_size.1); // Square
    }

    #[test]
    fn resize_only_validation() {
        let config = AugmentationConfig::default();
        let augmentor = ImageAugmentor::new(config);

        let image = DynamicImage::ImageRgb8(ImageBuffer::from_fn(100, 100, |x, y| {
            image::Rgb([x as u8, y as u8, 128])
        }));
        let mask =
            DynamicImage::ImageLuma8(ImageBuffer::from_fn(100, 100, |_, _| image::Luma([255])));

        let (result_image, result_mask) = augmentor.resize_only(image, mask);

        // Confirm resized to target size
        assert_eq!(result_image.dimensions(), (1024, 1024));
        assert_eq!(result_mask.dimensions(), (1024, 1024));
    }
}
