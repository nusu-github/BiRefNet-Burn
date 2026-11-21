//! Data augmentation transforms for `BiRefNet` training and validation.

use std::sync::atomic::{AtomicU64, Ordering};

use image::{DynamicImage, GenericImageView, Rgb, imageops::FilterType};
use rand::{RngExt, SeedableRng, rngs::StdRng};

use crate::transforms::{rotate_about_center_luma8, rotate_about_center_rgb8, sharpen_rgb};

/// Available data augmentation transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AugmentationMethod {
    /// Horizontal flip (50% probability).
    Flip,
    /// Color enhancement (brightness, contrast, hue rotation, and sharpness).
    Enhance,
    /// Random rotation (20% probability, up to plus or minus 15 degrees).
    Rotate,
    /// Sparse salt-and-pepper noise (0.15% pixel density).
    Pepper,
    /// Random border crop (up to 10% per edge).
    Crop,
}

/// Dynamic input size selection bounds (32-pixel aligned for Swin/PVT strides).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DynamicSizeConfig {
    /// Minimum square spatial resolution in pixels.
    pub min_size: u32,
    /// Maximum square spatial resolution in pixels.
    pub max_size: u32,
    /// Alignment increment between candidate resolutions (typically 32).
    pub size_step: u32,
}

/// Background color synthesis modes when compositing masks onto synthetic backdrops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackgroundPattern {
    /// Solid black, gray, or white background (40% probability).
    Solid,
    /// Jittered foreground-sampled color (40% probability).
    Similar,
    /// Uniformly sampled random RGB color (20% probability).
    Random,
}

/// Configuration for [`ImageAugmentor`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AugmentationConfig {
    /// Ordered list of augmentation transforms enabled during training.
    pub enabled_methods: Vec<AugmentationMethod>,
    /// Whether to replace background pixels with synthetic colors.
    pub background_color_synthesis: bool,
    /// Optional dynamic square sizing schedule used instead of `target_size`.
    pub dynamic_size_config: Option<DynamicSizeConfig>,
    /// Default `(width, height)` output size when dynamic sizing is disabled or during validation.
    pub target_size: (u32, u32),
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
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
    /// Creates an augmentation config enabling background color synthesis and horizontal flips.
    #[must_use]
    pub fn with_background_synthesis(target_size: (u32, u32)) -> Self {
        Self {
            enabled_methods: vec![AugmentationMethod::Flip],
            background_color_synthesis: true,
            dynamic_size_config: None,
            target_size,
        }
    }

    /// Creates an augmentation config enabling 32-aligned dynamic training resolutions.
    #[must_use]
    pub fn with_dynamic_size(min_size: u32, max_size: u32, target_size: (u32, u32)) -> Self {
        Self {
            dynamic_size_config: Some(DynamicSizeConfig {
                min_size,
                max_size,
                size_step: 32,
            }),
            target_size,
            ..Self::default()
        }
    }
}

const fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "crop border dimensions are non-negative and strictly bounded by u32 image size"
)]
fn ten_percent_border(dim: u32) -> u32 {
    (f64::from(dim) * 0.1) as u32
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "color delta offsets fit within i32 without overflow"
)]
fn float_to_i32(value: f32) -> i32 {
    value as i32
}

#[expect(
    clippy::cast_sign_loss,
    reason = "RGB channel value is clamped to [0, 255] before cast"
)]
fn clamp_u8(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

/// Applies training augmentations or deterministic validation resizing to image/mask pairs.
#[derive(Debug)]
pub struct ImageAugmentor {
    config: AugmentationConfig,
    seed: AtomicU64,
}

impl ImageAugmentor {
    /// Creates an [`ImageAugmentor`] seeded from entropy.
    #[must_use]
    pub fn new(config: AugmentationConfig) -> Self {
        Self::with_seed(config, rand::random())
    }

    /// Creates an [`ImageAugmentor`] with an explicit reproducible starting `seed`.
    #[must_use]
    pub const fn with_seed(config: AugmentationConfig, seed: u64) -> Self {
        Self {
            config,
            seed: AtomicU64::new(seed),
        }
    }

    /// Returns the configuration used by this augmentor.
    #[must_use]
    pub const fn config(&self) -> &AugmentationConfig {
        &self.config
    }

    fn next_rng(&self, stream_tag: u64) -> StdRng {
        let counter = self.seed.fetch_add(1, Ordering::Relaxed);
        StdRng::seed_from_u64(splitmix64(counter ^ stream_tag))
    }

    /// Applies the configured augmentations (or resize-only when `is_training` is `false`).
    #[must_use]
    pub fn augment(
        &self,
        mut image: DynamicImage,
        mut mask: DynamicImage,
        is_training: bool,
    ) -> (DynamicImage, DynamicImage) {
        if !is_training {
            return self.resize_only(&image, &mask);
        }

        let current_target_size = match self.config.dynamic_size_config {
            Some(dynamic_config) => self.select_dynamic_size(dynamic_config),
            None => self.config.target_size,
        };

        if self.config.background_color_synthesis {
            (image, mask) = self.apply_background_synthesis(&image, mask);
        }

        for method in &self.config.enabled_methods {
            (image, mask) = match method {
                AugmentationMethod::Flip => self.apply_horizontal_flip(image, mask),
                AugmentationMethod::Crop => self.apply_random_crop(image, mask),
                AugmentationMethod::Rotate => self.apply_random_rotation(image, mask),
                AugmentationMethod::Enhance => self.apply_color_enhancement(image, mask),
                AugmentationMethod::Pepper => self.apply_pepper_noise(&image, mask),
            };
        }

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

    /// Resizes `image` (`Lanczos3`) and `mask` (`Nearest`) to `config.target_size`.
    #[must_use]
    pub fn resize_only(
        &self,
        image: &DynamicImage,
        mask: &DynamicImage,
    ) -> (DynamicImage, DynamicImage) {
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

    fn select_dynamic_size(&self, config: DynamicSizeConfig) -> (u32, u32) {
        if config.size_step == 0 || config.max_size <= config.min_size {
            let aligned = (config.min_size.max(32) / 32) * 32;
            let clamped = aligned.max(32);
            return (clamped, clamped);
        }

        let mut rng = self.next_rng(0x01);
        let step_count = (config.max_size - config.min_size) / config.size_step;
        let selected_steps = rng.random_range(0..=step_count);
        let size = config.min_size + (selected_steps * config.size_step);

        let aligned_size = ((size / 32) * 32).max(32);
        (aligned_size, aligned_size)
    }

    fn apply_horizontal_flip(
        &self,
        image: DynamicImage,
        mask: DynamicImage,
    ) -> (DynamicImage, DynamicImage) {
        let mut rng = self.next_rng(0x02);
        if rng.random_bool(0.5) {
            (image.fliph(), mask.fliph())
        } else {
            (image, mask)
        }
    }

    fn apply_random_crop(
        &self,
        image: DynamicImage,
        mask: DynamicImage,
    ) -> (DynamicImage, DynamicImage) {
        let (width, height) = image.dimensions();
        if width <= 2 || height <= 2 {
            return (image, mask);
        }

        let border_x = ten_percent_border(width);
        let border_y = ten_percent_border(height);

        let min_crop_width = width.saturating_sub(2 * border_x).max(1);
        let min_crop_height = height.saturating_sub(2 * border_y).max(1);
        if min_crop_width >= width && min_crop_height >= height {
            return (image, mask);
        }

        let mut rng = self.next_rng(0x03);
        let crop_width = rng.random_range(min_crop_width..=width);
        let crop_height = rng.random_range(min_crop_height..=height);

        let max_x = width.saturating_sub(crop_width);
        let max_y = height.saturating_sub(crop_height);

        let crop_x = if max_x > 0 {
            rng.random_range(0..=max_x)
        } else {
            0
        };
        let crop_y = if max_y > 0 {
            rng.random_range(0..=max_y)
        } else {
            0
        };

        let cropped_image = image.crop_imm(crop_x, crop_y, crop_width, crop_height);
        let cropped_mask = mask.crop_imm(crop_x, crop_y, crop_width, crop_height);
        (cropped_image, cropped_mask)
    }

    fn apply_random_rotation(
        &self,
        image: DynamicImage,
        mask: DynamicImage,
    ) -> (DynamicImage, DynamicImage) {
        let mut rng = self.next_rng(0x04);
        if !rng.random_bool(0.2) {
            return (image, mask);
        }

        let angle_degrees: f32 = rng.random_range(-15.0..=15.0);
        let angle_radians = angle_degrees.to_radians();

        let rotated_image =
            DynamicImage::ImageRgb8(rotate_about_center_rgb8(&image.to_rgb8(), angle_radians));
        let rotated_mask =
            DynamicImage::ImageLuma8(rotate_about_center_luma8(&mask.to_luma8(), angle_radians));
        (rotated_image, rotated_mask)
    }

    fn apply_color_enhancement(
        &self,
        mut image: DynamicImage,
        mask: DynamicImage,
    ) -> (DynamicImage, DynamicImage) {
        let mut rng = self.next_rng(0x05);

        let brightness_factor: f32 = rng.random_range(0.5..=1.5);
        let brightness_value = float_to_i32((brightness_factor - 1.0) * 255.0);
        image = image.brighten(brightness_value);

        let contrast_factor: f32 = rng.random_range(0.5..=1.5);
        let contrast_value = (contrast_factor - 1.0) * 100.0;
        image = image.adjust_contrast(contrast_value);

        let hue_factor: f32 = rng.random_range(0.0..=2.0);
        let hue_value = float_to_i32((hue_factor - 1.0) * 180.0);
        image = image.huerotate(hue_value);

        let sharpness_factor: f32 = rng.random_range(0.0..=3.0);
        if sharpness_factor > 1.0 {
            let rgb = image.to_rgb8();
            image = DynamicImage::ImageRgb8(sharpen_rgb(&rgb, sharpness_factor - 1.0));
        } else if sharpness_factor < 1.0 {
            let blur_sigma = (1.0 - sharpness_factor) * 2.0;
            image = image.blur(blur_sigma);
        }

        (image, mask)
    }

    fn apply_pepper_noise(
        &self,
        image: &DynamicImage,
        mask: DynamicImage,
    ) -> (DynamicImage, DynamicImage) {
        let mut rng = self.next_rng(0x06);
        let noise_rate = 0.0015;
        let mut result = image.to_rgb8();

        for pixel in result.pixels_mut() {
            if rng.random_bool(noise_rate) {
                *pixel = if rng.random_bool(0.5) {
                    Rgb([0u8; 3])
                } else {
                    Rgb([255u8; 3])
                };
            }
        }

        (DynamicImage::ImageRgb8(result), mask)
    }

    fn apply_background_synthesis(
        &self,
        image: &DynamicImage,
        mask: DynamicImage,
    ) -> (DynamicImage, DynamicImage) {
        let mut rng = self.next_rng(0x07);

        let pattern = match rng.random_range(0..10) {
            0..=3 => BackgroundPattern::Solid,
            4..=7 => BackgroundPattern::Similar,
            _ => BackgroundPattern::Random,
        };

        let background_color = match pattern {
            BackgroundPattern::Solid => match rng.random_range(0..3) {
                0 => Rgb([0u8; 3]),
                1 => Rgb([128u8; 3]),
                _ => Rgb([255u8; 3]),
            },
            BackgroundPattern::Similar => self.extract_similar_background_color(image, &mask),
            BackgroundPattern::Random => {
                Rgb([rng.random::<u8>(), rng.random::<u8>(), rng.random::<u8>()])
            }
        };

        let composed_image = Self::compose_background(image, &mask, background_color);
        (composed_image, mask)
    }

    fn extract_similar_background_color(
        &self,
        image: &DynamicImage,
        mask: &DynamicImage,
    ) -> Rgb<u8> {
        let mut rng = self.next_rng(0x08);
        let rgb_image = image.to_rgb8();
        let gray_mask = mask.to_luma8();

        let foreground_colors: Vec<Rgb<u8>> = rgb_image
            .pixels()
            .zip(gray_mask.pixels())
            .filter_map(|(&rgb_pixel, mask_pixel)| (mask_pixel.0[0] > 128).then_some(rgb_pixel))
            .collect();

        if foreground_colors.is_empty() {
            return Rgb([rng.random::<u8>(), rng.random::<u8>(), rng.random::<u8>()]);
        }

        let base_color = foreground_colors[rng.random_range(0..foreground_colors.len())];
        Rgb([
            clamp_u8(i32::from(base_color.0[0]) + rng.random_range(-30..=30)),
            clamp_u8(i32::from(base_color.0[1]) + rng.random_range(-30..=30)),
            clamp_u8(i32::from(base_color.0[2]) + rng.random_range(-30..=30)),
        ])
    }

    fn compose_background(
        image: &DynamicImage,
        mask: &DynamicImage,
        background_color: Rgb<u8>,
    ) -> DynamicImage {
        let mut result = image.to_rgb8();
        let gray_mask = mask.to_luma8();

        for (result_pixel, mask_pixel) in result.pixels_mut().zip(gray_mask.pixels()) {
            if mask_pixel.0[0] <= 128 {
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
    fn dynamic_size_handles_valid_and_degenerate_ranges() {
        let valid_cfg = DynamicSizeConfig {
            min_size: 320,
            max_size: 640,
            size_step: 32,
        };
        let augmentor = ImageAugmentor::with_seed(AugmentationConfig::default(), 7);
        let selected = augmentor.select_dynamic_size(valid_cfg);
        assert!(selected.0 >= 320 && selected.0 <= 640);
        assert_eq!(selected.0 % 32, 0);
        assert_eq!(selected.0, selected.1);

        let degenerate = DynamicSizeConfig {
            min_size: 200,
            max_size: 100,
            size_step: 0,
        };
        let clamped = augmentor.select_dynamic_size(degenerate);
        assert_eq!(clamped, (192, 192));
    }

    #[test]
    fn tiny_crop_and_transforms_behave_without_panicking() {
        let small = DynamicImage::ImageRgb8(ImageBuffer::from_pixel(2, 2, Rgb([10, 20, 30])));
        let small_mask =
            DynamicImage::ImageLuma8(ImageBuffer::from_pixel(2, 2, image::Luma([255])));
        let augmentor = ImageAugmentor::with_seed(AugmentationConfig::default(), 99);
        let (cropped_img, cropped_mask) = augmentor.apply_random_crop(small, small_mask);
        assert_eq!(cropped_img.dimensions(), (2, 2));
        assert_eq!(cropped_mask.dimensions(), (2, 2));

        let non_uniform = ImageBuffer::from_fn(12, 12, |x, y| {
            let v = if (4..8).contains(&x) && (4..8).contains(&y) {
                200u8
            } else {
                20u8
            };
            Rgb([v, v / 2, 255 - v])
        });
        let sharpened = sharpen_rgb(&non_uniform, 1.0);
        assert_ne!(sharpened.as_raw(), non_uniform.as_raw());

        let rotated = rotate_about_center_rgb8(&non_uniform, 0.25);
        assert_eq!(rotated.dimensions(), non_uniform.dimensions());
        assert_ne!(rotated.as_raw(), non_uniform.as_raw());
    }
}
