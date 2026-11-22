//! Random training augmentations for image / mask pairs.

use std::{
    fmt,
    sync::atomic::{AtomicU64, Ordering},
};

use image::{
    GrayImage, Rgb, RgbImage,
    imageops::{self, colorops},
};
use rand::{Rng, RngExt, SeedableRng, rngs::SmallRng};

use crate::transforms::{rotate_about_center_luma8, rotate_about_center_rgb8, sharpen_rgb};

/// A random training transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AugmentationMethod {
    /// Horizontal flip with probability 50%.
    Flip,
    /// Random brightness, contrast, hue rotation, and sharpness.
    Enhance,
    /// Rotation by up to ±15° with probability 20%.
    Rotate,
    /// Salt-and-pepper noise on 0.15% of pixels.
    Pepper,
    /// Random crop removing up to 10% of each dimension.
    Crop,
}

impl fmt::Display for AugmentationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Flip => "flip",
            Self::Enhance => "enhance",
            Self::Rotate => "rotate",
            Self::Pepper => "pepper",
            Self::Crop => "crop",
        })
    }
}

/// Configuration for [`ImageAugmentor`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AugmentationConfig {
    methods: Vec<AugmentationMethod>,
    background_synthesis: bool,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl AugmentationConfig {
    /// Flip, enhance, rotate, and pepper noise (the upstream `preproc_methods`), without
    /// background synthesis.
    #[must_use]
    pub fn new() -> Self {
        Self {
            methods: vec![
                AugmentationMethod::Flip,
                AugmentationMethod::Enhance,
                AugmentationMethod::Rotate,
                AugmentationMethod::Pepper,
            ],
            background_synthesis: false,
        }
    }

    /// Replaces the transforms, applied in the given order.
    #[must_use]
    pub fn with_methods(self, methods: impl IntoIterator<Item = AugmentationMethod>) -> Self {
        Self {
            methods: methods.into_iter().collect(),
            ..self
        }
    }

    /// Enables replacing the background with a synthetic color before the other transforms.
    #[must_use]
    pub fn with_background_synthesis(self, enabled: bool) -> Self {
        Self {
            background_synthesis: enabled,
            ..self
        }
    }

    /// Returns the transforms in application order.
    #[must_use]
    pub fn methods(&self) -> &[AugmentationMethod] {
        &self.methods
    }

    /// Returns `true` if background synthesis is enabled.
    #[must_use]
    pub const fn background_synthesis(&self) -> bool {
        self.background_synthesis
    }
}

/// `SplitMix64` finalizer: decorrelates consecutive counters before seeding a generator.
pub(crate) const fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Thread-safe source of independent per-call generators derived from one seed.
#[derive(Debug)]
pub(crate) struct SeedSequence(AtomicU64);

impl SeedSequence {
    pub(crate) const fn new(seed: u64) -> Self {
        Self(AtomicU64::new(seed))
    }

    pub(crate) fn next_rng(&self) -> SmallRng {
        SmallRng::seed_from_u64(splitmix64(self.0.fetch_add(1, Ordering::Relaxed)))
    }
}

/// Fraction of pixels replaced by salt-and-pepper noise.
const PEPPER_DENSITY: f64 = 0.0015;
/// Probability and maximum angle (degrees) of a rotation.
const ROTATE_PROBABILITY: f64 = 0.2;
const ROTATE_MAX_DEGREES: f32 = 15.0;
/// Mask values above this are foreground when synthesizing backgrounds.
const FOREGROUND_THRESHOLD: u8 = 128;
/// Per-channel jitter of a background color sampled from the foreground.
const SIMILAR_COLOR_JITTER: i32 = 30;

/// Applies the configured random transforms to image / mask pairs.
///
/// Each call draws one generator from a seed sequence, so augmentation is reproducible for a
/// fixed seed and call order, and safe to share between data-loader workers.
#[derive(Debug)]
pub struct ImageAugmentor {
    config: AugmentationConfig,
    seeds: SeedSequence,
}

impl ImageAugmentor {
    /// Creates an augmentor seeded from the operating system's entropy.
    #[must_use]
    pub fn new(config: AugmentationConfig) -> Self {
        Self::with_seed(config, rand::random())
    }

    /// Creates an augmentor with a reproducible `seed`.
    #[must_use]
    pub const fn with_seed(config: AugmentationConfig, seed: u64) -> Self {
        Self {
            config,
            seeds: SeedSequence::new(seed),
        }
    }

    /// Returns the configuration.
    #[must_use]
    pub const fn config(&self) -> &AugmentationConfig {
        &self.config
    }

    /// Applies the configured transforms; the output size may differ from the input after a
    /// crop, but image and mask always keep matching sizes.
    #[must_use]
    pub fn augment(&self, image: RgbImage, mask: GrayImage) -> (RgbImage, GrayImage) {
        let mut rng = self.seeds.next_rng();
        let (mut image, mut mask) = (image, mask);
        if self.config.background_synthesis {
            synthesize_background(&mut image, &mask, &mut rng);
        }
        for method in &self.config.methods {
            match method {
                AugmentationMethod::Flip => {
                    if rng.random_bool(0.5) {
                        imageops::flip_horizontal_in_place(&mut image);
                        imageops::flip_horizontal_in_place(&mut mask);
                    }
                }
                AugmentationMethod::Crop => (image, mask) = random_crop(image, mask, &mut rng),
                AugmentationMethod::Rotate => {
                    if rng.random_bool(ROTATE_PROBABILITY) {
                        let angle = rng
                            .random_range(-ROTATE_MAX_DEGREES..=ROTATE_MAX_DEGREES)
                            .to_radians();
                        image = rotate_about_center_rgb8(&image, angle);
                        mask = rotate_about_center_luma8(&mask, angle);
                    }
                }
                AugmentationMethod::Enhance => image = enhance_colors(image, &mut rng),
                AugmentationMethod::Pepper => add_pepper_noise(&mut image, &mut rng),
            }
        }
        (image, mask)
    }
}

fn random_crop(image: RgbImage, mask: GrayImage, rng: &mut impl Rng) -> (RgbImage, GrayImage) {
    let (width, height) = image.dimensions();
    // Up to 10% of each dimension is removed from either side.
    let min_side = |dim: u32| dim.saturating_sub(2 * (dim / 10)).max(1);
    let (min_w, min_h) = (min_side(width), min_side(height));
    if width <= 2 || height <= 2 || (min_w >= width && min_h >= height) {
        return (image, mask);
    }

    let crop_w = rng.random_range(min_w..=width);
    let crop_h = rng.random_range(min_h..=height);
    let x = rng.random_range(0..=width - crop_w);
    let y = rng.random_range(0..=height - crop_h);
    (
        imageops::crop_imm(&image, x, y, crop_w, crop_h).to_image(),
        imageops::crop_imm(&mask, x, y, crop_w, crop_h).to_image(),
    )
}

/// Random brightness (×0.5–1.5), contrast (×0.5–1.5), hue (±180°), and sharpness (0–3).
fn enhance_colors(mut image: RgbImage, rng: &mut impl Rng) -> RgbImage {
    let brightness: f32 = rng.random_range(0.5..=1.5);
    colorops::brighten_in_place(&mut image, round_i32((brightness - 1.0) * 255.0));

    let contrast: f32 = rng.random_range(0.5..=1.5);
    colorops::contrast_in_place(&mut image, (contrast - 1.0) * 100.0);

    let hue: f32 = rng.random_range(0.0..=2.0);
    colorops::huerotate_in_place(&mut image, round_i32((hue - 1.0) * 180.0));

    let sharpness: f32 = rng.random_range(0.0..=3.0);
    if sharpness > 1.0 {
        sharpen_rgb(&image, sharpness - 1.0)
    } else if sharpness < 1.0 {
        imageops::blur(&image, (1.0 - sharpness) * 2.0)
    } else {
        image
    }
}

fn round_i32(value: f32) -> i32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "brightness and hue offsets are within ±255"
    )]
    let rounded = value.round() as i32;
    rounded
}

/// Sets `PEPPER_DENSITY` of the pixels to black or white.
///
/// Gaps between noisy pixels are drawn from the geometric distribution, so the generator runs
/// once per noisy pixel instead of once per pixel.
fn add_pepper_noise(image: &mut RgbImage, rng: &mut impl Rng) {
    let pixels = image.as_mut();
    let count = pixels.len() / 3;
    let log_keep = (1.0 - PEPPER_DENSITY).ln();
    let mut idx = 0_usize;
    loop {
        let uniform: f64 = rng.random_range(f64::MIN_POSITIVE..1.0);
        let gap = (uniform.ln() / log_keep).floor();
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "the gap is a non-negative float; huge gaps saturate and end the loop"
        )]
        let gap = gap as usize;
        idx = idx.saturating_add(gap);
        if idx >= count {
            break;
        }
        let value = if rng.random_bool(0.5) { 0 } else { 255 };
        pixels[3 * idx..3 * idx + 3].fill(value);
        idx += 1;
    }
}

/// Replaces background pixels (mask `<= 128`) with a solid, foreground-like, or random color.
fn random_color(rng: &mut impl Rng) -> Rgb<u8> {
    Rgb([rng.random(), rng.random(), rng.random()])
}

fn synthesize_background(image: &mut RgbImage, mask: &GrayImage, rng: &mut impl Rng) {
    let color = match rng.random_range(0..10) {
        // 40%: black, gray, or white.
        0..=3 => Rgb([[0, 128, 255][rng.random_range(0..3)]; 3]),
        // 40%: a jittered foreground color.
        4..=7 => similar_color(image, mask, rng).unwrap_or_else(|| random_color(rng)),
        // 20%: uniformly random.
        _ => random_color(rng),
    };
    for (pixel, alpha) in image.pixels_mut().zip(mask.pixels()) {
        if alpha.0[0] <= FOREGROUND_THRESHOLD {
            *pixel = color;
        }
    }
}

/// A uniformly chosen foreground pixel jittered by ±30 per channel, or `None` without foreground.
fn similar_color(image: &RgbImage, mask: &GrayImage, rng: &mut impl Rng) -> Option<Rgb<u8>> {
    let is_fg = |alpha: &image::Luma<u8>| alpha.0[0] > FOREGROUND_THRESHOLD;
    let fg_count = mask.pixels().filter(|alpha| is_fg(alpha)).count();
    if fg_count == 0 {
        return None;
    }
    let target = rng.random_range(0..fg_count);
    let (base, _) = image
        .pixels()
        .zip(mask.pixels())
        .filter(|(_, alpha)| is_fg(alpha))
        .nth(target)?;
    let mut jitter = |channel: u8| {
        let shifted =
            i32::from(channel) + rng.random_range(-SIMILAR_COLOR_JITTER..=SIMILAR_COLOR_JITTER);
        u8::try_from(shifted.clamp(0, 255)).unwrap_or(u8::MAX)
    };
    Some(Rgb(base.0.map(&mut jitter)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pepper_noise_density_matches_configuration() {
        let mut image = RgbImage::from_pixel(400, 250, Rgb([100, 100, 100]));
        let mut rng = SmallRng::seed_from_u64(3);
        add_pepper_noise(&mut image, &mut rng);
        let noisy = image.pixels().filter(|p| p.0 != [100; 3]).count();
        let expected = 100_000.0 * PEPPER_DENSITY;
        let noisy = f64::from(u32::try_from(noisy).unwrap());
        // Binomial standard deviation is ~12; allow 5 sigma.
        assert!((noisy - expected).abs() < 60.0, "{noisy} noisy pixels");
        assert!(
            image
                .pixels()
                .all(|p| [[0; 3], [255; 3], [100; 3]].contains(&p.0))
        );
    }

    #[test]
    fn similar_color_samples_only_foreground_pixels() {
        let mut image = RgbImage::from_pixel(8, 8, Rgb([0, 0, 0]));
        let mut mask = GrayImage::new(8, 8);
        image.put_pixel(5, 6, Rgb([200, 100, 50]));
        mask.put_pixel(5, 6, image::Luma([255]));
        let mut rng = SmallRng::seed_from_u64(1);
        for _ in 0..20 {
            let color = similar_color(&image, &mask, &mut rng).unwrap();
            assert!(color.0[0].abs_diff(200) <= 30 && color.0[2].abs_diff(50) <= 30);
        }
        assert!(similar_color(&image, &GrayImage::new(8, 8), &mut rng).is_none());
    }
}
