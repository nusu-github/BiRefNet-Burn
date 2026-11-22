//! Mask postprocessing: thresholding, smoothing, morphology, and connected-component cleanup.
//!
//! Device steps keep the mask's float dtype. Connected-component steps run on the host and
//! share a single readback when both are enabled.

use std::collections::VecDeque;

use birefnet_util::{
    filters::gaussian_filter_matlab,
    morphology::{StructuringElement, closing, opening},
};
use burn::tensor::{
    Shape, Tensor, TensorData,
    module::interpolate,
    ops::{InterpolateMode, InterpolateOptions},
};
use image::GrayImage;

use crate::error::InferenceError;

/// Gaussian smoothing parameters: an odd kernel side length and a positive `sigma`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianBlur {
    kernel_size: usize,
    sigma: f64,
}

impl GaussianBlur {
    /// Creates blur parameters.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] unless `kernel_size` is odd and greater than 1 and `sigma` is
    /// finite and positive.
    pub fn new(kernel_size: usize, sigma: f64) -> Result<Self, InferenceError> {
        if kernel_size <= 1 || kernel_size.is_multiple_of(2) || !(sigma.is_finite() && sigma > 0.0)
        {
            return Err(InferenceError::invalid_config(format!(
                "Gaussian blur needs an odd kernel size > 1 and a positive sigma, got \
                 {kernel_size} and {sigma}"
            )));
        }
        Ok(Self { kernel_size, sigma })
    }

    /// Returns the kernel side length.
    #[must_use]
    pub const fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Returns the Gaussian standard deviation.
    #[must_use]
    pub const fn sigma(&self) -> f64 {
        self.sigma
    }
}

/// Postprocessing pipeline for predicted masks; every step is off by default.
///
/// Steps run in the order threshold, blur, opening, closing, small-component removal, and
/// hole filling.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PostprocessConfig {
    threshold: Option<f64>,
    blur: Option<GaussianBlur>,
    opening_radius: Option<usize>,
    closing_radius: Option<usize>,
    min_component_size: Option<usize>,
    fill_holes: bool,
}

impl PostprocessConfig {
    /// Returns a configuration with every step disabled.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            threshold: None,
            blur: None,
            opening_radius: None,
            closing_radius: None,
            min_component_size: None,
            fill_holes: false,
        }
    }

    /// Binarizes the mask with `mask > threshold`.
    ///
    /// # Errors
    ///
    /// Returns [`InferenceError`] unless `threshold` lies in `[0, 1]`.
    pub fn with_threshold(self, threshold: f64) -> Result<Self, InferenceError> {
        if !(0.0..=1.0).contains(&threshold) {
            return Err(InferenceError::invalid_config(format!(
                "threshold must be in [0, 1], got {threshold}"
            )));
        }
        Ok(Self {
            threshold: Some(threshold),
            ..self
        })
    }

    /// Smooths the mask with a MATLAB-style Gaussian.
    #[must_use]
    pub const fn with_blur(self, blur: GaussianBlur) -> Self {
        Self {
            blur: Some(blur),
            ..self
        }
    }

    /// Applies morphological opening with a disk of `radius` pixels (`0` disables it).
    #[must_use]
    pub const fn with_opening(self, radius: usize) -> Self {
        Self {
            opening_radius: if radius == 0 { None } else { Some(radius) },
            ..self
        }
    }

    /// Applies morphological closing with a disk of `radius` pixels (`0` disables it).
    #[must_use]
    pub const fn with_closing(self, radius: usize) -> Self {
        Self {
            closing_radius: if radius == 0 { None } else { Some(radius) },
            ..self
        }
    }

    /// Removes 4-connected components smaller than `min_size` pixels (`<= 1` disables it).
    #[must_use]
    pub const fn with_min_component_size(self, min_size: usize) -> Self {
        Self {
            min_component_size: if min_size <= 1 { None } else { Some(min_size) },
            ..self
        }
    }

    /// Fills background holes not connected to the image border.
    #[must_use]
    pub const fn with_fill_holes(self, fill_holes: bool) -> Self {
        Self { fill_holes, ..self }
    }

    /// Returns the binarization threshold, if any.
    #[must_use]
    pub const fn threshold(&self) -> Option<f64> {
        self.threshold
    }

    /// Returns the Gaussian blur parameters, if any.
    #[must_use]
    pub const fn blur(&self) -> Option<GaussianBlur> {
        self.blur
    }

    /// Returns the opening disk radius, if any.
    #[must_use]
    pub const fn opening_radius(&self) -> Option<usize> {
        self.opening_radius
    }

    /// Returns the closing disk radius, if any.
    #[must_use]
    pub const fn closing_radius(&self) -> Option<usize> {
        self.closing_radius
    }

    /// Returns the minimum kept component size, if any.
    #[must_use]
    pub const fn min_component_size(&self) -> Option<usize> {
        self.min_component_size
    }

    /// Returns `true` if enclosed holes are filled.
    #[must_use]
    pub const fn fills_holes(&self) -> bool {
        self.fill_holes
    }
}

/// Binarizes `mask` to `{0, 1}` with `mask > threshold`, keeping its float dtype.
#[must_use]
pub fn apply_threshold(mask: Tensor<4>, threshold: f64) -> Tensor<4> {
    let dtype = mask.dtype();
    mask.greater_elem(threshold).float().cast(dtype)
}

/// Smooths a `[B, C, H, W]` mask with a MATLAB-style Gaussian.
#[must_use]
pub fn gaussian_blur(mask: Tensor<4>, blur: GaussianBlur) -> Tensor<4> {
    gaussian_filter_matlab(mask, (blur.kernel_size, blur.kernel_size), blur.sigma)
}

/// Morphological opening with a disk structuring element of `radius` (`0` returns `mask`).
#[must_use]
pub fn morphological_opening(mask: Tensor<4>, radius: usize) -> Tensor<4> {
    if radius == 0 {
        return mask;
    }
    opening(mask, &StructuringElement::disk(radius))
}

/// Morphological closing with a disk structuring element of `radius` (`0` returns `mask`).
#[must_use]
pub fn morphological_closing(mask: Tensor<4>, radius: usize) -> Tensor<4> {
    if radius == 0 {
        return mask;
    }
    closing(mask, &StructuringElement::disk(radius))
}

/// Calls `visit` for the 4-connected neighbors of `pos` in a `width x height` plane.
fn for_each_neighbor_4(pos: usize, width: usize, height: usize, mut visit: impl FnMut(usize)) {
    let (x, y) = (pos % width, pos / width);
    if y > 0 {
        visit(pos - width);
    }
    if y + 1 < height {
        visit(pos + width);
    }
    if x > 0 {
        visit(pos - 1);
    }
    if x + 1 < width {
        visit(pos + 1);
    }
}

/// Clears 4-connected foreground (`>= 0.5`) components with fewer than `min_size` pixels.
fn remove_small_components_plane(plane: &mut [f32], [height, width]: [usize; 2], min_size: usize) {
    let mut visited = vec![false; plane.len()];
    let mut queue = VecDeque::with_capacity(width.max(height));
    let mut component = Vec::new();
    for start in 0..plane.len() {
        if visited[start] || plane[start] < 0.5 {
            continue;
        }
        visited[start] = true;
        queue.push_back(start);
        while let Some(pos) = queue.pop_front() {
            component.push(pos);
            for_each_neighbor_4(pos, width, height, |next| {
                if !visited[next] && plane[next] >= 0.5 {
                    visited[next] = true;
                    queue.push_back(next);
                }
            });
        }
        if component.len() < min_size {
            for &idx in &component {
                plane[idx] = 0.0;
            }
        }
        component.clear();
    }
}

/// Sets background (`< 0.5`) pixels that are not 4-connected to the border to `1.0`.
fn fill_holes_plane(plane: &mut [f32], [height, width]: [usize; 2]) {
    let mut reachable = vec![false; plane.len()];
    let mut queue = VecDeque::with_capacity(2 * (width + height));
    let border = (0..width)
        .flat_map(|x| [x, (height - 1) * width + x])
        .chain((0..height).flat_map(|y| [y * width, y * width + width - 1]));
    for idx in border {
        if !reachable[idx] && plane[idx] < 0.5 {
            reachable[idx] = true;
            queue.push_back(idx);
        }
    }
    while let Some(pos) = queue.pop_front() {
        for_each_neighbor_4(pos, width, height, |next| {
            if !reachable[next] && plane[next] < 0.5 {
                reachable[next] = true;
                queue.push_back(next);
            }
        });
    }
    for (value, &outside) in plane.iter_mut().zip(&reachable) {
        if *value < 0.5 && !outside {
            *value = 1.0;
        }
    }
}

/// Runs host-side per-plane steps with one readback and one upload in the input dtype.
fn map_planes(
    mask: Tensor<4>,
    mut step: impl FnMut(&mut [f32], [usize; 2]),
) -> Result<Tensor<4>, InferenceError> {
    let dims @ [_, _, height, width] = mask.dims();
    if height == 0 || width == 0 {
        return Ok(mask);
    }
    let options = (&mask.device(), mask.dtype());
    let mut values = mask.try_into_vec_as::<f32>()?;
    for plane in values.chunks_exact_mut(height * width) {
        step(plane, [height, width]);
    }
    Ok(Tensor::from_data(
        TensorData::new(values, Shape::new(dims)),
        options,
    ))
}

/// Removes 4-connected foreground (`>= 0.5`) components strictly smaller than `min_size` pixels.
///
/// # Errors
///
/// Returns [`InferenceError`] if the mask cannot be read back from its device.
pub fn remove_small_components(
    mask: Tensor<4>,
    min_size: usize,
) -> Result<Tensor<4>, InferenceError> {
    if min_size <= 1 {
        return Ok(mask);
    }
    map_planes(mask, |plane, dims| {
        remove_small_components_plane(plane, dims, min_size);
    })
}

/// Fills enclosed background (`< 0.5`) holes surrounded by foreground pixels.
///
/// # Errors
///
/// Returns [`InferenceError`] if the mask cannot be read back from its device.
pub fn fill_holes(mask: Tensor<4>) -> Result<Tensor<4>, InferenceError> {
    map_planes(mask, fill_holes_plane)
}

/// Runs the configured postprocessing steps on a `[B, C, H, W]` mask.
///
/// # Errors
///
/// Returns [`InferenceError`] if a host-side step cannot read the mask back.
pub fn postprocess_mask(
    mask: Tensor<4>,
    config: &PostprocessConfig,
) -> Result<Tensor<4>, InferenceError> {
    let mut processed = mask;
    if let Some(threshold) = config.threshold {
        processed = apply_threshold(processed, threshold);
    }
    if let Some(blur) = config.blur {
        processed = gaussian_blur(processed, blur);
    }
    if let Some(radius) = config.opening_radius {
        processed = morphological_opening(processed, radius);
    }
    if let Some(radius) = config.closing_radius {
        processed = morphological_closing(processed, radius);
    }
    if config.min_component_size.is_none() && !config.fill_holes {
        return Ok(processed);
    }
    map_planes(processed, |plane, dims| {
        if let Some(min_size) = config.min_component_size {
            remove_small_components_plane(plane, dims, min_size);
        }
        if config.fill_holes {
            fill_holes_plane(plane, dims);
        }
    })
}

/// Resizes a `[B, C, H, W]` tensor with native Burn interpolation (no-op at the target size).
#[must_use]
pub fn resize_tensor(
    tensor: Tensor<4>,
    target_height: usize,
    target_width: usize,
    mode: InterpolateMode,
) -> Tensor<4> {
    let [.., current_height, current_width] = tensor.dims();
    if current_height == target_height && current_width == target_width {
        return tensor;
    }
    interpolate(
        tensor,
        InterpolateOptions::new(mode).with_output_size([target_height, target_width]),
    )
}

fn to_u8_pixel(value: f32) -> u8 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value is clamped to [0, 1] and rounded into 0..=255"
    )]
    let byte = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
    byte
}

/// Converts a `[1, 1, H, W]` mask in `[0, 1]` into an 8-bit [`GrayImage`] (values are rounded).
///
/// # Errors
///
/// Returns [`InferenceError`] if `mask` is not `[1, 1, H, W]`, exceeds `u32` dimensions, or
/// cannot be read back from its device.
pub fn mask_to_u8(mask: &Tensor<4>) -> Result<GrayImage, InferenceError> {
    let [batch, channels, height, width] = mask.dims();
    if batch != 1 || channels != 1 {
        return Err(InferenceError::invalid_input(format!(
            "mask_to_u8 expects a [1, 1, H, W] tensor, got [{batch}, {channels}, {height}, {width}]"
        )));
    }
    let to_u32 = |value: usize, name: &str| {
        u32::try_from(value).map_err(|_overflow| {
            InferenceError::invalid_input(format!("mask {name} {value} exceeds u32::MAX"))
        })
    };
    let (w, h) = (to_u32(width, "width")?, to_u32(height, "height")?);
    let bytes: Vec<u8> = mask
        .try_to_vec_as::<f32>()?
        .into_iter()
        .map(to_u8_pixel)
        .collect();
    GrayImage::from_raw(w, h, bytes).ok_or_else(|| {
        InferenceError::invalid_input(format!(
            "mask buffer length does not match a {height}x{width} grayscale image"
        ))
    })
}
