//! Mask postprocessing, morphology, connected components, and tensor resizing helpers.

use std::collections::VecDeque;

use birefnet_util::{StructuringElement, closing, gaussian_filter_matlab, opening};
use burn::tensor::{
    Shape, Tensor, TensorData,
    module::interpolate,
    ops::{InterpolateMode, InterpolateOptions},
};
use image::GrayImage;

use crate::error::InferenceError;

/// Configuration for [`postprocess_mask`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PostprocessConfig {
    /// Optional binarization threshold in `[0.0, 1.0]`.
    pub threshold: Option<f64>,
    /// Optional Gaussian smoothing `(kernel_size, sigma)`.
    pub blur_kernel: Option<(usize, f64)>,
    /// Optional morphological opening disk radius in pixels.
    pub opening_radius: Option<usize>,
    /// Optional morphological closing disk radius in pixels.
    pub closing_radius: Option<usize>,
    /// Optional minimum 4-connected foreground component area in pixels.
    pub min_component_size: Option<usize>,
    /// Whether to fill enclosed interior holes inside foreground regions.
    pub fill_holes: bool,
}

/// Applies element-wise thresholding (`mask > threshold`) producing `0.0` / `1.0`.
#[must_use]
pub fn apply_threshold(mask: Tensor<4>, threshold: f64) -> Tensor<4> {
    mask.greater_elem(threshold).float()
}

/// Smooths a mask `[B, C, H, W]` with a 2D Gaussian kernel (`sigma > 0`, `kernel_size > 1`).
///
/// Even `kernel_size` values are rounded up to the next odd integer. Returns `mask` unchanged
/// if `kernel_size <= 1` or `sigma <= 0.0` (or non-finite).
#[must_use]
pub fn gaussian_blur(mask: Tensor<4>, kernel_size: usize, sigma: f64) -> Tensor<4> {
    if kernel_size <= 1 || sigma <= 0.0 || !sigma.is_finite() {
        return mask;
    }
    let odd_size = kernel_size | 1;
    gaussian_filter_matlab(mask, (odd_size, odd_size), sigma)
}

/// Applies morphological opening with a circular disk structuring element of `radius`.
#[must_use]
pub fn morphological_opening(mask: Tensor<4>, radius: usize) -> Tensor<4> {
    if radius == 0 {
        return mask;
    }
    let device = mask.device();
    let kernel = StructuringElement::disk(radius, &device);
    opening(mask, &kernel)
}

/// Applies morphological closing with a circular disk structuring element of `radius`.
#[must_use]
pub fn morphological_closing(mask: Tensor<4>, radius: usize) -> Tensor<4> {
    if radius == 0 {
        return mask;
    }
    let device = mask.device();
    let kernel = StructuringElement::disk(radius, &device);
    closing(mask, &kernel)
}

fn push_neighbors_4(pos: usize, width: usize, height: usize, enqueue: &mut impl FnMut(usize)) {
    let x = pos % width;
    let y = pos / width;
    if y > 0 {
        enqueue(pos - width);
    }
    if y + 1 < height {
        enqueue(pos + width);
    }
    if x > 0 {
        enqueue(pos - 1);
    }
    if x + 1 < width {
        enqueue(pos + 1);
    }
}

/// Removes 4-connected foreground (`>= 0.5`) components strictly smaller than `min_size` pixels.
#[must_use]
pub fn remove_small_components(mask: Tensor<4>, min_size: usize) -> Tensor<4> {
    if min_size <= 1 {
        return mask;
    }

    let [batch, channels, height, width] = mask.dims();
    if height == 0 || width == 0 {
        return mask;
    }

    let device = mask.device();
    let plane_len = height * width;
    let mut values: Vec<f32> = mask.to_data().iter::<f32>().collect();
    let mut visited = vec![false; plane_len];
    let mut queue = VecDeque::new();
    let mut component = Vec::new();

    for plane in values.chunks_exact_mut(plane_len) {
        visited.fill(false);

        for start in 0..plane_len {
            if visited[start] || plane[start] < 0.5 {
                continue;
            }
            visited[start] = true;
            queue.push_back(start);

            while let Some(pos) = queue.pop_front() {
                component.push(pos);
                push_neighbors_4(pos, width, height, &mut |next| {
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

    Tensor::<4>::from_data(
        TensorData::new(values, Shape::new([batch, channels, height, width])),
        &device,
    )
}

/// Fills enclosed background (`< 0.5`) holes surrounded by foreground pixels (`>= 0.5`).
#[must_use]
pub fn fill_holes(mask: Tensor<4>) -> Tensor<4> {
    let [batch, channels, height, width] = mask.dims();
    if height < 3 || width < 3 {
        return mask;
    }

    let device = mask.device();
    let plane_len = height * width;
    let mut values: Vec<f32> = mask.to_data().iter::<f32>().collect();
    let mut reachable_bg = vec![false; plane_len];
    let mut queue = VecDeque::new();

    for plane in values.chunks_exact_mut(plane_len) {
        reachable_bg.fill(false);

        for x in 0..width {
            let top = x;
            let bottom = (height - 1) * width + x;
            for idx in [top, bottom] {
                if !reachable_bg[idx] && plane[idx] < 0.5 {
                    reachable_bg[idx] = true;
                    queue.push_back(idx);
                }
            }
        }
        for y in 0..height {
            let left = y * width;
            let right = y * width + (width - 1);
            for idx in [left, right] {
                if !reachable_bg[idx] && plane[idx] < 0.5 {
                    reachable_bg[idx] = true;
                    queue.push_back(idx);
                }
            }
        }

        while let Some(pos) = queue.pop_front() {
            push_neighbors_4(pos, width, height, &mut |next| {
                if !reachable_bg[next] && plane[next] < 0.5 {
                    reachable_bg[next] = true;
                    queue.push_back(next);
                }
            });
        }

        for idx in 0..plane_len {
            if plane[idx] < 0.5 && !reachable_bg[idx] {
                plane[idx] = 1.0;
            }
        }
    }

    Tensor::<4>::from_data(
        TensorData::new(values, Shape::new([batch, channels, height, width])),
        &device,
    )
}

/// Runs the configured postprocessing steps sequentially on a `[B, C, H, W]` mask.
#[must_use]
pub fn postprocess_mask(mask: Tensor<4>, config: &PostprocessConfig) -> Tensor<4> {
    let mut processed = mask;

    if let Some(threshold) = config.threshold {
        processed = apply_threshold(processed, threshold);
    }
    if let Some((kernel_size, sigma)) = config.blur_kernel {
        processed = gaussian_blur(processed, kernel_size, sigma);
    }
    if let Some(radius) = config.opening_radius {
        processed = morphological_opening(processed, radius);
    }
    if let Some(radius) = config.closing_radius {
        processed = morphological_closing(processed, radius);
    }
    if let Some(min_size) = config.min_component_size {
        processed = remove_small_components(processed, min_size);
    }
    if config.fill_holes {
        processed = fill_holes(processed);
    }

    processed
}

/// Resizes a `[B, C, H, W]` tensor directly via native Burn spatial interpolation.
#[must_use]
pub fn resize_tensor(
    tensor: Tensor<4>,
    target_height: usize,
    target_width: usize,
    mode: InterpolateMode,
) -> Tensor<4> {
    let [_batch, _channels, current_height, current_width] = tensor.dims();
    if current_height == target_height && current_width == target_width {
        return tensor;
    }

    interpolate(
        tensor,
        InterpolateOptions::new(mode).with_output_size([target_height, target_width]),
    )
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "normalized values are clamped into [0.0, 1.0] and scaled into 0..=255"
)]
fn to_u8_pixel(val: f64) -> u8 {
    (val.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Converts a single `[1, 1, H, W]` normalized mask tensor into an 8-bit [`GrayImage`].
///
/// # Errors
///
/// Returns [`InferenceError`] if `mask` shape is not `[1, 1, H, W]` or exceeds `u32` dimensions.
pub fn mask_to_u8(mask: &Tensor<4>) -> Result<GrayImage, InferenceError> {
    let [batch, channels, height, width] = mask.dims();
    if batch != 1 || channels != 1 {
        return Err(InferenceError::invalid_input(format!(
            "mask_to_u8 expects [1, 1, H, W] tensor, got [{batch}, {channels}, {height}, {width}]"
        )));
    }

    let w_u32 = u32::try_from(width)
        .map_err(|_err| InferenceError::invalid_input(format!("width {width} exceeds u32::MAX")))?;
    let h_u32 = u32::try_from(height).map_err(|_err| {
        InferenceError::invalid_input(format!("height {height} exceeds u32::MAX"))
    })?;

    let bytes: Vec<u8> = mask.to_data().iter::<f64>().map(to_u8_pixel).collect();
    GrayImage::from_raw(w_u32, h_u32, bytes).ok_or_else(|| {
        InferenceError::invalid_input(format!(
            "mask buffer length mismatch for [{height}, {width}] grayscale image"
        ))
    })
}
