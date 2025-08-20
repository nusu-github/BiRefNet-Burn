//! Filtering operations for computer vision
//!
//! This module provides filtering and convolution operations required for implementing
//! evaluation metrics like WeightedFMeasure that depend on spatial filtering.

use std::f64::consts::PI;

use burn::tensor::{
    backend::Backend, module::conv2d, ops::ConvOptions, s, ElementConversion, Tensor,
};

/// MATLAB-style 2D Gaussian kernel
///
/// Creates a 2D Gaussian kernel that matches MATLAB's fspecial('gaussian', shape, sigma).
/// This is specifically required for the WeightedFMeasure metric.
///
/// # Arguments
/// * `shape` - Kernel shape as (height, width)
/// * `sigma` - Standard deviation of the Gaussian
/// * `device` - Device to create the kernel on
///
/// # Returns
/// Normalized 2D Gaussian kernel tensor [height, width]
pub fn matlab_gaussian_2d<B: Backend>(
    shape: (usize, usize),
    sigma: f64,
    device: &B::Device,
) -> Tensor<B, 2> {
    let (height, width) = shape;

    // Calculate half-sizes
    let m = (height - 1) as f64 / 2.0;
    let n = (width - 1) as f64 / 2.0;

    // Create coordinate grids
    let mut kernel_data = Vec::with_capacity(height * width);

    for i in 0..height {
        for j in 0..width {
            let y = i as f64 - m;
            let x = j as f64 - n;

            // Gaussian function: exp(-(x^2 + y^2) / (2 * sigma^2))
            let value = (-x.mul_add(x, y * y) / (2.0 * sigma * sigma)).exp();
            kernel_data.push(value);
        }
    }

    // Create tensor
    let mut kernel =
        Tensor::<B, 1>::from_floats(kernel_data.as_slice(), device).reshape([height, width]);

    // Apply MATLAB's epsilon threshold: h[h < eps * max(h)] = 0
    let max_val = kernel.clone().max().into_scalar().elem::<f64>();
    let eps = f64::EPSILON;
    let threshold = eps * max_val;

    // Create mask and apply threshold
    let threshold_tensor = Tensor::<B, 2>::ones_like(&kernel) * threshold;
    let mask = kernel.clone().lower(threshold_tensor);
    kernel = kernel.mask_fill(mask, 0.0);

    // Normalize (divide by sum)
    let sum_val = kernel.clone().sum().into_scalar().elem::<f64>();
    if sum_val > eps {
        kernel = kernel / sum_val;
    }

    kernel
}

/// Standard 2D Gaussian kernel
///
/// Creates a standard 2D Gaussian kernel for general filtering purposes.
///
/// # Arguments
/// * `size` - Kernel size (will create size x size kernel)
/// * `sigma` - Standard deviation of the Gaussian
/// * `device` - Device to create the kernel on
///
/// # Returns
/// Normalized 2D Gaussian kernel tensor [size, size]
pub fn gaussian_kernel<B: Backend>(size: usize, sigma: f64, device: &B::Device) -> Tensor<B, 2> {
    assert!(size > 0 && size % 2 == 1, "Kernel size must be odd and > 0");

    let center = (size - 1) as f64 / 2.0;
    let mut kernel_data = Vec::with_capacity(size * size);

    let two_sigma_squared = 2.0 * sigma * sigma;

    for i in 0..size {
        for j in 0..size {
            let y = i as f64 - center;
            let x = j as f64 - center;

            let value = (-x.mul_add(x, y * y) / two_sigma_squared).exp() / (two_sigma_squared * PI);
            kernel_data.push(value);
        }
    }

    let kernel = Tensor::<B, 1>::from_floats(kernel_data.as_slice(), device).reshape([size, size]);

    // Normalize
    let sum_val = kernel.clone().sum().into_scalar().elem::<f64>();
    if sum_val > f64::EPSILON {
        kernel / sum_val
    } else {
        kernel
    }
}

/// Apply 2D convolution with custom kernel
///
/// # Arguments
/// * `image` - Input image tensor [B, C, H, W]
/// * `kernel` - Convolution kernel [kernel_h, kernel_w]
///
/// # Returns
/// Filtered image tensor [B, C, H, W]
pub fn custom_filter<B: Backend>(image: Tensor<B, 4>, kernel: Tensor<B, 2>) -> Tensor<B, 4> {
    let [batch_size, channels, height, width] = image.dims();
    let [kernel_h, kernel_w] = kernel.dims();
    let device = image.device();

    // Add necessary dimensions to kernel for conv2d: [out_channels, in_channels, kernel_h, kernel_w]
    let conv_kernel = kernel.unsqueeze::<3>().unsqueeze::<4>();

    // Apply padding to maintain output size
    let pad_h = kernel_h / 2;
    let pad_w = kernel_w / 2;

    let mut result = Tensor::<B, 4>::zeros([batch_size, channels, height, width], &device);

    // Process each channel separately due to kernel shape requirements
    for c in 0..channels {
        let channel_image = image.clone().slice(s![.., c..c + 1, .., ..]);

        // Apply padding using existing replicate padding from foreground_refiner
        let padded = pad_replicate_for_conv(channel_image, pad_h, pad_w);

        // Apply convolution with proper options
        let options = ConvOptions::new([1, 1], [0, 0], [1, 1], 1);
        let filtered = conv2d(
            padded,
            conv_kernel.clone(),
            None, // no bias
            options,
        );

        // Assign result back
        result = result.slice_assign(s![.., c..c + 1, .., ..], filtered);
    }

    result
}

/// Gaussian filtering with MATLAB-style kernel
///
/// Applies Gaussian filtering using the MATLAB-compatible kernel.
/// This is specifically designed for WeightedFMeasure metric calculations.
///
/// # Arguments
/// * `image` - Input image tensor [B, C, H, W]
/// * `kernel_shape` - Shape of Gaussian kernel (height, width)
/// * `sigma` - Standard deviation of Gaussian
///
/// # Returns
/// Filtered image tensor [B, C, H, W]
pub fn gaussian_filter_matlab<B: Backend>(
    image: Tensor<B, 4>,
    kernel_shape: (usize, usize),
    sigma: f64,
) -> Tensor<B, 4> {
    let device = image.device();
    let kernel = matlab_gaussian_2d(kernel_shape, sigma, &device);
    custom_filter(image, kernel)
}

/// Replicate padding for convolution operations
fn pad_replicate_for_conv<B: Backend>(
    tensor: Tensor<B, 4>,
    pad_h: usize,
    pad_w: usize,
) -> Tensor<B, 4> {
    let [batch_size, channels, height, width] = tensor.dims();

    if pad_h == 0 && pad_w == 0 {
        return tensor;
    }

    let new_height = height + 2 * pad_h;
    let new_width = width + 2 * pad_w;

    let device = tensor.device();
    let mut result = Tensor::<B, 4>::zeros([batch_size, channels, new_height, new_width], &device);

    // Copy original tensor to center
    result = result.slice_assign(
        s![.., .., pad_h..pad_h + height, pad_w..pad_w + width],
        tensor.clone(),
    );

    // Replicate edges
    if pad_h > 0 {
        let top_row = tensor.clone().slice(s![.., .., 0..1, ..]);
        let bottom_row = tensor.slice(s![.., .., height - 1..height, ..]);

        for i in 0..pad_h {
            result =
                result.slice_assign(s![.., .., i..i + 1, pad_w..pad_w + width], top_row.clone());
            result = result.slice_assign(
                s![
                    ..,
                    ..,
                    pad_h + height + i..pad_h + height + i + 1,
                    pad_w..pad_w + width
                ],
                bottom_row.clone(),
            );
        }
    }

    if pad_w > 0 {
        let left_col = result.clone().slice(s![.., .., .., pad_w..pad_w + 1]);
        let right_col = result
            .clone()
            .slice(s![.., .., .., pad_w + width - 1..pad_w + width]);

        for i in 0..pad_w {
            result = result.slice_assign(s![.., .., .., i..i + 1], left_col.clone());
            result = result.slice_assign(
                s![.., .., .., pad_w + width + i..pad_w + width + i + 1],
                right_col.clone(),
            );
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    use super::*;

    type TestBackend = NdArray;

    #[test]
    fn matlab_gaussian_2d_creates_normalized_kernel() {
        let device = Default::default();
        let kernel = matlab_gaussian_2d::<TestBackend>((7, 7), 5.0, &device);

        let [h, w] = kernel.dims();
        assert_eq!(h, 7);
        assert_eq!(w, 7);

        // Check normalization (sum should be approximately 1)
        let sum_val = kernel.clone().sum().into_scalar().elem::<f32>();
        assert!((sum_val - 1.0).abs() < 1e-4);

        // Center value should be the maximum
        let center_val = kernel
            .clone()
            .slice(s![3..4, 3..4])
            .into_scalar()
            .elem::<f32>();
        let max_val = kernel.clone().max().into_scalar().elem::<f32>();
        assert!((center_val - max_val).abs() < 1e-6);
    }

    #[test]
    fn gaussian_kernel_creates_symmetric_kernel() {
        let device = Default::default();
        let kernel = gaussian_kernel::<TestBackend>(5, 1.0, &device);

        let [h, w] = kernel.dims();
        assert_eq!(h, 5);
        assert_eq!(w, 5);

        // Check symmetry
        let top_left = kernel
            .clone()
            .slice(s![0..1, 0..1])
            .into_scalar()
            .elem::<f32>();
        let top_right = kernel
            .clone()
            .slice(s![0..1, 4..5])
            .into_scalar()
            .elem::<f32>();
        let bottom_left = kernel
            .clone()
            .slice(s![4..5, 0..1])
            .into_scalar()
            .elem::<f32>();
        let bottom_right = kernel
            .clone()
            .slice(s![4..5, 4..5])
            .into_scalar()
            .elem::<f32>();

        assert!((top_left - top_right).abs() < 1e-6);
        assert!((top_left - bottom_left).abs() < 1e-6);
        assert!((top_left - bottom_right).abs() < 1e-6);
    }

    #[test]
    fn custom_filter_preserves_image_dimensions() {
        let device = Default::default();
        let image = Tensor::<TestBackend, 4>::ones([1, 1, 10, 10], &device);
        let kernel = gaussian_kernel(3, 1.0, &device);

        let filtered = custom_filter(image, kernel);
        let [b, c, h, w] = filtered.dims();
        assert_eq!([b, c, h, w], [1, 1, 10, 10]);
    }

    #[test]
    fn gaussian_filter_matlab_matches_manual_kernel() {
        let device = Default::default();
        let image = Tensor::<TestBackend, 4>::ones([1, 1, 8, 8], &device);

        // Compare direct kernel application vs convenience function
        let kernel = matlab_gaussian_2d((5, 5), 2.0, &device);
        let manual_filtered = custom_filter(image.clone(), kernel);

        let convenience_filtered = gaussian_filter_matlab(image, (5, 5), 2.0);

        // Results should be identical
        let diff = (manual_filtered - convenience_filtered)
            .abs()
            .sum()
            .into_scalar()
            .elem::<f32>();
        assert!(diff < 1e-6);
    }

    #[test]
    fn matlab_gaussian_matches_expected_values() {
        let device = Default::default();

        // Test with known parameters from WeightedFMeasure
        let kernel = matlab_gaussian_2d::<TestBackend>((7, 7), 5.0, &device);

        // Check properties
        let sum_val = kernel.clone().sum().into_scalar().elem::<f32>();
        assert!((sum_val - 1.0).abs() < 1e-4); // Should be normalized

        let center_val = kernel
            .clone()
            .slice(s![3..4, 3..4])
            .into_scalar()
            .elem::<f32>();
        assert!(center_val > 0.0); // Center should be positive

        // Edge values should be smaller than center
        let edge_val = kernel
            .clone()
            .slice(s![0..1, 3..4])
            .into_scalar()
            .elem::<f32>();
        assert!(edge_val < center_val);
    }
}
