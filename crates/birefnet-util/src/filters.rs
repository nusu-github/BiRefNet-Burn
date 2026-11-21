//! Spatial filtering and MATLAB-compatible Gaussian convolution utilities.

use std::f64::consts::PI;

use burn::tensor::{
    Device, Shape, Tensor, TensorData,
    module::conv2d,
    ops::{ConvOptions, PadMode},
};

fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("kernel dimension exceeds u32::MAX"))
}

/// Generates a normalized 2D Gaussian kernel matching MATLAB `fspecial('gaussian', shape, sigma)`.
///
/// # Panics
///
/// Panics if either kernel dimension is even or zero, or if `sigma` is not finite and positive.
#[must_use]
pub fn matlab_gaussian_2d(shape: (usize, usize), sigma: f64, device: &Device) -> Tensor<2> {
    let (height, width) = shape;
    assert!(
        height > 0 && width > 0 && height % 2 == 1 && width % 2 == 1,
        "matlab_gaussian_2d kernel dimensions must be positive and odd, got {height}x{width}"
    );
    assert!(
        sigma.is_finite() && sigma > 0.0,
        "matlab_gaussian_2d sigma must be finite and positive, got {sigma}"
    );

    let half_y = usize_to_f64(height - 1) / 2.0;
    let half_x = usize_to_f64(width - 1) / 2.0;
    let two_sigma_sq = 2.0 * sigma * sigma;

    let mut raw = Vec::with_capacity(height * width);
    let mut max_val = 0.0_f64;
    for row in 0..height {
        let dy = usize_to_f64(row) - half_y;
        for col in 0..width {
            let dx = usize_to_f64(col) - half_x;
            let val = (-dx.mul_add(dx, dy * dy) / two_sigma_sq).exp();
            max_val = max_val.max(val);
            raw.push(val);
        }
    }

    let cutoff = f64::EPSILON * max_val;
    let mut sum_val = 0.0_f64;
    for val in &mut raw {
        if *val < cutoff {
            *val = 0.0;
        }
        sum_val += *val;
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "normalized Gaussian weights are strictly inside [0, 1]"
    )]
    let normalized: Vec<f32> = raw
        .into_iter()
        .map(|val| {
            (if sum_val > f64::EPSILON {
                val / sum_val
            } else {
                val
            }) as f32
        })
        .collect();

    Tensor::<2>::from_data(
        TensorData::new(normalized, Shape::new([height, width])),
        device,
    )
}

/// Generates a square normalized 2D Gaussian kernel `[size, size]`.
///
/// # Panics
///
/// Panics if `size` is even or zero, or if `sigma` is not finite and positive.
#[must_use]
pub fn gaussian_kernel(size: usize, sigma: f64, device: &Device) -> Tensor<2> {
    assert!(
        size > 0 && size % 2 == 1,
        "gaussian_kernel size must be positive and odd, got {size}"
    );
    assert!(
        sigma.is_finite() && sigma > 0.0,
        "gaussian_kernel sigma must be finite and positive, got {sigma}"
    );

    let center = usize_to_f64(size - 1) / 2.0;
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut raw = Vec::with_capacity(size * size);
    let mut sum_val = 0.0_f64;

    for row in 0..size {
        let dy = usize_to_f64(row) - center;
        for col in 0..size {
            let dx = usize_to_f64(col) - center;
            let val = (-dx.mul_add(dx, dy * dy) / two_sigma_sq).exp() / (two_sigma_sq * PI);
            sum_val += val;
            raw.push(val);
        }
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "normalized Gaussian weights are strictly inside [0, 1]"
    )]
    let normalized: Vec<f32> = raw
        .into_iter()
        .map(|val| {
            (if sum_val > f64::EPSILON {
                val / sum_val
            } else {
                val
            }) as f32
        })
        .collect();

    Tensor::<2>::from_data(
        TensorData::new(normalized, Shape::new([size, size])),
        device,
    )
}

/// Applies a 2D spatial correlation kernel `[kH, kW]` channel-wise with edge-replicated borders.
///
/// # Panics
///
/// Panics if `kernel` has zero or even dimensions.
#[must_use]
pub fn custom_filter(image: Tensor<4>, kernel: Tensor<2>) -> Tensor<4> {
    let [_, channels, _, _] = image.dims();
    let [kernel_h, kernel_w] = kernel.dims();
    assert!(
        kernel_h > 0 && kernel_w > 0 && kernel_h % 2 == 1 && kernel_w % 2 == 1,
        "custom_filter kernel dimensions must be positive and odd, got {kernel_h}x{kernel_w}"
    );

    let pad_h = kernel_h / 2;
    let pad_w = kernel_w / 2;
    let padded = if pad_h == 0 && pad_w == 0 {
        image
    } else {
        image.pad(
            [(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)],
            PadMode::Edge,
        )
    };

    let grouped_kernel = kernel.unsqueeze::<4>().repeat(&[channels, 1, 1, 1]);
    conv2d(
        padded,
        grouped_kernel,
        None,
        ConvOptions::new([1, 1], [0, 0], [1, 1], channels),
    )
}

/// Filters a `[B, C, H, W]` tensor with a MATLAB-style 2D Gaussian kernel.
#[must_use]
pub fn gaussian_filter_matlab(
    image: Tensor<4>,
    kernel_shape: (usize, usize),
    sigma: f64,
) -> Tensor<4> {
    let device = image.device();
    let kernel = matlab_gaussian_2d(kernel_shape, sigma, &device);
    custom_filter(image, kernel)
}
