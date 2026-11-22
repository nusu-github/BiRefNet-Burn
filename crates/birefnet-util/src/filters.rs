//! Spatial filtering and MATLAB-compatible Gaussian convolution utilities.

use burn::tensor::{
    Device, Shape, Tensor, TensorData,
    module::conv2d,
    ops::{ConvOptions, PadMode},
};

fn usize_to_f64(value: usize) -> f64 {
    f64::from(
        u32::try_from(value)
            .unwrap_or_else(|_| panic!("kernel dimension {value} exceeds u32::MAX")),
    )
}

fn assert_odd_kernel(name: &str, height: usize, width: usize) {
    assert!(
        height > 0 && width > 0 && height % 2 == 1 && width % 2 == 1,
        "{name} kernel dimensions must be positive and odd, got {height}x{width}"
    );
}

fn assert_sigma(name: &str, sigma: f64) {
    assert!(
        sigma.is_finite() && sigma > 0.0,
        "{name} sigma must be finite and positive, got {sigma}"
    );
}

/// Normalizes `weights` to sum to one (when the sum is non-negligible) and narrows to `f32`.
fn normalized_f32(weights: Vec<f64>) -> Vec<f32> {
    let sum: f64 = weights.iter().sum();
    let scale = if sum > f64::EPSILON { sum.recip() } else { 1.0 };
    #[expect(
        clippy::cast_possible_truncation,
        reason = "normalized Gaussian weights lie in [0, 1]"
    )]
    weights.into_iter().map(|w| (w * scale) as f32).collect()
}

/// Unnormalized 1D Gaussian taps centred on `(len - 1) / 2`.
fn gaussian_taps(len: usize, sigma: f64) -> Vec<f64> {
    let center = usize_to_f64(len - 1) / 2.0;
    let two_sigma_sq = 2.0 * sigma * sigma;
    Vec::from_fn(len, |idx| {
        let offset = usize_to_f64(idx) - center;
        (-(offset * offset) / two_sigma_sq).exp()
    })
}

/// Generates a normalized 2D Gaussian kernel matching MATLAB `fspecial('gaussian', shape, sigma)`.
///
/// Weights below `eps * max` are zeroed before normalization, as in MATLAB.
///
/// # Panics
///
/// Panics if either kernel dimension is even or zero, or if `sigma` is not finite and positive.
#[must_use]
pub fn matlab_gaussian_2d(shape: (usize, usize), sigma: f64, device: &Device) -> Tensor<2> {
    let (height, width) = shape;
    assert_odd_kernel("matlab_gaussian_2d", height, width);
    assert_sigma("matlab_gaussian_2d", sigma);

    let rows = gaussian_taps(height, sigma);
    let cols = gaussian_taps(width, sigma);
    let mut raw: Vec<f64> = rows
        .iter()
        .flat_map(|&row| cols.iter().map(move |&col| row * col))
        .collect();
    let cutoff = f64::EPSILON * raw.iter().copied().fold(0.0, f64::max);
    for val in &mut raw {
        if *val < cutoff {
            *val = 0.0;
        }
    }

    Tensor::from_data(
        TensorData::new(normalized_f32(raw), Shape::new([height, width])),
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
    assert_odd_kernel("gaussian_kernel", size, size);
    assert_sigma("gaussian_kernel", sigma);

    let taps = gaussian_taps(size, sigma);
    let raw: Vec<f64> = taps
        .iter()
        .flat_map(|&row| taps.iter().map(move |&col| row * col))
        .collect();
    Tensor::from_data(
        TensorData::new(normalized_f32(raw), Shape::new([size, size])),
        device,
    )
}

fn pad_edges(image: Tensor<4>, pad_h: usize, pad_w: usize) -> Tensor<4> {
    if pad_h == 0 && pad_w == 0 {
        image
    } else {
        image.pad(
            [(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)],
            PadMode::Edge,
        )
    }
}

/// Applies a 2D spatial correlation kernel `[kH, kW]` channel-wise with edge-replicated borders.
///
/// The kernel is cast to the image's float dtype.
///
/// # Panics
///
/// Panics if `kernel` has zero or even dimensions.
#[must_use]
pub fn custom_filter(image: Tensor<4>, kernel: Tensor<2>) -> Tensor<4> {
    let [_, channels, _, _] = image.dims();
    let [kernel_h, kernel_w] = kernel.dims();
    assert_odd_kernel("custom_filter", kernel_h, kernel_w);

    let grouped_kernel = kernel
        .cast(image.dtype())
        .unsqueeze::<4>()
        .repeat_dim(0, channels);
    conv2d(
        pad_edges(image, kernel_h / 2, kernel_w / 2),
        grouped_kernel,
        None,
        ConvOptions::new([1, 1], [0, 0], [1, 1], channels),
    )
}

/// Filters a `[B, C, H, W]` tensor with a MATLAB-style Gaussian (`fspecial` + `imfilter`
/// with `'replicate'` borders).
///
/// The Gaussian is applied as two separable 1D passes, which equals the 2D kernel up to the
/// sub-`eps` weights MATLAB truncates.
///
/// # Panics
///
/// Panics if either kernel dimension is even or zero, or if `sigma` is not finite and positive.
#[must_use]
pub fn gaussian_filter_matlab(
    image: Tensor<4>,
    kernel_shape: (usize, usize),
    sigma: f64,
) -> Tensor<4> {
    let (height, width) = kernel_shape;
    assert_odd_kernel("gaussian_filter_matlab", height, width);
    assert_sigma("gaussian_filter_matlab", sigma);

    let [_, channels, _, _] = image.dims();
    let options = (&image.device(), image.dtype());
    let taps = |len: usize, shape: [usize; 4]| {
        Tensor::<4>::from_data(
            TensorData::new(normalized_f32(gaussian_taps(len, sigma)), Shape::new(shape)),
            options,
        )
        .repeat_dim(0, channels)
    };
    let vertical = taps(height, [1, 1, height, 1]);
    let horizontal = taps(width, [1, 1, 1, width]);
    let depthwise = ConvOptions::new([1, 1], [0, 0], [1, 1], channels);

    let padded = pad_edges(image, height / 2, width / 2);
    let rows = conv2d(padded, vertical, None, depthwise.clone());
    conv2d(rows, horizontal, None, depthwise)
}
