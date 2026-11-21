//! Truncated normal distribution sampling over Burn tensors.

use core::f64::consts::SQRT_2;

use burn::{prelude::*, tensor::Distribution};

use crate::erfinv::Erfinv;

/// Returns a tensor of shape `shape` on `device` sampled from a normal
/// distribution `N(mean, std^2)` truncated to `[a, b]`.
///
/// # Panics
///
/// Panics if `std` is not finite and strictly positive, if `mean`, `a`, or `b` are not
/// finite, or if `a > b`.
#[must_use]
pub fn trunc_normal<const D: usize>(
    shape: impl Into<Shape>,
    mean: f64,
    std: f64,
    a: f64,
    b: f64,
    device: &Device,
) -> Tensor<D> {
    assert!(
        mean.is_finite(),
        "trunc_normal: mean must be finite, got {mean}"
    );
    assert!(
        std.is_finite() && std > 0.0,
        "trunc_normal: std must be finite and positive, got {std}"
    );
    assert!(
        a.is_finite() && b.is_finite(),
        "trunc_normal: bounds must be finite, got [{a}, {b}]"
    );
    assert!(
        a <= b,
        "trunc_normal: lower bound a ({a}) must not exceed upper bound b ({b})"
    );

    let low = 2.0_f64.mul_add(norm_cdf((a - mean) / std), -1.0);
    let high = 2.0_f64.mul_add(norm_cdf((b - mean) / std), -1.0);
    if low >= high {
        return Tensor::full(shape, a.clamp(a, b), device);
    }

    let uniform = Tensor::random(shape, Distribution::Uniform(low, high), device);
    let normal = uniform.erfinv().mul_scalar(std * SQRT_2).add_scalar(mean);
    normal.clamp(a, b)
}

fn norm_cdf(x: f64) -> f64 {
    f64::midpoint(1.0, libm::erf(x / SQRT_2))
}
