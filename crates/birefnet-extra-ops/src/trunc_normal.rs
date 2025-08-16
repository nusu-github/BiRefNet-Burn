//! # Truncated Normal Distribution
//!
//! Provides a function to initialize a tensor with values drawn from a truncated
//! normal distribution. This is commonly used for initializing weight matrices in
//! neural networks, such as Vision Transformers.

use burn::{prelude::*, tensor::Distribution};

use super::erfinv::Erfinv;

/// Fills a tensor with values from a truncated normal distribution.
///
/// The values are drawn from a normal distribution with the specified `mean` and `std`,
/// and are clipped to be within the range `[a, b]`.
///
/// # Arguments
///
/// * `x` - The tensor to be filled.
/// * `mean` - The mean of the normal distribution.
/// * `std` - The standard deviation of the normal distribution.
/// * `a` - The lower bound of the truncation.
/// * `b` - The upper bound of the truncation.
pub fn trunc_normal<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    mean: f64,
    std: f64,
    a: f64,
    b: f64,
) -> Tensor<B, D> {
    fn norm_cdf(x: f64) -> f64 {
        f64::midpoint(1., libm::erf(x / 2.0_f64.sqrt()))
    }

    let l = norm_cdf((a - mean) / std);
    let u = norm_cdf((b - mean) / std);

    // Fill with uniform values in the range [2l-1, 2u-1]
    let x = x.random_like(Distribution::Uniform(
        2.0f64.mul_add(l, -1.0),
        2.0f64.mul_add(u, -1.0),
    ));

    // Apply inverse error function to transform to a normal distribution
    let x = x.erfinv();

    // Scale and shift to the correct mean and std
    let x = x.mul_scalar(std * (2.0f64.sqrt()));
    let x = x.add_scalar(mean);

    // Clamp to the desired range
    x.clamp(a, b)
}

/// Convenience function for 2D tensor initialization (common for weight matrices)
pub fn trunc_normal_<B: Backend>(
    tensor: Tensor<B, 2>,
    mean: f64,
    std: f64,
    a: f64,
    b: f64,
) -> Tensor<B, 2> {
    trunc_normal(tensor, mean, std, a, b)
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tensor;

    use super::*;
    use crate::tests::TestBackend;

    #[test]
    fn trunc_normal_preserves_input_tensor_dimensions() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 2>::zeros([3, 4], &device);
        let result = trunc_normal_(tensor, 0.0, 1.0, -2.0, 2.0);
        assert_eq!(result.dims(), [3, 4]);
    }

    #[test]
    fn trunc_normal_handles_multidimensional_tensors() {
        let device = Default::default();
        let tensor = Tensor::<TestBackend, 3>::zeros([2, 3, 4], &device);
        let result = trunc_normal(tensor, 0.0, 1.0, -2.0, 2.0);
        assert_eq!(result.dims(), [2, 3, 4]);
    }
}
