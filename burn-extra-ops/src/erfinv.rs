//! # Inverse Error Function (erfinv)
//!
//! This module provides an implementation of the inverse error function, `erfinv`.
//! It is used for initializing weights with a truncated normal distribution.
//!
//! The implementation is a direct translation of the `erfinv` M-file from MATLAB v2.0,
//! authored by Gary L. Pavlis, Indiana University.
//! It uses a rational approximation for an initial guess, followed by two steps of
//! Newton's method to refine the result to full accuracy.

use burn::prelude::*;
use core::f64::consts::PI;

const CENTRAL_RANGE: f64 = 0.7;
const A: [f64; 4] = [0.886226899, -1.645349621, 0.914624893, -0.140543331];
const B: [f64; 4] = [-2.118377725, 1.442710462, -0.329097515, 0.012229801];
const C: [f64; 4] = [-1.970840454, -1.624906493, 3.429567803, 1.641345311];
const D_COEFF: [f64; 2] = [3.543889200, 1.637067800];

/// A trait for calculating the inverse error function on a tensor.
pub trait Erfinv {
    /// Calculates the inverse error function element-wise.
    fn erfinv(self) -> Self;
}

impl<B: Backend, const D: usize> Erfinv for Tensor<B, D> {
    fn erfinv(self) -> Self {
        erfinv_(self)
    }
}

/// The core implementation of the inverse error function.
fn erfinv_<B: Backend, const D: usize>(y: Tensor<B, D>) -> Tensor<B, D> {
    let y_abs = y.clone().abs();
    let mut result = y.zeros_like();

    // Handle edge cases: |y| >= 1
    let ge_one_mask = y_abs.clone().greater_equal_elem(1.0);
    result = result.mask_fill(ge_one_mask, f64::INFINITY);

    // Main computation for |y| < 1
    let lt_one_mask = y_abs.clone().lower_elem(1.0);

    // Compute for central range: |y| <= 0.7
    let central_mask = y_abs
        .clone()
        .lower_equal_elem(CENTRAL_RANGE)
        .equal(lt_one_mask.clone());
    let central_result = compute_central_range(y.clone());
    result = result.mask_where(central_mask, central_result);

    // Compute for outer range: 0.7 < |y| < 1
    let outer_mask = y_abs.clone().greater_elem(CENTRAL_RANGE).equal(lt_one_mask);
    let outer_result = compute_outer_range(y.clone());
    result = result.mask_where(outer_mask, outer_result);

    // Apply Newton-Raphson correction only for |y| < 1
    let lt_one_mask_for_newton = y_abs.lower_elem(1.0);
    let y_lt_one = y
        .clone()
        .mask_where(lt_one_mask_for_newton.clone(), y.clone());
    let result_lt_one = result
        .clone()
        .mask_where(lt_one_mask_for_newton.clone(), result.clone());
    let corrected_result = apply_newton_raphson(result_lt_one, y_lt_one);
    result = result.mask_where(lt_one_mask_for_newton, corrected_result);

    result * y.sign()
}

/// Computes the inverse error function for the central range `|y| <= 0.7`.
fn compute_central_range<B: Backend, const D: usize>(y: Tensor<B, D>) -> Tensor<B, D> {
    let z = y.clone().powf_scalar(2.0);

    // Horner's method with minimized clones: calculate both numerator and denominator
    // First, we need z for both calculations, so we'll reuse it strategically
    let z_a3 = z.clone() * A[3];
    let z_b3 = z.clone() * B[3];

    // Calculate numerator: A[3]*z^3 + A[2]*z^2 + A[1]*z + A[0]
    let num = ((z_a3 + A[2]) * z.clone() + A[1]) * z.clone() + A[0];

    // Calculate denominator: B[3]*z^4 + B[2]*z^3 + B[1]*z^2 + B[0]*z + 1
    let dem = (((z_b3 + B[2]) * z.clone() + B[1]) * z.clone() + B[0]) * z + 1.0;

    y * num / dem
}

/// Computes the inverse error function for the outer range `0.7 < |y| < 1`.
fn compute_outer_range<B: Backend, const D: usize>(y: Tensor<B, D>) -> Tensor<B, D> {
    let y_abs = y.clone().abs();
    // Correct formula: sqrt(-log((1 - |y|) / 2))
    let z = ((1.0_f64 - y_abs) / 2.0_f64).log().neg().sqrt();
    let num = z.clone() * C[3] + C[2];
    let num = (num * z.clone()) + C[1];
    let num = (num * z.clone()) + C[0];
    let dem = z.clone() * D_COEFF[1] + D_COEFF[0];
    let dem = (dem * z) + 1.0;
    y.sign() * num / dem
}

/// Refines the result using two steps of Newton-Raphson iteration.
fn apply_newton_raphson<B: Backend, const D: usize>(
    mut result: Tensor<B, D>,
    y: Tensor<B, D>,
) -> Tensor<B, D> {
    let two_over_sqrt_pi = 2.0 / PI.sqrt();
    for _ in 0..2 {
        let correction = (result.clone().erf() - y.clone())
            / ((-result.clone().powf_scalar(2.0)).exp() * two_over_sqrt_pi);
        result = result - correction;
    }
    result
}

/// Convenience function for inverse error function
pub fn erfinv<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    x.erfinv()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::{ndarray::NdArray, Autodiff},
        tensor::Tensor,
    };

    type TestBackend = Autodiff<NdArray<f32>>;

    #[test]
    fn erfinv_valid_input_returns_correct_dimensions() {
        let device = Default::default();
        let x = Tensor::<TestBackend, 1>::from_floats([0.0, 0.5, 0.9], &device);
        let result = erfinv(x);
        assert_eq!(result.dims(), [3]);
    }

    #[test]
    fn erfinv_pytorch_values_matches_expected() {
        let device = Default::default();

        // Test values that match PyTorch's expected outputs
        // torch.special.erfinv(torch.tensor([0.0, 0.5, -1.0, 0.9]))
        // Expected: tensor([ 0.0000,  0.4769,    -inf,  1.1631])

        let x = Tensor::<TestBackend, 1>::from_floats([0.0, 0.5, 0.9], &device);
        let result = erfinv(x);
        let data = result.to_data();

        // Test zero value
        assert!((data.iter::<f32>().next().unwrap() - 0.0).abs() < 1e-6);

        // Test central range value (|y| <= 0.7)
        let expected_half = 0.4769_f32; // Expected for erfinv(0.5)
        assert!((data.iter::<f32>().nth(1).unwrap() - expected_half).abs() < 1e-3);

        // Test outer range value (|y| > 0.7)
        let expected_nine = 1.1631_f32; // Expected for erfinv(0.9)
        assert!((data.iter::<f32>().nth(2).unwrap() - expected_nine).abs() < 1e-2);
    }
}
