///
/// Calculates the inverse error function.
///
/// Function to calculate inverse error function.  Rational approximation
/// is used to generate an initial approximation, which is then improved to
/// full accuracy by two steps of Newton's method.  Code is a direct
/// translation of the erfinv m file in matlab version 2.0.
/// Author:  Gary L. Pavlis, Indiana University
/// Date:  February 1996
///
use burn::prelude::*;

const CENTRAL_RANGE: f32 = 0.7;
const A: [f32; 4] = [0.886226899, -1.645349621, 0.914624893, -0.140543331];
const B: [f32; 4] = [-2.118377725, 1.442710462, -0.329097515, 0.012229801];
const C: [f32; 4] = [-1.970840454, -1.624906493, 3.429567803, 1.641345311];
const D: [f32; 2] = [3.543889200, 1.637067800];

pub trait Erfinv {
    fn erfinv(self) -> Self;
}

impl<B: Backend, const D: usize> Erfinv for Tensor<B, D, Float> {
    fn erfinv(self) -> Self {
        erfinv_(self)
    }
}

pub fn erfinv_<B: Backend, const D: usize>(y: Tensor<B, D, Float>) -> Tensor<B, D, Float> {
    let y_abs = y.clone().abs();
    let result = y.zeros_like().add_scalar(f32::NAN);

    // Handle edge cases
    let mask = y_abs.clone().lower_equal_elem(1.0);
    let inf_mask = y_abs.clone().equal_elem(1.0);
    let result = result.mask_fill(inf_mask.clone(), f32::INFINITY);

    // Compute for central range
    let central_mask = y_abs
        .clone()
        .lower_equal_elem(CENTRAL_RANGE)
        .equal(mask.clone());
    let central_result = compute_central_range(y.clone());
    let result = result.mask_where(central_mask.clone(), central_result);

    // Compute for outer range
    let outer_mask = y_abs
        .clone()
        .greater_elem(CENTRAL_RANGE)
        .equal(mask.clone());
    let outer_result = compute_outer_range(y.clone());
    let result = result.mask_where(outer_mask.clone(), outer_result);

    // Apply Newton-Raphson correction
    apply_newton_raphson(result, y)
}

fn compute_central_range<B: Backend, const D: usize>(
    y: Tensor<B, D, Float>,
) -> Tensor<B, D, Float> {
    let z = y.clone().powf_scalar(2.0);
    let num = z.clone().mul_scalar(A[3]).add_scalar(A[2]);
    let num = (num * z.clone()).add_scalar(A[1]);
    let num = (num * z.clone()).add_scalar(A[0]);
    let dem = z.clone().mul_scalar(B[3]).add_scalar(B[2]);
    let dem = (dem * z.clone()).add_scalar(B[1]);
    let dem = (dem * z.clone()).add_scalar(B[0]);
    let dem = (dem * z).add_scalar(1.0);
    y * num / dem
}

fn compute_outer_range<B: Backend, const _D: usize>(
    y: Tensor<B, _D, Float>,
) -> Tensor<B, _D, Float> {
    let y_abs = y.clone().abs();
    let z = (-(-y_abs.clone().sub_scalar(1.0).div_scalar(2.0)).log()).sqrt();
    let num = z.clone().mul_scalar(C[3]).add_scalar(C[2]);
    let num = (num * z.clone()).add_scalar(C[1]);
    let num = (num * z.clone()).add_scalar(C[0]);
    let dem = z.clone().mul_scalar(D[1]).add_scalar(D[0]);
    let dem = (dem * z).add_scalar(1.0);
    y.sign() * num / dem
}

fn apply_newton_raphson<B: Backend, const D: usize>(
    mut result: Tensor<B, D, Float>,
    y: Tensor<B, D, Float>,
) -> Tensor<B, D, Float> {
    let two_over_sqrt_pi = 2.0 / std::f32::consts::PI.sqrt();
    for _ in 0..2 {
        let correction = (result.clone().erf() - y.clone())
            / ((-result.clone().powf_scalar(2.0))
                .exp()
                .mul_scalar(two_over_sqrt_pi));
        result = result - correction;
    }
    result
}
