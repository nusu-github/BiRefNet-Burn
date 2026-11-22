//! Inverse error function (`erfinv`) over Burn tensors.
//!
//! Approximates `erfinv(y)` over `(-1, 1)` using Strecok / MATLAB rational polynomials
//! refined by two Newton-Raphson iterations, returning signed infinities at `±1` and `NaN`
//! outside `[-1, 1]`.

use core::f64::consts::FRAC_2_SQRT_PI;

use burn::prelude::*;

/// Switch point between the central rational fit and the tail asymptotic fit (Strecok 1968).
const CENTRAL_RANGE: f64 = 0.7;
/// Placeholder fed to the tail fit for masked-out lanes; any value in `(0.7, 1)` keeps the
/// intermediate `log`/`sqrt` finite so masked lanes cannot produce NaN gradients.
const TAIL_PLACEHOLDER: f64 = 0.85;
const A: [f64; 4] = [0.886_226_899, -1.645_349_621, 0.914_624_893, -0.140_543_331];
const B: [f64; 4] = [-2.118_377_725, 1.442_710_462, -0.329_097_515, 0.012_229_801];
const C: [f64; 4] = [-1.970_840_454, -1.624_906_493, 3.429_567_803, 1.641_345_311];
const D_COEFF: [f64; 2] = [3.543_889_200, 1.637_067_800];

mod sealed {
    pub trait Sealed {}
}

/// Extension trait adding [`erfinv`](Self::erfinv) to Burn tensors.
pub trait Erfinv: sealed::Sealed {
    /// Computes the element-wise inverse error function.
    ///
    /// Inputs in `(-1, 1)` map to finite values of matching sign, `±1` maps to `±inf`,
    /// and inputs with `|y| > 1` or `NaN` map to `NaN`.
    #[must_use]
    fn erfinv(self) -> Self;
}

impl<const D: usize> sealed::Sealed for Tensor<D> {}

impl<const D: usize> Erfinv for Tensor<D> {
    fn erfinv(self) -> Self {
        erfinv_impl(self)
    }
}

/// Computes the element-wise inverse error function of `x`.
#[must_use]
pub fn erfinv<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    x.erfinv()
}

fn erfinv_impl<const D: usize>(y: Tensor<D>) -> Tensor<D> {
    let y_abs = y.clone().abs();
    // Start from NaN so inputs outside [-1, 1] and existing NaNs stay NaN.
    let mut result = y.full_like(f64::NAN);

    let pos_one_mask = y.clone().equal_elem(1.0);
    result = result.mask_fill(pos_one_mask, f64::INFINITY);

    let neg_one_mask = y.clone().equal_elem(-1.0);
    result = result.mask_fill(neg_one_mask, f64::NEG_INFINITY);

    let lt_one_mask = y_abs.clone().lower_elem(1.0);
    let central_mask = y_abs.clone().lower_equal_elem(CENTRAL_RANGE);
    let outer_mask = y_abs
        .greater_elem(CENTRAL_RANGE)
        .bool_and(lt_one_mask.clone());

    let safe_central_y = y.zeros_like().mask_where(central_mask.clone(), y.clone());
    let central_result = compute_central_range(safe_central_y);
    result = result.mask_where(central_mask, central_result);

    let safe_outer_y = y
        .full_like(TAIL_PLACEHOLDER)
        .mask_where(outer_mask.clone(), y.clone());
    let outer_result = compute_outer_range(safe_outer_y);
    result = result.mask_where(outer_mask, outer_result);

    // Mask out non-interior entries before Newton-Raphson so ±Inf/NaN cannot contaminate gradients or values.
    let safe_y = y.zeros_like().mask_where(lt_one_mask.clone(), y);
    let safe_result = result
        .zeros_like()
        .mask_where(lt_one_mask.clone(), result.clone());
    let corrected = apply_newton_raphson(safe_result, &safe_y);
    result.mask_where(lt_one_mask, corrected)
}

/// Rational approximation for `|y| <= 0.7`, preserving the sign of `y`.
fn compute_central_range<const D: usize>(y: Tensor<D>) -> Tensor<D> {
    let z = y.clone() * y.clone();
    let num = (((z.clone() * A[3] + A[2]) * z.clone() + A[1]) * z.clone()) + A[0];
    let den = ((((z.clone() * B[3] + B[2]) * z.clone() + B[1]) * z.clone() + B[0]) * z) + 1.0;
    y * num / den
}

/// Asymptotic approximation for `0.7 < |y| < 1.0`, preserving the sign of `y`.
fn compute_outer_range<const D: usize>(y: Tensor<D>) -> Tensor<D> {
    let y_abs = y.clone().abs();
    let z = ((1.0_f64 - y_abs) / 2.0_f64).log().neg().sqrt();
    let num = (((z.clone() * C[3] + C[2]) * z.clone() + C[1]) * z.clone()) + C[0];
    let den = ((z.clone() * D_COEFF[1] + D_COEFF[0]) * z) + 1.0;
    y.sign() * num / den
}

/// Two Newton-Raphson refinement steps solving `erf(x) - y = 0`.
fn apply_newton_raphson<const D: usize>(mut result: Tensor<D>, y: &Tensor<D>) -> Tensor<D> {
    for _ in 0..2 {
        let derivative = result.clone().square().neg().exp() * FRAC_2_SQRT_PI;
        let correction = (result.clone().erf() - y.clone()) / derivative;
        result = result - correction;
    }
    result
}
