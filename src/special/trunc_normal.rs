use burn::{prelude::*, tensor::Distribution};

use crate::special::erfinv::Erfinv;

pub fn trunc_normal<B: Backend, const D: usize>(
    x: Tensor<B, D, Float>,
    mean: f64,
    std: f64,
    a: f64,
    b: f64,
) -> Tensor<B, D, Float> {
    fn norm_cdf(x: f64) -> f64 {
        (1. + libm::erf(x / 2.0_f64.sqrt())) / 2.
    }

    let l = norm_cdf((a - mean) / std);
    let u = norm_cdf((b - mean) / std);

    let x = x.random_like(Distribution::Uniform(2.0 * l - 1.0, 2.0 * u - 1.0));

    let x = x.erfinv();

    let x = x.mul_scalar(std * (2.0_f64.sqrt()));
    let x = x.add_scalar(mean);

    let x = x.clamp(a, b);
    x
}
