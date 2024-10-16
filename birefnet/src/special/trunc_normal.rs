use burn::{prelude::*, tensor::Distribution};

use super::erfinv::Erfinv;

pub fn trunc_normal<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    mean: f64,
    std: f64,
    a: f64,
    b: f64,
) -> Tensor<B, D> {
    fn norm_cdf(x: f64) -> f64 {
        (1. + libm::erf(x / 2.0_f64.sqrt())) / 2.
    }

    let l = norm_cdf((a - mean) / std);
    let u = norm_cdf((b - mean) / std);

    let x = x.random_like(Distribution::Uniform(
        2.0_f64.mul_add(l, -1.0),
        2.0_f64.mul_add(u, -1.0),
    ));

    let x = x.erfinv();

    let x = x.mul_scalar(std * (2.0_f64.sqrt()));
    let x = x.add_scalar(mean);

    x.clamp(a, b)
}
