//! Internal shared tensor math and validation helpers for `birefnet-loss`.

use burn::{nn::loss::Reduction, prelude::*, tensor::activation::log_sigmoid};

pub(crate) fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or_else(|_| {
        panic!("dimension {value} exceeds u32::MAX and cannot be represented exactly in f64")
    }))
}

/// Element-wise numerically stable binary cross-entropy from raw logits and float targets in `[0, 1]`.
///
/// Uses `(1 - y) * x - log_sigmoid(x)`, which is smooth everywhere. The common
/// `max(x, 0) - x * y + log(1 + exp(-|x|))` form has a kink at `x == 0` where Burn's
/// `sign(0) == 0` subgradient yields `dL/dx = -y` instead of the true `0.5 - y`.
pub(crate) fn bce_with_logits<const D: usize>(logits: Tensor<D>, targets: Tensor<D>) -> Tensor<D> {
    targets.neg().add_scalar(1.0) * logits.clone() - log_sigmoid(logits)
}

/// Flattens `[B, ...]` to `[B, N]` so per-sample reductions can run along dim 1.
pub(crate) fn flatten_per_sample<const D: usize>(tensor: Tensor<D>) -> Tensor<2> {
    tensor.flatten::<2>(1, D - 1)
}

/// Mean over all non-batch axes, returning `[B]`.
pub(crate) fn mean_per_sample<const D: usize>(tensor: Tensor<D>) -> Tensor<1> {
    flatten_per_sample(tensor).mean_dim(1).squeeze_dim::<1>(1)
}

/// Reduces an arbitrary-rank loss tensor to `[1]` according to `reduction`.
///
/// `BatchMean` divides the total by the size of axis 0; an empty batch reduces to `0`.
#[expect(
    clippy::needless_pass_by_value,
    reason = "matches Burn nn::loss forward(..., Reduction) by-value convention"
)]
pub(crate) fn reduce_loss<const D: usize>(loss: Tensor<D>, reduction: Reduction) -> Tensor<1> {
    match reduction {
        Reduction::Mean | Reduction::Auto => loss.mean(),
        Reduction::Sum => loss.sum(),
        Reduction::BatchMean => {
            let batch_size = loss.dims()[0];
            let total = loss.sum();
            if batch_size == 0 {
                total
            } else {
                total.div_scalar(usize_to_f64(batch_size))
            }
        }
    }
}

/// Asserts that `predictions` and `targets` share identical shapes.
pub(crate) fn assert_same_shape<const D: usize>(predictions: &Tensor<D>, targets: &Tensor<D>) {
    let pred_dims = predictions.dims();
    let target_dims = targets.dims();
    assert_eq!(
        pred_dims, target_dims,
        "shape of predictions ({pred_dims:?}) must match targets ({target_dims:?})"
    );
}

pub(crate) fn accumulate_tensor<const D: usize>(acc: &mut Option<Tensor<D>>, term: Tensor<D>) {
    *acc = Some(match acc.take() {
        Some(prev) => prev + term,
        None => term,
    });
}
