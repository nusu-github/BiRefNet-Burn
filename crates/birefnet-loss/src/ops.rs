//! Internal shared tensor math and validation helpers for `birefnet-loss`.

use burn::{nn::loss::Reduction, prelude::*};

pub(crate) fn usize_to_f64(value: usize) -> f64 {
    f64::from(u32::try_from(value).expect("dimension exceeds u32::MAX"))
}

pub(crate) fn usize_to_i32(value: usize) -> i32 {
    i32::try_from(value).expect("dimension exceeds i32::MAX")
}

/// Element-wise numerically stable binary cross-entropy from raw logits and float targets in `[0, 1]`.
pub(crate) fn bce_with_logits<const D: usize>(logits: Tensor<D>, targets: Tensor<D>) -> Tensor<D> {
    let stable_relu = logits.clone().clamp_min(0.0) - logits.clone() * targets;
    let log_part = logits.abs().neg().exp().add_scalar(1.0).log();
    stable_relu + log_part
}

/// Reduces an arbitrary-rank loss tensor to `[1]` according to `reduction`.
#[expect(
    clippy::needless_pass_by_value,
    reason = "matches Burn nn::loss forward(..., Reduction) by-value convention"
)]
pub(crate) fn reduce_loss<const D: usize>(loss: Tensor<D>, reduction: Reduction) -> Tensor<1> {
    match reduction {
        Reduction::Mean | Reduction::Auto => loss.mean(),
        Reduction::Sum => loss.sum(),
        Reduction::BatchMean => {
            let batch_size = usize_to_f64(loss.dims()[0]);
            loss.sum().div_scalar(batch_size)
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

pub(crate) fn accumulate_tensor(acc: &mut Option<Tensor<1>>, term: Tensor<1>) {
    *acc = Some(match acc.take() {
        Some(prev) => prev + term,
        None => term,
    });
}
