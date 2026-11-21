//! Shared tensor slicing and preprocessing helpers for `birefnet-metric`.

use burn::prelude::*;

pub(crate) fn usize_to_f64(value: usize) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "pixel and sample counts fit comfortably within 52-bit f64 mantissa"
    )]
    {
        value as f64
    }
}

pub(crate) fn extract_sample_3d(tensor: &Tensor<4>, batch_idx: usize) -> Tensor<3> {
    tensor
        .clone()
        .slice(s![batch_idx..=batch_idx, .., .., ..])
        .squeeze_dim::<3>(0)
}

pub(crate) fn extract_sample_2d(tensor: &Tensor<4>, batch_idx: usize) -> Tensor<2> {
    tensor
        .clone()
        .slice(s![batch_idx..=batch_idx, 0..1, .., ..])
        .squeeze_dims::<2>(&[0, 1])
}

/// Normalizes prediction and ground-truth tensors following PyTorch `_prepare_data`:
/// - `gt` is binarized (`> 128 -> 1.0`, else `0.0`)
/// - `pred` is scaled by `1 / 255` and min-max normalized to `[0, 1]` when non-constant.
pub(crate) fn prepare_data<const D: usize>(
    pred: Tensor<D>,
    gt: Tensor<D>,
) -> (Tensor<D>, Tensor<D>) {
    let gt_binary = gt.greater_elem(128).float();
    let pred_norm = pred.div_scalar(255.0);

    let pred_min = pred_norm.clone().min();
    let pred_max = pred_norm.clone().max();
    let range = pred_max - pred_min.clone();

    let pred_final = if range.clone().greater_elem(1e-8).into_scalar::<bool>() {
        let pred_min_scalar = pred_min.into_scalar::<f64>();
        let range_scalar = range.into_scalar::<f64>();
        (pred_norm - pred_min_scalar) / range_scalar
    } else {
        pred_norm
    };

    (pred_final, gt_binary)
}
