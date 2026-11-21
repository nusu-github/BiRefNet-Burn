//! Convenience evaluation helpers computing all segmentation metrics simultaneously.

use burn::prelude::*;

use crate::{
    biou::calculate_biou,
    common::{extract_sample_2d, extract_sample_3d, usize_to_f64},
    e_measure::calculate_e_measure,
    f_measure::calculate_f_measure,
    mae::calculate_mae,
    mse::calculate_mse,
    s_measure::calculate_s_measure,
    weighted_f_measure::calculate_weighted_f_measure,
};

/// Combined results returned by [`calculate_all_metrics`].
#[derive(Debug, Clone, PartialEq)]
pub struct AllMetricsResult {
    /// Boundary IoU (`BIoU`) score.
    pub iou: f64,
    /// Adaptive F-measure (`beta = 0.3`).
    pub f_measure: f64,
    /// Mean Absolute Error (`MAE`).
    pub mae: f64,
    /// Mean Squared Error (`MSE`).
    pub mse: f64,
    /// Structure Measure (`SMeasure`, `alpha = 0.5`).
    pub s_measure: f64,
    /// Adaptive Enhanced-alignment Measure (`EMeasure`).
    pub e_measure_adaptive: f64,
    /// Changeable E-measure curve across thresholds `0..=255`.
    pub e_measure_curve: Vec<f64>,
    /// Weighted F-measure (`beta = 1.0`).
    pub weighted_f_measure: f64,
}

/// Computes every `BiRefNet` evaluation metric across a batch of `[B, 1, H, W]` tensors.
///
/// `dilation_ratio` controls the boundary thickness in [`calculate_biou`] (using `0.02` when `<= 0.0`).
#[must_use]
pub fn calculate_all_metrics(
    predictions: &Tensor<4>,
    targets: &Tensor<4>,
    dilation_ratio: f64,
) -> AllMetricsResult {
    let [batch_size, ..] = predictions.dims();
    if batch_size == 0 {
        return AllMetricsResult {
            iou: 0.0,
            f_measure: 0.0,
            mae: 0.0,
            mse: 0.0,
            s_measure: 0.0,
            e_measure_adaptive: 0.0,
            e_measure_curve: vec![0.0; 256],
            weighted_f_measure: 0.0,
        };
    }

    let ratio = if dilation_ratio > 0.0 {
        dilation_ratio
    } else {
        0.02
    };

    let mut iou_sum = 0.0;
    let mut f_measure_sum = 0.0;
    let mut abs_error_sum = 0.0;
    let mut sq_error_sum = 0.0;
    let mut s_measure_sum = 0.0;
    let mut e_measure_adaptive_sum = 0.0;
    let mut weighted_f_measure_sum = 0.0;
    let mut curve_sum = vec![0.0_f64; 256];

    for b in 0..batch_size {
        let pred_2d = extract_sample_2d(predictions, b);
        let target_2d = extract_sample_2d(targets, b);
        let pred_3d = extract_sample_3d(predictions, b);
        let target_3d = extract_sample_3d(targets, b);

        iou_sum += calculate_biou(pred_2d.clone(), target_2d.clone(), ratio);
        f_measure_sum += calculate_f_measure(pred_2d.clone(), target_2d.clone(), 0.3);
        abs_error_sum += calculate_mae(pred_2d.clone(), target_2d.clone());
        sq_error_sum += calculate_mse(pred_2d.clone(), target_2d.clone());
        s_measure_sum += calculate_s_measure(pred_2d.clone(), target_2d.clone(), 0.5);

        let (e_adaptive, e_curve) = calculate_e_measure(pred_2d, target_2d);
        e_measure_adaptive_sum += e_adaptive;
        for (dst, src) in curve_sum.iter_mut().zip(e_curve.iter()) {
            *dst += *src;
        }

        weighted_f_measure_sum += calculate_weighted_f_measure(pred_3d, target_3d, 1.0);
    }

    let batch_f64 = usize_to_f64(batch_size);
    for val in &mut curve_sum {
        *val /= batch_f64;
    }

    AllMetricsResult {
        iou: iou_sum / batch_f64,
        f_measure: f_measure_sum / batch_f64,
        mae: abs_error_sum / batch_f64,
        mse: sq_error_sum / batch_f64,
        s_measure: s_measure_sum / batch_f64,
        e_measure_adaptive: e_measure_adaptive_sum / batch_f64,
        e_measure_curve: curve_sum,
        weighted_f_measure: weighted_f_measure_sum / batch_f64,
    }
}
