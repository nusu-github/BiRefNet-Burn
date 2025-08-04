//! Utility functions for BiRefNet metrics.
//!
//! This module provides convenience functions for calculating multiple metrics
//! at once and other metric-related utilities.

use burn::{
    prelude::*,
    tensor::{backend::Backend, Tensor},
};

use crate::metrics::{
    e_measure::calculate_e_measure, f_measure::calculate_f_measure, iou::calculate_iou,
    mae::calculate_mae, mse::calculate_mse, s_measure::calculate_s_measure,
    weighted_f_measure::calculate_weighted_f_measure,
};

/// Results from calculating all metrics.
#[derive(Debug, Clone)]
pub struct AllMetricsResult {
    pub iou: f32,
    pub f_measure: f32,
    pub mae: f32,
    pub mse: f32,
    pub s_measure: f32,
    pub e_measure_adaptive: f32,
    pub e_measure_curve: Vec<f32>,
    pub weighted_f_measure: f32,
}

/// Calculate all metrics at once.
pub fn calculate_all_metrics<B: Backend>(
    predictions: Tensor<B, 4>,
    targets: Tensor<B, 4>,
    threshold: f32,
) -> AllMetricsResult {
    // Ensure predictions and targets have correct shape
    let [batch_size, _, _, _] = predictions.dims();

    // Calculate metrics that work with 4D tensors
    let iou = calculate_iou(predictions.clone(), targets.clone(), threshold);
    let f_measure = calculate_f_measure(predictions.clone(), targets.clone(), threshold);
    let mae = calculate_mae(predictions.clone(), targets.clone(), true);

    // For metrics that need 2D tensors, process batch
    let mut mse_sum = 0.0;
    let mut s_measure_sum = 0.0;
    let mut e_measure_adaptive_sum = 0.0;
    let mut weighted_f_measure_sum = 0.0;
    let mut e_measure_curves = Vec::new();

    for b in 0..batch_size {
        let pred_2d: Tensor<B, 2> = predictions
            .clone()
            .slice(s![b..b + 1, 0..1, .., ..])
            .squeeze::<3>(0)
            .squeeze::<2>(0);
        let target_2d: Tensor<B, 2> = targets
            .clone()
            .slice(s![b..b + 1, 0..1, .., ..])
            .squeeze::<3>(0)
            .squeeze::<2>(0);

        mse_sum += calculate_mse(pred_2d.clone(), target_2d.clone()) as f64;
        s_measure_sum += calculate_s_measure(pred_2d.clone(), target_2d.clone(), 0.5) as f64;

        let (e_adaptive, e_curve) = calculate_e_measure(pred_2d.clone(), target_2d.clone());
        e_measure_adaptive_sum += e_adaptive as f64;
        e_measure_curves.push(e_curve);

        weighted_f_measure_sum += calculate_weighted_f_measure(pred_2d, target_2d, 1.0) as f64;
    }

    // Average the metrics
    let batch_size_f64 = batch_size as f64;
    let mse = (mse_sum / batch_size_f64) as f32;
    let s_measure = (s_measure_sum / batch_size_f64) as f32;
    let e_measure_adaptive = (e_measure_adaptive_sum / batch_size_f64) as f32;
    let weighted_f_measure = (weighted_f_measure_sum / batch_size_f64) as f32;

    // Average E-measure curves
    let curve_len = if e_measure_curves.is_empty() {
        256
    } else {
        e_measure_curves[0].len()
    };
    let mut avg_curve = vec![0.0f32; curve_len];
    for curve in &e_measure_curves {
        for (i, &val) in curve.iter().enumerate() {
            avg_curve[i] += val / batch_size as f32;
        }
    }

    AllMetricsResult {
        iou,
        f_measure,
        mae,
        mse,
        s_measure,
        e_measure_adaptive,
        e_measure_curve: avg_curve,
        weighted_f_measure,
    }
}
