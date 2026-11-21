//! Integration tests for `birefnet-metric` accuracy, edge cases, and multi-batch aggregation.

use approx::assert_abs_diff_eq;
use birefnet_metric::{
    BIoUInput, BIoUMetric, EMeasureInput, EMeasureMetric, FMeasureInput, FMeasureMetric, MAEInput,
    MAEMetric, MSEInput, MSEMetric, MetricsAggregator, SMeasureInput, SMeasureMetric,
    WeightedFMeasureInput, WeightedFMeasureMetric, calculate_all_metrics, calculate_biou,
    calculate_e_measure, calculate_f_measure, calculate_mae, calculate_mse, calculate_s_measure,
    calculate_weighted_f_measure,
};
use burn::{
    data::dataloader::Progress,
    prelude::*,
    train::metric::{Metric, MetricMetadata, Numeric},
};

fn metadata() -> MetricMetadata {
    MetricMetadata {
        progress: Progress {
            items_processed: 1,
            items_total: 1,
            unit: None,
        },
        iteration: None,
        lr: None,
    }
}

/// Constructs a 12x12 mask with a centered 6x6 foreground square (`255.0`) on `0.0` background.
fn centered_square_2d(device: &Device, inverted: bool) -> Tensor<2> {
    let mut data = [if inverted { 255.0_f32 } else { 0.0_f32 }; 12 * 12];
    for row in 3..9 {
        for col in 3..9 {
            data[row * 12 + col] = if inverted { 0.0 } else { 255.0 };
        }
    }
    Tensor::<2>::from_floats(
        core::array::from_fn::<[f32; 12], 12, _>(|r| core::array::from_fn(|c| data[r * 12 + c])),
        device,
    )
}

#[test]
fn scalar_metrics_match_perfect_and_inverted_predictions() {
    let device = Device::flex();
    let gt_2d = centered_square_2d(&device, false);
    let pred_perfect_2d = gt_2d.clone();
    let pred_inverted_2d = centered_square_2d(&device, true);

    // Perfect match
    assert_abs_diff_eq!(
        calculate_mae(pred_perfect_2d.clone(), gt_2d.clone()),
        0.0,
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        calculate_mse(pred_perfect_2d.clone(), gt_2d.clone()),
        0.0,
        epsilon = 1e-6
    );
    assert_abs_diff_eq!(
        calculate_f_measure(pred_perfect_2d.clone(), gt_2d.clone(), 0.3),
        1.0,
        epsilon = 1e-5
    );
    assert_abs_diff_eq!(
        calculate_s_measure(pred_perfect_2d.clone(), gt_2d.clone(), 0.5),
        1.0,
        epsilon = 1e-5
    );
    let (em_adaptive, em_curve) = calculate_e_measure(pred_perfect_2d.clone(), gt_2d.clone());
    assert_abs_diff_eq!(em_adaptive, 144.0 / 143.0, epsilon = 1e-4);
    assert_eq!(em_curve.len(), 256);
    assert_abs_diff_eq!(
        calculate_biou(pred_perfect_2d.clone(), gt_2d.clone(), 0.08),
        1.0,
        epsilon = 1e-5
    );
    assert_abs_diff_eq!(
        calculate_weighted_f_measure(
            pred_perfect_2d.unsqueeze_dim(0),
            gt_2d.clone().unsqueeze_dim(0),
            1.0
        ),
        1.0,
        epsilon = 1e-4
    );

    // Inverted predictions should have high error and low similarity
    let inv_abs_err = calculate_mae(pred_inverted_2d.clone(), gt_2d.clone());
    let inv_sq_err = calculate_mse(pred_inverted_2d.clone(), gt_2d.clone());
    let inv_f_score = calculate_f_measure(pred_inverted_2d.clone(), gt_2d.clone(), 0.3);
    let inv_struct = calculate_s_measure(pred_inverted_2d.clone(), gt_2d.clone(), 0.5);
    let inv_boundary = calculate_biou(pred_inverted_2d.clone(), gt_2d.clone(), 0.08);
    let inv_weighted = calculate_weighted_f_measure(
        pred_inverted_2d.unsqueeze_dim(0),
        gt_2d.unsqueeze_dim(0),
        1.0,
    );

    assert_abs_diff_eq!(inv_abs_err, 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(inv_sq_err, 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(inv_f_score, 0.0, epsilon = 1e-5);
    assert!(
        inv_struct < 0.3,
        "expected low S-measure for inverted mask, got {inv_struct}"
    );
    assert_abs_diff_eq!(inv_boundary, 0.0, epsilon = 1e-5);
    assert_abs_diff_eq!(inv_weighted, 0.0, epsilon = 1e-5);
}

#[test]
fn metric_adapters_handle_4d_batches_without_double_averaging() {
    let device = Device::flex();
    let gt = centered_square_2d(&device, false).unsqueeze::<4>();
    let perfect = gt.clone();
    let inverted = centered_square_2d(&device, true).unsqueeze::<4>();

    let meta = metadata();

    let mut f_metric = FMeasureMetric::new();
    let mut e_metric = EMeasureMetric::new();
    let mut s_metric = SMeasureMetric::new();
    let mut b_metric = BIoUMetric::new();
    let mut w_metric = WeightedFMeasureMetric::new();
    let mut abs_err_metric = MAEMetric::new();
    let mut sq_err_metric = MSEMetric::new();

    // Update 1: perfect sample (F = 1.0, MAE = 0.0)
    f_metric
        .update(&FMeasureInput::new(perfect.clone(), gt.clone()), &meta)
        .unwrap();
    e_metric
        .update(&EMeasureInput::new(perfect.clone(), gt.clone()), &meta)
        .unwrap();
    s_metric
        .update(&SMeasureInput::new(perfect.clone(), gt.clone()), &meta)
        .unwrap();
    b_metric
        .update(&BIoUInput::new(perfect.clone(), gt.clone()), &meta)
        .unwrap();
    w_metric
        .update(
            &WeightedFMeasureInput::new(perfect.clone(), gt.clone()),
            &meta,
        )
        .unwrap();
    abs_err_metric
        .update(&MAEInput::new(perfect.clone(), gt.clone()), &meta)
        .unwrap();
    sq_err_metric
        .update(&MSEInput::new(perfect, gt.clone()), &meta)
        .unwrap();

    // Update 2: inverted prediction (F = 0.0, MAE = 1.0)
    f_metric
        .update(&FMeasureInput::new(inverted.clone(), gt.clone()), &meta)
        .unwrap();
    abs_err_metric
        .update(&MAEInput::new(inverted, gt), &meta)
        .unwrap();

    // Exact unweighted average of [1.0, 0.0] is 0.5 (double-averaging would yield 0.75).
    let running_f = match f_metric.running_value().unwrap() {
        burn::train::metric::NumericEntry::Value(v)
        | burn::train::metric::NumericEntry::Final(v) => v,
        burn::train::metric::NumericEntry::Aggregated {
            aggregated_value, ..
        } => aggregated_value,
    };
    assert_abs_diff_eq!(running_f, 0.5, epsilon = 1e-4);

    // Clear resets accumulated samples.
    f_metric.clear();
    assert_abs_diff_eq!(f_metric.adaptive_fm(), 0.0, epsilon = 1e-6);

    // Ensure Debug works on all metrics and EMeasureState.
    let _ = format!(
        "{f_metric:?} {e_metric:?} {s_metric:?} {b_metric:?} {w_metric:?} {abs_err_metric:?} {sq_err_metric:?}"
    );
    let (e_avg, _) = e_metric.state().results();
    assert!(e_avg > 0.9, "expected high E-measure, got {e_avg}");
}

#[test]
fn aggregator_and_calculate_all_metrics_compute_real_scores() {
    let device = Device::flex();
    let gt = centered_square_2d(&device, false).unsqueeze::<4>();
    let all_metrics = calculate_all_metrics(&gt, &gt, 0.08);
    assert_abs_diff_eq!(all_metrics.iou, 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(all_metrics.f_measure, 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(all_metrics.mae, 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(all_metrics.mse, 0.0, epsilon = 1e-6);
    assert_abs_diff_eq!(all_metrics.s_measure, 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(
        all_metrics.e_measure_adaptive,
        144.0 / 143.0,
        epsilon = 1e-4
    );
    assert_abs_diff_eq!(all_metrics.weighted_f_measure, 1.0, epsilon = 1e-4);

    let mut aggregator = MetricsAggregator::new();
    aggregator.update(&gt, &gt, 0.08);
    let (biou, fm, mae) = aggregator.averages();
    assert_abs_diff_eq!(biou, 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(fm, 1.0, epsilon = 1e-5);
    assert_abs_diff_eq!(mae, 0.0, epsilon = 1e-6);

    aggregator.reset();
    assert_abs_diff_eq!(aggregator.averages().0, 0.0, epsilon = 1e-6);
}
