//! Behavioral tests for `birefnet-metric` on hand-constructed masks.

use approx::assert_abs_diff_eq;
use birefnet_metric::{EvaluationConfig, MetricsAggregator, Sample, TARGET_THRESHOLD};
use burn::prelude::*;

/// 12x12 mask with a centered 6x6 square (`1.0`) on a `0.0` background.
fn square(inverted: bool) -> Vec<f32> {
    Vec::from_fn(144, |i| {
        let (row, col) = (i / 12, i % 12);
        let inside = (3..9).contains(&row) && (3..9).contains(&col);
        if inside == inverted { 0.0 } else { 1.0 }
    })
}

fn sample(pred: &[f32], gt: &[f32]) -> Sample {
    Sample::from_slices(pred, gt, [12, 12]).unwrap()
}

#[test]
fn perfect_predictions_score_ideally() {
    let gt = square(false);
    let perfect = sample(&gt, &gt);
    let scores = perfect.scores(&EvaluationConfig::new());

    assert_abs_diff_eq!(scores.mae, 0.0);
    assert_abs_diff_eq!(scores.mse, 0.0);
    assert_abs_diff_eq!(scores.f_measure.adaptive, 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(scores.f_measure.curve.max(), 1.0, epsilon = 1e-12);
    assert_abs_diff_eq!(scores.s_measure, 1.0, epsilon = 1e-9);
    assert_abs_diff_eq!(scores.weighted_f_measure, 1.0, epsilon = 1e-9);
    assert_abs_diff_eq!(scores.biou.max(), 1.0, epsilon = 1e-12);
    // The reference normalizes the E-measure by `N - 1`, so a perfect match scores N / (N - 1).
    assert_abs_diff_eq!(scores.e_measure.adaptive, 144.0 / 143.0, epsilon = 1e-12);
}

#[test]
fn inverted_predictions_score_worst() {
    let inverted = sample(&square(true), &square(false));
    let scores = inverted.scores(&EvaluationConfig::new());

    assert_abs_diff_eq!(scores.mae, 1.0);
    assert_abs_diff_eq!(scores.mse, 1.0);
    assert_abs_diff_eq!(scores.f_measure.adaptive, 0.0);
    assert_abs_diff_eq!(scores.weighted_f_measure, 0.0, epsilon = 1e-9);
    assert!(scores.s_measure < 0.1, "S-measure {}", scores.s_measure);
    assert!(
        scores.e_measure.adaptive < 0.1,
        "E-measure {}",
        scores.e_measure.adaptive
    );
}

#[test]
fn curves_are_monotone_in_the_threshold() {
    let gt = square(false);
    let pred: Vec<f32> = Vec::from_fn(144, |i| {
        let (row, col) = (i / 12, i % 12);
        f32::from(u8::try_from(row * 12 + col).unwrap()) / 143.0
    });
    let f = sample(&pred, &gt).f_measure(0.3);
    // Raising the threshold never adds positives: recall is non-increasing in the threshold.
    for [lo, hi] in (0..=255_u8).collect::<Vec<_>>().array_windows::<2>() {
        assert!(f.recall.at(*lo) >= f.recall.at(*hi));
    }
    assert_abs_diff_eq!(f.recall.at(0), 1.0);
}

#[test]
fn degenerate_targets_follow_the_reference_conventions() {
    let empty_gt = vec![0.0_f32; 144];
    let half = vec![0.5_f32; 144];
    let all_background = sample(&half, &empty_gt);
    // Constant predictions are not min-max normalized; S = 1 - mean(pred) without foreground.
    assert_abs_diff_eq!(all_background.s_measure(0.5), 0.5, epsilon = 1e-12);
    assert_abs_diff_eq!(all_background.weighted_f_measure(1.0), 0.0);
    assert_abs_diff_eq!(all_background.f_measure(0.3).adaptive, 0.0);

    let full_gt = vec![1.0_f32; 144];
    let all_foreground = sample(&half, &full_gt);
    assert_abs_diff_eq!(all_foreground.s_measure(0.5), 0.5, epsilon = 1e-12);

    // A single pixel does not divide by `N - 1 = 0`.
    let one = Sample::from_slices(&[1.0], &[1.0], [1, 1]).unwrap();
    assert!(one.e_measure().adaptive.is_finite());
}

#[test]
fn targets_are_binarized_strictly_above_128_over_255() {
    let just_below = TARGET_THRESHOLD;
    let just_above = f32::from_bits(TARGET_THRESHOLD.to_bits() + 1);
    let sample = Sample::from_slices(&[0.0, 1.0], &[just_below, just_above], [1, 2]).unwrap();
    assert_eq!(sample.targets(), &[false, true]);

    let from_bytes = Sample::from_u8(&[0, 255], &[128, 129], [1, 2]).unwrap();
    assert_eq!(from_bytes.targets(), &[false, true]);
}

#[test]
fn invalid_inputs_are_rejected() {
    assert!(
        Sample::from_slices(&[0.0; 3], &[0.0; 4], [2, 2])
            .unwrap_err()
            .is_invalid_input()
    );
    assert!(
        Sample::from_slices(&[f32::NAN; 4], &[0.0; 4], [2, 2])
            .unwrap_err()
            .is_invalid_input()
    );
    assert!(
        Sample::from_u8(&[], &[], [0, 0])
            .unwrap_err()
            .is_invalid_input()
    );

    let device = Device::flex();
    let pred = Tensor::<4>::zeros([1, 1, 4, 4], &device);
    let mismatched = Tensor::<4>::zeros([1, 1, 4, 5], &device);
    let multi_channel = Tensor::<4>::zeros([1, 2, 4, 4], &device);
    for targets in [mismatched, multi_channel] {
        assert!(
            Sample::batch_from_tensors(&pred, &targets)
                .unwrap_err()
                .is_invalid_input()
        );
    }
    let empty = Tensor::<4>::zeros([0, 1, 4, 4], &device);
    assert_eq!(
        Sample::batch_from_tensors(&empty, &empty).unwrap(),
        Vec::new()
    );
}

#[test]
fn evaluation_config_validates_parameters() {
    let config = EvaluationConfig::new();
    assert!(config.with_f_beta_sq(-1.0).unwrap_err().is_invalid_config());
    assert!(
        config
            .with_weighted_f_beta_sq(f64::NAN)
            .unwrap_err()
            .is_invalid_config()
    );
    assert!(config.with_s_alpha(1.5).unwrap_err().is_invalid_config());
    assert!(
        config
            .with_biou_dilation_ratio(0.0)
            .unwrap_err()
            .is_invalid_config()
    );
    let tuned = config
        .with_f_beta_sq(1.0)
        .unwrap()
        .with_s_alpha(0.7)
        .unwrap();
    assert_abs_diff_eq!(tuned.f_beta_sq(), 1.0);
    assert_abs_diff_eq!(tuned.s_alpha(), 0.7);
}

#[test]
fn aggregator_reads_a_batch_and_averages_samples_equally() {
    let device = Device::flex();
    let gt = square(false);
    let inverted = square(true);
    let to_tensor =
        |values: &[f32]| Tensor::<1>::from_floats(values, &device).reshape([1, 1, 12, 12]);
    let predictions = Tensor::cat(vec![to_tensor(&gt), to_tensor(&inverted)], 0);
    let targets = Tensor::cat(vec![to_tensor(&gt), to_tensor(&gt)], 0);

    let mut aggregator = MetricsAggregator::default();
    assert!(aggregator.summary().is_none());
    assert_eq!(aggregator.update(&predictions, &targets).unwrap(), 2);
    assert_eq!(aggregator.len(), 2);

    let mut manual = MetricsAggregator::default();
    for pred in [&gt, &inverted] {
        manual.push(&sample(pred, &gt).scores(manual.config()));
    }
    let (batched, pushed) = (aggregator.summary().unwrap(), manual.summary().unwrap());
    assert_eq!(batched, pushed);
    assert_abs_diff_eq!(batched.mae, 0.5);

    // A failed update leaves the aggregator untouched.
    let bad = Tensor::<4>::zeros([1, 1, 3, 3], &device);
    assert!(
        aggregator
            .update(&predictions, &bad)
            .unwrap_err()
            .is_invalid_input()
    );
    assert_eq!(aggregator.len(), 2);

    let rendered = batched.to_string();
    assert!(
        rendered.starts_with("maxFm=0.500") && rendered.contains("MAE=0.500"),
        "{rendered}"
    );

    aggregator.reset();
    assert!(aggregator.is_empty());
}
