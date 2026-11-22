//! Burn `Metric` adapter tests (`train` feature).

#![cfg(feature = "train")]

use std::collections::HashSet;

use approx::assert_abs_diff_eq;
use birefnet_metric::{Measure, Sample, SegmentationInput, SegmentationMetric};
use burn::{
    data::dataloader::Progress,
    prelude::*,
    train::metric::{Metric, MetricAttributes, MetricMetadata, Numeric},
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

fn all_measures() -> [Measure; 9] {
    [
        Measure::Mae,
        Measure::Mse,
        Measure::AdaptiveFMeasure { beta_sq: 0.3 },
        Measure::MeanFMeasure { beta_sq: 0.3 },
        Measure::AdaptiveEMeasure,
        Measure::MeanEMeasure,
        Measure::SMeasure { alpha: 0.5 },
        Measure::WeightedFMeasure { beta_sq: 1.0 },
        Measure::MeanBiou {
            dilation_ratio: 0.02,
        },
    ]
}

fn input(device: &Device) -> SegmentationInput {
    let mut gt = [[0.0_f32; 10]; 10];
    for row in &mut gt[2..8] {
        row[3..7].fill(1.0);
    }
    let targets = Tensor::<4>::from_floats([[gt]], device).repeat_dim(0, 2);
    let noise = Tensor::<4>::random(
        [2, 1, 10, 10],
        burn::tensor::Distribution::Uniform(0.0, 0.4),
        device,
    );
    SegmentationInput::new((targets.clone() * 0.6 + noise).clamp(0.0, 1.0), targets)
}

#[test]
fn metric_values_equal_the_per_sample_average() {
    let device = Device::flex();
    device.seed(4);
    let item = input(&device);
    let samples = Sample::batch_from_tensors(&item.predictions, &item.targets).unwrap();

    for measure in all_measures() {
        let mut metric = SegmentationMetric::new(measure).unwrap();
        metric.update(&item, &metadata()).unwrap();
        let expected = match measure {
            Measure::Mae => f64::midpoint(samples[0].mae(), samples[1].mae()),
            Measure::SMeasure { alpha } => {
                f64::midpoint(samples[0].s_measure(alpha), samples[1].s_measure(alpha))
            }
            _ => continue,
        };
        assert_abs_diff_eq!(metric.value().unwrap().current(), expected, epsilon = 1e-9);
    }
}

#[test]
fn names_are_unique_and_error_measures_prefer_lower_values() {
    let metrics: Vec<_> = all_measures()
        .into_iter()
        .chain([Measure::AdaptiveFMeasure { beta_sq: 1.0 }])
        .map(|measure| SegmentationMetric::new(measure).unwrap())
        .collect();
    let names: HashSet<_> = metrics.iter().map(|m| m.name().to_string()).collect();
    assert_eq!(names.len(), metrics.len());

    for metric in &metrics {
        let MetricAttributes::Numeric(attributes) = metric.attributes() else {
            panic!("{} should be numeric", metric.name());
        };
        let is_error = matches!(metric.measure(), Measure::Mae | Measure::Mse);
        assert_eq!(attributes.higher_is_better, !is_error, "{}", metric.name());
    }
}

#[test]
fn empty_batches_report_not_available_and_parameters_are_validated() {
    let device = Device::flex();
    let mut metric = SegmentationMetric::new(Measure::Mae).unwrap();
    let empty = Tensor::<4>::zeros([0, 1, 4, 4], &device);
    let entry = metric
        .update(&SegmentationInput::new(empty.clone(), empty), &metadata())
        .unwrap();
    assert!(entry.is_not_available());

    for measure in [
        Measure::AdaptiveFMeasure { beta_sq: -0.3 },
        Measure::SMeasure { alpha: 2.0 },
        Measure::MeanBiou {
            dilation_ratio: f64::INFINITY,
        },
    ] {
        assert!(
            SegmentationMetric::new(measure)
                .unwrap_err()
                .is_invalid_config()
        );
    }
}
