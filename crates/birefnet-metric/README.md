# birefnet-metric

Dichotomous / salient object segmentation metrics for BiRefNet, numerically matching the
`py_sod_metrics` implementation used by the upstream evaluation scripts.

| Measure | Method on [`Sample`] | Reported as |
| --- | --- | --- |
| Mean absolute / squared error | [`mae`](Sample::mae), [`mse`](Sample::mse) | `MAE`, `MSE` |
| F-measure (`β² = 0.3`) | [`f_measure`](Sample::f_measure) | `adpFm`, `meanFm`, `maxFm` |
| E-measure | [`e_measure`](Sample::e_measure) | `adpEm`, `meanEm`, `maxEm` |
| S-measure (`α = 0.5`) | [`s_measure`](Sample::s_measure) | `Smeasure` |
| Weighted F-measure (`β² = 1`) | [`weighted_f_measure`](Sample::weighted_f_measure) | `wFmeasure` |
| Boundary IoU | [`biou`](Sample::biou) | `meanBIoU`, `maxBIoU` |

Predictions and targets are `[0, 1]` maps. A [`Sample`] is read from the device once (a batch
costs one transaction) and every measure runs on the host; threshold curves come from a single
256-bin histogram. [`MetricsAggregator`] averages [`Scores`] over a dataset in constant memory.
With the `train` feature, `SegmentationMetric` exposes any linearly averageable measure as a Burn
`Metric`.

## Examples

```rust
use birefnet_metric::{EvaluationConfig, MetricsAggregator};
use burn::prelude::*;

let device = Device::flex();
let mut mask = [[0.0_f32; 16]; 16];
for row in &mut mask[4..12] {
    row[4..12].fill(1.0);
}
let targets = Tensor::<4>::from_floats([[mask]], &device);
let predictions = targets.clone();

let mut aggregator = MetricsAggregator::new(EvaluationConfig::new());
aggregator.update(&predictions, &targets)?;
let scores = aggregator.summary().expect("one sample was added");

assert_eq!(scores.mae, 0.0);
assert!((scores.f_measure.curve.max() - 1.0).abs() < 1e-12);
println!("{scores}");
# Ok::<(), birefnet_metric::MetricError>(())
```

Saved 8-bit masks can be evaluated without a device:

```rust
use birefnet_metric::Sample;

let prediction = [0_u8, 64, 255, 255];
let target = [0_u8, 0, 255, 255];
let sample = Sample::from_u8(&prediction, &target, [2, 2])?;
assert_eq!(sample.f_measure(0.3).adaptive, 1.0);
# Ok::<(), birefnet_metric::MetricError>(())
```

## License

MIT OR Apache-2.0
