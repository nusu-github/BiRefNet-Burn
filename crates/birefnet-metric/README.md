# birefnet-metric

Dichotomous and salient object segmentation metrics for BiRefNet, numerically matching the
`py_sod_metrics` implementation used by upstream `evaluation/metrics.py` (verified to `<= 1e-9`
in `tests/upstream_parity.rs`).

## Features

| Feature | Default | Effect                                                                    |
| ------- | ------- | ------------------------------------------------------------------------- |
| `train` | no      | Exposes `SegmentationMetric`, `Measure`, and `SegmentationInput` for Burn |

## Measures

| Measure                       | Method on `Sample`          | Reported as                |
| ----------------------------- | --------------------------- | -------------------------- |
| Mean absolute / squared error | `mae`, `mse`                | `MAE`, `MSE`               |
| F-measure (`β² = 0.3`)        | `f_measure`                 | `adpFm`, `meanFm`, `maxFm` |
| E-measure                     | `e_measure`                 | `adpEm`, `meanEm`, `maxEm` |
| S-measure (`α = 0.5`)         | `s_measure`                 | `Smeasure`                 |
| Weighted F-measure (`β² = 1`) | `weighted_f_measure`        | `wFmeasure`                |
| Boundary IoU                  | `biou`                      | `meanBIoU`, `maxBIoU`      |

Predictions and targets are `[0, 1]` maps. A `Sample` reads each batch from the device once and
computes every measure on the host; 256-threshold curves come from a single histogram pass.
`MetricsAggregator` averages `Scores` across a dataset in constant memory.

## Examples

Evaluating Burn tensors with `MetricsAggregator`:

```rust
use birefnet_metric::{EvaluationConfig, MetricError, MetricsAggregator};
use burn::prelude::*;

fn main() -> Result<(), MetricError> {
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
    Ok(())
}
```

Evaluating saved 8-bit masks without a Burn device:

```rust
use birefnet_metric::{MetricError, Sample};

fn main() -> Result<(), MetricError> {
    let prediction = [0_u8, 64, 255, 255];
    let target = [0_u8, 0, 255, 255];
    let sample = Sample::from_u8(&prediction, &target, [2, 2])?;
    assert_eq!(sample.f_measure(0.3).adaptive, 1.0);
    Ok(())
}
```

## License

MIT OR Apache-2.0
