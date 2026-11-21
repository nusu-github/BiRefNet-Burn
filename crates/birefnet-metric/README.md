# birefnet-metric

[![Crates.io](https://img.shields.io/crates/v/birefnet-metric.svg)](https://crates.io/crates/birefnet-metric)
[![Documentation](https://docs.rs/birefnet-metric/badge.svg)](https://docs.rs/birefnet-metric)

**Segmentation evaluation metrics and Burn `Metric` adapters for `BiRefNet`**

## Implemented Metrics

- **MAE (`MAEMetric` / `calculate_mae`)**: Mean Absolute Error
- **MSE (`MSEMetric` / `calculate_mse`)**: Mean Squared Error
- **F-measure (`FMeasureMetric` / `calculate_f_measure`)**: Adaptive threshold and 256-threshold precision/recall curves
- **Weighted F-measure (`WeightedFMeasureMetric` / `calculate_weighted_f_measure`)**: Precision/recall weighted by Euclidean distance transform (`birefnet-util`) and error dependencies
- **S-measure (`SMeasureMetric` / `calculate_s_measure`)**: Structural similarity combining object-aware ($S_o$) and region-aware ($S_r$) terms
- **E-measure (`EMeasureMetric` / `calculate_e_measure`)**: Enhanced alignment matrix metric (adaptive and 256-threshold curves)
- **Boundary IoU (`BIoUMetric` / `calculate_biou`)**: Morphological disk boundary IoU across 255 thresholds

All Burn `Metric` adapters consume `[B, 1, H, W]` batches (`MAEInput`, `MSEInput`, `FMeasureInput`, `WeightedFMeasureInput`, `SMeasureInput`, `EMeasureInput`, `BIoUInput`) with exact unweighted sample averaging across batches.

## Batch Evaluation

```rust
use birefnet_metric::{MetricsAggregator, calculate_all_metrics};
use burn::tensor::{Device, Tensor};

let device = Device::flex();
let pred = Tensor::<4>::ones([1, 1, 16, 16], &device);
let gt = Tensor::<4>::ones([1, 1, 16, 16], &device);

let summary = calculate_all_metrics(&pred, &gt, 0.02);
let mut aggregator = MetricsAggregator::new();
aggregator.update(&pred, &gt, 0.02);
let (biou, f_measure, mae) = aggregator.averages();
# let _ = (summary, biou, f_measure, mae);
```

## License

MIT OR Apache-2.0
