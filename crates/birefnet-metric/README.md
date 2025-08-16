# birefnet-metric

[![Crates.io](https://img.shields.io/crates/v/birefnet-metric.svg)](https://crates.io/crates/birefnet-metric)
[![Documentation](https://docs.rs/birefnet-metric/badge.svg)](https://docs.rs/birefnet-metric)

**Evaluation metrics for BiRefNet implemented in Rust using Burn framework**

## Implementation

This crate implements evaluation metrics for segmentation tasks:

### Implemented Metrics

- `FMeasure`: Adaptive F-measure with precision-recall curves
- `MAE`: Mean Absolute Error with data preprocessing
- `MSE`: Mean Squared Error with normalization
- `BIoU`: Boundary IoU replacing standard IoU
- `WeightedFMeasure`: Distance-weighted F-measure for boundary evaluation
- `LossMetric`: Simple loss value tracking

### Pending Implementation

- `SMeasure`: Structure measure for object-aware evaluation
- `EMeasure`: Enhanced-alignment measure
- Additional boundary-based metrics

### Core Components

- Unified 4D tensor input handling `[batch, channel, height, width]`
- Data preprocessing following Python reference implementation
- Burn framework integration with standard metric traits
- Batch processing utilities for multiple metrics

## Status

This crate implements evaluation functionality and is not part of the core model pipeline. Many metrics are incomplete with placeholder implementations and have not been tested. Most functionality is not yet working properly.

## License

MIT OR Apache-2.0