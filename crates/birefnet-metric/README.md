# birefnet-metric

[![Crates.io](https://img.shields.io/crates/v/birefnet-metric.svg)](https://crates.io/crates/birefnet-metric)
[![Documentation](https://docs.rs/birefnet-metric/badge.svg)](https://docs.rs/birefnet-metric)

**Evaluation metrics for image segmentation using the Burn deep learning framework**

## Implemented Features

### ✅ Fully Implemented (Working Metrics)

- **MSE**: Mean Squared Error for pixel-wise accuracy
- **BIoU**: Boundary IoU for edge accuracy assessment
- **Weighted F-measure**: Precision-recall based metric with weighting
- **Metric Aggregation**: Batch processing and result aggregation

### ❌ Not Implemented (Placeholder Functions Return 0.0)

- **IoU**: Returns fixed 0.0 value (needs proper intersection/union computation)
- **F-measure**: Returns fixed 0.0 value (needs precision/recall implementation)
- **MAE**: Returns fixed 0.0 value (needs proper absolute error computation)
- **S-measure**: Returns fixed 0.0 value (structural similarity not implemented)
- **E-measure**: Returns fixed 0.0 value (enhanced alignment not implemented)

## Core Components

### Working Metrics

- **`calculate_mse`**: Mean squared error computation
- **`calculate_biou`**: Boundary IoU with morphological operations
- **`calculate_weighted_f_measure`**: Weighted F-measure with error dependency
- **`MetricAggregator`**: Batch metric computation and aggregation

### Metrics with Placeholder Implementation

- **`calculate_iou`**: Returns 0.0 (needs proper intersection/union computation)
- **`calculate_f_measure`**: Returns 0.0 (needs precision/recall implementation)
- **`calculate_mae`**: Returns 0.0 (needs proper absolute error computation)
- **`calculate_s_measure`**: Returns 0.0 (structural similarity not implemented)
- **`calculate_e_measure`**: Returns 0.0 (enhanced alignment not implemented)

## Usage

```rust
use birefnet_metric::{calculate_mse, calculate_biou, calculate_weighted_f_measure};

// Working metrics
let mse = calculate_mse(predictions, targets);
let biou = calculate_biou(predictions, targets, threshold);
let wfm = calculate_weighted_f_measure(predictions, targets, beta);

// Placeholder metrics (return 0.0)
let iou = calculate_iou(predictions, targets);     // TODO: Implement
let f_measure = calculate_f_measure(predictions, targets, beta); // TODO: Implement
let mae = calculate_mae(predictions, targets);     // TODO: Implement
```

## Batch Processing

```rust
use birefnet_metric::utils::calculate_batch_metrics;

// Process entire batch
let metrics = calculate_batch_metrics(batch_predictions, batch_targets);

// Access working metrics
println!("MSE: {}", metrics.mse);
println!("BIoU: {}", metrics.biou);
println!("Weighted F-measure: {}", metrics.weighted_f_measure);

// Placeholder values (always 0.0)
println!("IoU: {}", metrics.iou);     // 0.0
println!("F-measure: {}", metrics.f_measure);  // 0.0
println!("MAE: {}", metrics.mae);     // 0.0
```

## Implementation Status

### Fully Working

1. **MSE**: Complete pixel-wise error computation
2. **BIoU**: Complete boundary IoU with morphological operations
3. **Weighted F-measure**: Complete implementation with error dependency mapping

### Needs Implementation

1. **IoU**: Proper intersection over union calculation
2. **F-measure**: Precision and recall based F-measure
3. **MAE**: Mean absolute error computation
4. **S-measure**: Structural similarity measure
5. **E-measure**: Enhanced alignment measure

## Current Limitations

- Several core metrics return 0.0 placeholders
- Advanced metrics (S-measure, E-measure) are not implemented
- Some functions have dimension mismatch issues noted in comments
- Full PyTorch metric parity not yet achieved

## Future Work

- Complete IoU, F-measure, and MAE implementations
- Implement S-measure and E-measure calculations
- Fix tensor dimension handling issues
- Add comprehensive metric validation tests

## License

MIT OR Apache-2.0