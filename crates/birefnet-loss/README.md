# birefnet-loss

[![Crates.io](https://img.shields.io/crates/v/birefnet-loss.svg)](https://crates.io/crates/birefnet-loss)
[![Documentation](https://docs.rs/birefnet-loss/badge.svg)](https://docs.rs/birefnet-loss)

**Comprehensive loss function implementations for BiRefNet training using the Burn deep learning framework**

## Implemented Features

- ✅ **Pixel-wise losses**: Binary Cross-Entropy (BCE) with numerical stability
- ✅ **Region-based losses**: IoU loss with smooth approximation
- ✅ **Structural losses**: SSIM loss for spatial structure preservation
- ✅ **Contour losses**: Boundary-aware loss functions
- ✅ **Auxiliary losses**: Classification loss for multi-task learning
- ✅ **Regularization**: Threshold regularization for binary segmentation
- ✅ **Patch-based losses**: Patch IoU for local consistency
- ✅ **Mean Absolute Error**: MAE loss for regression tasks
- ✅ **Combined loss**: BiRefNet-specific weighted loss combination

## Core Loss Functions

### Primary Losses

- **`PixelLoss`**: Binary cross-entropy with sigmoid activation
- **`IoULoss`**: Intersection over Union loss with smooth gradients
- **`SSIMLoss`**: Structural Similarity Index loss
- **`StructureLoss`**: Multi-scale structure preservation

### Auxiliary Losses

- **`ContourLoss`**: Edge-aware boundary loss
- **`ClassificationLoss`**: Auxiliary classification for multi-task learning
- **`ThresholdRegularizationLoss`**: Encourages binary predictions
- **`PatchIoULoss`**: Local patch-based IoU computation
- **`MAELoss`**: Mean absolute error

### Combined Loss

- **`BiRefNetLoss`**: Weighted combination of all loss components
- **`BiRefNetLossConfig`**: Configuration for loss weights and parameters

## Usage

```rust
use birefnet_loss::BiRefNetLossConfig;
use burn::tensor::{Device, Tensor};

let device = Device::flex();

// Create loss configuration
let loss_fn = BiRefNetLossConfig::new()
    .with_iou_weight(1.0)
    .with_ssim_weight(1.0)
    .init(&device);

// Multi-scale predictions (logits) and ground-truth masks, `[batch, 1, height, width]`
let predictions = vec![Tensor::<4>::zeros([1, 1, 64, 64], &device)];
let targets = Tensor::<4>::ones([1, 1, 64, 64], &device);

// Total loss, plus a per-component breakdown
let total_loss = loss_fn.forward(predictions.clone(), targets.clone())?;
let (_, components) = loss_fn.forward_detailed(predictions, targets)?;
```

## Loss Components Details

### Pixel-wise Loss (BCE)

- Robust binary cross-entropy implementation
- Numerical stability with clipping
- Handles class imbalance

### IoU Loss

- Smooth intersection over union
- Differentiable approximation
- Handles empty predictions gracefully

### SSIM Loss

- Multi-scale structural similarity
- Preserves spatial relationships
- Complementary to pixel-wise losses

### Structure Loss

- Hierarchical feature comparison
- Multi-resolution consistency
- Edge-preserving properties

## Configuration Options

- **Loss weights**: Individual weighting for each loss component
- **Multi-scale supervision**: Loss computation at multiple resolutions
- **Auxiliary classification**: Optional classification head training
- **Threshold parameters**: Regularization strength control

## PyTorch Weight Loading

- **Weight initialization**: Supports loading from PyTorch loss implementations
- **Configuration**: Similar parameter structure to PyTorch losses

## License

MIT OR Apache-2.0