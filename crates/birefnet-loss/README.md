# birefnet-loss

[![Crates.io](https://img.shields.io/crates/v/birefnet-loss.svg)](https://crates.io/crates/birefnet-loss)
[![Documentation](https://docs.rs/birefnet-loss/badge.svg)](https://docs.rs/birefnet-loss)

**Segmentation loss functions and combined `BiRefNetLoss` on Burn**

## Core Components

- **Primary Losses**: `PixelLoss`, `IoULoss`, `SSIMLoss`, `StructureLoss`
- **Auxiliary & Regularization Losses**: `ContourLoss`, `ClassificationLoss`, `ThresholdRegularizationLoss`, `PatchIoULoss`, `MAELoss`
- **Shared Operations**: Numerically stable `ops::bce_with_logits`
- **Combined Objective**: `BiRefNetLoss` & `BiRefNetLossConfig` returning typed `LossError` values

## Usage

```rust
use birefnet_loss::BiRefNetLossConfig;
use burn::tensor::{Device, Tensor};

let device = Device::flex();
let loss_fn = BiRefNetLossConfig::new()
    .with_iou_weight(1.0)
    .with_ssim_weight(1.0)
    .init(&device);

let predictions = vec![Tensor::<4>::zeros([1, 1, 64, 64], &device)];
let targets = Tensor::<4>::ones([1, 1, 64, 64], &device);

let total = loss_fn.forward(predictions.clone(), targets.clone())?;
let (_, components) = loss_fn.forward_detailed(predictions, targets)?;
# let _ = (total, components);
# Ok::<(), birefnet_loss::LossError>(())
```

## License

MIT OR Apache-2.0
