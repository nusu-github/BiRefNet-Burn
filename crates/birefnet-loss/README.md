# birefnet-loss

[![Crates.io](https://img.shields.io/crates/v/birefnet-loss.svg)](https://crates.io/crates/birefnet-loss)
[![Documentation](https://docs.rs/birefnet-loss/badge.svg)](https://docs.rs/birefnet-loss)

**Loss functions for deep learning segmentation tasks using the Burn framework**

## Implementation

This crate implements comprehensive loss functions for image segmentation:

### Pixel-wise Losses

- `BinaryCrossEntropyLoss`: Standard BCE with optional class weighting
- `FocalLoss`: Addresses class imbalance by focusing on hard examples
- `DiceLoss`: Direct optimization of Dice coefficient

### Region-based Losses

- `IoULoss`: Intersection over Union loss
- `PatchIoULoss`: IoU computed on local patches

### Structural Losses

- `SSIMLoss`: Structural Similarity Index loss
- `StructureLoss`: Preserves structural information using gradients
- `ContourLoss`: Emphasizes boundary accuracy

### Advanced Losses

- `ThresholdRegularization`: Encourages sharp decision boundaries
- `BiRefNetLoss`: Composite loss combining multiple loss types
- Multi-scale supervision with weighted loss combination

### Core Components

- Individual loss function implementations
- Composite loss with configurable weights
- Multi-scale supervision support
- Numerical stability and efficient computation

## License

MIT OR Apache-2.0