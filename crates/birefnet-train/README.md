# birefnet-train

[![Crates.io](https://img.shields.io/crates/v/birefnet-train.svg)](https://crates.io/crates/birefnet-train)
[![Documentation](https://docs.rs/birefnet-train/badge.svg)](https://docs.rs/birefnet-train)

**Training infrastructure for BiRefNet models**

## Implementation

This crate implements training capabilities for BiRefNet models:

- Dataset loading and batching for segmentation tasks (basic implementation)
- Training infrastructure with DIS5K dataset support

### Planned Features (Not Yet Implemented)

- Data augmentation pipeline (geometric and photometric transforms)
- Learning rate scheduling (cosine, polynomial, step)
- Multi-GPU distributed training support
- Advanced checkpointing and model persistence
- Comprehensive training metrics and logging integration

### Core Components

- Dataset interfaces for DIS5K segmentation data
- Basic data batching and loading
- Foundation for training pipeline integration

## License

MIT OR Apache-2.0