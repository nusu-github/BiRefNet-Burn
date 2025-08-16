# birefnet-train

[![Crates.io](https://img.shields.io/crates/v/birefnet-train.svg)](https://crates.io/crates/birefnet-train)
[![Documentation](https://docs.rs/birefnet-train/badge.svg)](https://docs.rs/birefnet-train)

**Training infrastructure for BiRefNet models**

## Implementation

This crate implements training capabilities for BiRefNet models:

- `BiRefNetTrainer`: Main training engine
- Dataset loading and batching for segmentation tasks
- Data augmentation pipeline (geometric and photometric transforms)
- Learning rate scheduling (cosine, polynomial, step)
- Multi-GPU distributed training support
- Checkpointing and model persistence
- Training metrics and logging integration

### Core Components

- Training configuration and management
- Dataset interfaces for segmentation data
- Augmentation transforms and pipelines
- Optimization and scheduling
- Distributed training coordination
- Checkpoint management and resuming

## License

MIT OR Apache-2.0