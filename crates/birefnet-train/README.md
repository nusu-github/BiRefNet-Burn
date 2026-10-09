# birefnet-train

[![Crates.io](https://img.shields.io/crates/v/birefnet-train.svg)](https://crates.io/crates/birefnet-train)
[![Documentation](https://docs.rs/birefnet-train/badge.svg)](https://docs.rs/birefnet-train)

**Dataset handling and data augmentation for BiRefNet training using the Burn deep learning framework**

## Implemented Features

- ✅ **Complete dataset handling**: DIS5K dataset loading and processing
- ✅ **Comprehensive data augmentation**: All augmentation methods fully implemented
  - Horizontal flip with 50% probability
  - Color enhancement (brightness, contrast, saturation, sharpness)
  - Random rotation (±15 degrees, 20% probability)
  - Pepper noise (0.15% density)
  - Random crop with 10% border
- ✅ **Dynamic sizing**: Multi-resolution training support with background patterns
- ✅ **Burn integration**: Native Burn dataset traits and batch processing
- ✅ **Thread-safe augmentation**: Deterministic random generation with configurable seeds

## Core Components

### Dataset Management

- **`BiRefNetDataset`**: Complete dataset implementation for DIS5K format
- **`BiRefNetItem`**: Individual data item with image and mask
- **`BiRefNetBatch`**: Batched data for training
- **`BiRefNetBatcher`**: Custom batching with augmentation pipeline

### Data Augmentation

- **`ImageAugmentor`**: Main augmentation orchestrator
- **`AugmentationConfig`**: Configuration for all augmentation parameters
- **`AugmentationMethod`**: All augmentation methods implemented:
  - `Flip`: Horizontal flipping
  - `Enhance`: Color/brightness enhancement
  - `Rotate`: Random rotation with interpolation
  - `Pepper`: Salt-and-pepper noise
  - `Crop`: Random cropping with resizing

### Dynamic Sizing

- **`DynamicSizeConfig`**: Multi-resolution training configuration
- **`BackgroundPattern`**: Background fill patterns for resizing

## Usage

```rust
use birefnet_model::{InterpolationStrategy, ModelConfig};
use birefnet_train::{AugmentationConfig, BiRefNetBatcher, BiRefNetDataset};
use burn::{data::dataloader::DataLoaderBuilder, tensor::Device};

// Dataset paths come from the model configuration (`path.data_root_dir`, `task.training_set`)
let model_config = ModelConfig::new(InterpolationStrategy::Bilinear);
let dataset = BiRefNetDataset::new_with_augmentation(
    &model_config,
    "train",
    AugmentationConfig::default(),
)?;

// `Dataset::get` returns `Result` in Burn 0.22: unreadable files surface as errors
// instead of being skipped silently.
let device = Device::flex().autodiff();
let dataloader = DataLoaderBuilder::new(BiRefNetBatcher::new())
    .batch_size(8)
    .shuffle(42)
    .set_device(device) // batches must live on the training device
    .build(dataset);
```

## Dataset Format

Supports DIS5K dataset structure:

```
DIS5K/
├── DIS-TR/
│   ├── im/          # Training images
│   └── gt/          # Ground truth masks
└── DIS-VD/
    ├── im/          # Validation images  
    └── gt/          # Validation masks
```

## Augmentation Details

### Implemented Transformations

1. **Horizontal Flip**: 50% probability, applies to both image and mask
2. **Color Enhancement**: Random brightness/contrast/saturation/sharpness adjustment
3. **Rotation**: ±15 degrees with bilinear interpolation
4. **Pepper Noise**: 0.15% pixel density salt-and-pepper noise
5. **Random Crop**: 10% border crop with intelligent resizing

### Thread Safety

- Deterministic augmentation with configurable seeds
- Thread-safe random number generation
- Reproducible training runs

## Current Status

### ✅ Fully Implemented

- Dataset loading and preprocessing
- All data augmentation methods
- Batch processing and iteration
- Multi-resolution support
- Error handling and validation

### ❌ Not Implemented

- **Training loop**: The actual model training logic is not implemented
- **Optimizer integration**: No optimizer setup or parameter updates
- **Checkpoint management**: No model saving/loading during training
- **Validation loop**: No validation evaluation during training

## Integration

This crate provides the data pipeline for training but requires:

- [`birefnet-model`](../birefnet-model): Model definitions
- [`birefnet-loss`](../birefnet-loss): Loss function computation
- [`birefnet-metric`](../birefnet-metric): Training metrics
- Training loop implementation (not yet available)

## License

MIT OR Apache-2.0