# birefnet-model

[![Crates.io](https://img.shields.io/crates/v/birefnet-model.svg)](https://crates.io/crates/birefnet-model)
[![Documentation](https://docs.rs/birefnet-model/badge.svg)](https://docs.rs/birefnet-model)

**Core BiRefNet model implementations using the Burn deep learning framework**

## Implemented Features

- ✅ **Complete BiRefNet architecture**: Full bilateral reference network implementation
- ✅ **Multi-scale decoder**: Progressive refinement with lateral connections
- ✅ **ASPP module**: Atrous Spatial Pyramid Pooling for multi-scale feature extraction
- ✅ **Decoder blocks**: BasicDecBlk, ResBlk, and ASPP variants
- ✅ **Refinement modules**: RefUNet and Refiner implementations
- ✅ **Backbone integration**: Seamless integration with all backbone networks
- ✅ **Configuration system**: Type-safe model configuration with validation
- ✅ **PyTorch compatibility**: Direct weight loading from PyTorch checkpoints

## Core Components

### Main Model

- **`BiRefNet`**: Complete model implementation with forward pass
- **`BiRefNetConfig`**: Configuration builder for different tasks
- **`BiRefNetDecoder`**: Multi-scale feature decoder with attention

### Decoder Modules

- **`BasicDecBlk`**: Basic decoder block with upsampling
- **`ResBlk`**: Residual decoder block
- **`ASPP`**: Atrous Spatial Pyramid Pooling
- **`ASPPDeformable`**: Deformable convolution variant

### Refinement

- **`RefUNet`**: UNet-based refinement module
- **`Refiner`**: Lightweight refinement module
- **`RefinerPVTInChannels4`**: PVT-specific refinement

## Usage

```rust
use birefnet_model::{BiRefNet, BiRefNetConfig};
use birefnet_backbones::BackboneType;

// Create model configuration
let config = BiRefNetConfig::new()
.with_backbone(BackboneType::SwinV1Tiny)
.with_task("General");

// Initialize model
let model: BiRefNet<Backend> = config.init( & device) ?;

// Forward pass
let output = model.forward(input_tensor) ?;
```

## Supported Configurations

- **Tasks**: DIS, COD, HRSOD, General, Matting
- **Backbones**: Swin Transformer v1, ResNet, VGG, PVT v2
- **Multi-scale supervision**: Optional auxiliary outputs
- **Context levels**: 0-3 context enhancement levels

## Model Components Integration

This crate integrates with:

- [`birefnet-backbones`](../birefnet-backbones): Backbone networks
- [`birefnet-loss`](../birefnet-loss): Training loss functions (when `train` feature enabled)
- [`birefnet-util`](../birefnet-util): Weight loading utilities

## License

MIT OR Apache-2.0