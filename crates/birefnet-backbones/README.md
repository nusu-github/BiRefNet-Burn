# birefnet-backbones

[![Crates.io](https://img.shields.io/crates/v/birefnet-backbones.svg)](https://crates.io/crates/birefnet-backbones)
[![Documentation](https://docs.rs/birefnet-backbones/badge.svg)](https://docs.rs/birefnet-backbones)

**Backbone network implementations for BiRefNet using the Burn deep learning framework**

## Implemented Features

- ✅ **Swin Transformer v1**: Complete hierarchical vision transformer implementation
  - Variants: Tiny, Small, Base, Large
  - Shifted windows attention mechanism
  - Multi-scale feature extraction
  - PyTorch weight compatibility

- ✅ **ResNet**: Residual convolutional neural network
  - ResNet-50 implementation
  - Bottleneck blocks with skip connections
  - Batch normalization and ReLU activation

- ✅ **VGG**: Traditional CNN architecture
  - VGG-16 implementation
  - Sequential convolutional layers
  - Max pooling operations

- ✅ **PVT v2**: Pyramid Vision Transformer v2
  - Variants: B0, B1, B2, B5
  - Spatial reduction attention
  - Overlapping patch embedding

## Core Components

### Backbone Trait

```rust
pub trait Backbone<B: Backend> {
    fn forward(&self, x: Tensor<B, 4>) -> Vec<Tensor<B, 4>>;
}
```

### Implemented Backbones

- **`SwinTransformer`**: Full Swin Transformer implementation with all variants
- **`ResNetBackbone`**: ResNet-50 with multi-scale feature extraction
- **`VGGBackbone`**: VGG-16 with feature layer extraction
- **`PyramidVisionTransformerImpr`**: PVT v2 with all variants

### Factory Function

- **`create_backbone`**: Type-safe backbone creation from configuration

## Usage

```rust
use birefnet_backbones::{create_backbone, BackboneType, SwinVariant};

// Create Swin Transformer backbone
let backbone_type = BackboneType::SwinV1(SwinVariant::Tiny);
let backbone = create_backbone(backbone_type, &device);

// Forward pass - returns multi-scale features
let features = backbone.forward(input_tensor);
// features[0]: 1/4 resolution
// features[1]: 1/8 resolution  
// features[2]: 1/16 resolution
// features[3]: 1/32 resolution
```

## Supported Variants

### Swin Transformer v1

- **Tiny**: 28M parameters, suitable for mobile/edge deployment
- **Small**: 50M parameters, balanced performance
- **Base**: 88M parameters, high accuracy
- **Large**: 197M parameters, maximum accuracy

### PVT v2

- **B0**: Ultra-lightweight variant
- **B1**: Mobile-friendly variant
- **B2**: Standard variant
- **B5**: Large variant

### ResNet & VGG

- **ResNet-50**: Industry standard CNN backbone
- **VGG-16**: Classic CNN architecture

## Features

- **Multi-scale output**: All backbones output 4 different resolution levels
- **PyTorch weight loading**: Direct weight loading from official PyTorch models
- **Tensor operations**: Built on Burn framework tensor operations
- **Type safety**: Compile-time backbone selection and validation

## License

MIT OR Apache-2.0