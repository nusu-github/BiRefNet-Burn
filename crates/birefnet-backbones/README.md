# birefnet-backbones

[![Crates.io](https://img.shields.io/crates/v/birefnet-backbones.svg)](https://crates.io/crates/birefnet-backbones)
[![Documentation](https://docs.rs/birefnet-backbones/badge.svg)](https://docs.rs/birefnet-backbones)

**Backbone network implementations for BiRefNet using the Burn deep learning framework**

## Implementation

This crate implements feature extraction backbone networks:

### Swin Transformer

- Hierarchical vision transformer with shifted windows
- Variants: Tiny, Small, Base, Large
- Multi-scale feature extraction

### PVT v2 (Pyramid Vision Transformer)

- Pyramid structure with spatial reduction attention
- Variants: B0, B1, B2, B5
- Overlapping patch embedding

### ResNet

- Residual convolutional neural network
- Variant: ResNet-50
- Bottleneck blocks with skip connections

### VGG

- Traditional CNN architecture
- Variant: VGG-16
- Sequential convolutional layers with max pooling

### Core Components

- `SwinTransformer`: Swin transformer implementation
- `PVTv2`: Pyramid Vision Transformer v2
- `ResNet`: ResNet backbone
- `VGG`: VGG backbone
- Feature extraction interfaces for multi-scale outputs

## License

MIT OR Apache-2.0