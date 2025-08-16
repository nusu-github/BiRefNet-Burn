# birefnet-model

[![Crates.io](https://img.shields.io/crates/v/birefnet-model.svg)](https://crates.io/crates/birefnet-model)
[![Documentation](https://docs.rs/birefnet-model/badge.svg)](https://docs.rs/birefnet-model)

**Core BiRefNet model implementations using the Burn deep learning framework**

## Implementation

This crate implements the complete BiRefNet architecture including:

- BiRefNet main model with backbone integration
- Multi-scale decoder with lateral connections
- ASPP (Atrous Spatial Pyramid Pooling) module
- Output refinement stages
- Configuration system for different tasks and backbones

### Core Modules

- `BiRefNet`: Main model struct combining all components
- `BiRefNetDecoder`: Multi-scale feature decoder
- `ASPPModule`: Atrous Spatial Pyramid Pooling implementation
- `RefinementModule`: Output post-processing stages

### Supported Components

- Backbone integration (Swin Transformer, PVT v2, ResNet, VGG)
- Task-specific configurations (DIS, COD, HRSOD, General, Matting)
- Multi-scale supervision support
- Optional training mode with loss integration

## License

MIT OR Apache-2.0