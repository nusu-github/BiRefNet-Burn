# birefnet-backbones

[![Crates.io](https://img.shields.io/crates/v/birefnet-backbones.svg)](https://crates.io/crates/birefnet-backbones)
[![Documentation](https://docs.rs/birefnet-backbones/badge.svg)](https://docs.rs/birefnet-backbones)

**Encoder backbone networks (`SwinV1`, `PVTv2`, `ResNet`, `VGG`) for `BiRefNet` on Burn**

## Implemented Backbones

- **Swin Transformer v1 (`SwinVariant`)**: `Tiny`, `Small`, `Base`, `Large` (`WindowSize12` defaults in BiRefNet)
- **PVT v2 (`PvtV2Variant`)**: `B0`, `B1`, `B2`, `B5` Pyramid Vision Transformers
- **ResNet (`ResNetVariant`)**: `ResNet18`, `ResNet50`
- **VGG (`VggVariant`)**: `Vgg16`, `Vgg16Bn`

All backbones implement the `Backbone` trait, producing 4 hierarchical feature maps `[1/4, 1/8, 1/16, 1/32]` together with typed configuration errors via `BackboneError`.

## Usage

```rust
use birefnet_backbones::{Backbone, BackboneType, PvtV2Variant};
use burn::tensor::{Device, Tensor};

let device = Device::flex();
let backbone = BackboneType::PvtV2(PvtV2Variant::B0).init(&device)?;
let features = backbone.forward(Tensor::zeros([1, 3, 64, 64], &device));
assert_eq!(features[0].dims()[1], backbone.output_channels()[0]);
# Ok::<(), birefnet_backbones::BackboneError>(())
```

## License

MIT OR Apache-2.0
