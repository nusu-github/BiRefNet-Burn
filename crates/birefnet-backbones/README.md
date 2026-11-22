# birefnet-backbones

Multi-scale encoder backbones for BiRefNet on Burn.

| Family | Variants | Configuration |
| --- | --- | --- |
| Swin Transformer v1 | [`SwinVariant`]: T, S, B, L | [`SwinTransformerConfig`] |
| Pyramid Vision Transformer v2 | [`PvtV2Variant`]: B0–B5 | [`PvtV2Config`] |
| ResNet | [`ResNetVariant`]: 18, 34, 50, 101, 152 | [`ResNetConfig`] |
| VGG | [`VggVariant`]: 16, 16-BN | [`VggConfig`] |

Every backbone implements [`Backbone`] and returns four feature maps at strides
`4, 8, 16, 32`, shallowest first. [`BackboneType`] selects a preset and builds a
[`BackboneModel`]; configuration problems surface as [`BackboneError`].

## Examples

```rust
use birefnet_backbones::{Backbone, BackboneType, PvtV2Variant};
use burn::prelude::*;

let device = Device::flex();
let backbone = BackboneType::PvtV2(PvtV2Variant::B0).init(&device)?;
let features = backbone.forward(Tensor::zeros([1, 3, 64, 64], &device));

assert_eq!(features[0].dims(), [1, 32, 16, 16]);
assert_eq!(features[3].dims()[1], backbone.output_channels()[3]);
# Ok::<(), birefnet_backbones::BackboneError>(())
```

PVTv2 accepts any number of input channels, which the refiner uses for RGB + mask input:

```rust
use birefnet_backbones::{PvtV2Config, PvtV2Variant};
use burn::prelude::*;

let device = Device::flex();
let pvt = PvtV2Config::from_variant(PvtV2Variant::B0, 4).init(&device)?;
let [stride4, ..] = pvt.forward(Tensor::zeros([1, 4, 64, 64], &device));
assert_eq!(stride4.dims(), [1, 32, 16, 16]);
# Ok::<(), birefnet_backbones::BackboneError>(())
```

## License

MIT OR Apache-2.0
