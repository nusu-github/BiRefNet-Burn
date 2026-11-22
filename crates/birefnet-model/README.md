# birefnet-model

The BiRefNet architecture on Burn: encoder backbone, squeeze stage, bilateral multi-scale decoder,
optional refinement, and validated configuration.

## Features

| Feature | Default | Effect                                                                                |
| ------- | ------- | ------------------------------------------------------------------------------------- |
| `train` | no      | Enables `BiRefNetBatch`, `BiRefNetOutput`, `BiRefNet::forward_step`, and `TrainStep` / `InferenceStep` |

## Core Components

- **Model (`BiRefNet`, `BiRefNetConfig`)**: Encoder backbone (`SwinV1T/S/B/L`, `PvtV2B0/B1/B2/B5`, `ResNet50`, `Vgg16`, `Vgg16Bn`), squeeze block (`BasicDecBlk`, `ResBlk`, `ASPP`, `ASPPDeformable`), bilateral decoder, and optional refinement (`RefUNet`, `Refiner`, `RefinerPVTInChannels4`).
- **Configuration (`ModelConfig`, `TaskConfig`, `BackboneConfig`, `DecoderConfig`, `RefineConfig`)**: Serializable architecture configuration validated by `ModelConfig::validate` returning `ModelError`.
- **Training Support (`train` feature)**: `BiRefNetBatch`, `BiRefNetOutput`, and Burn `TrainStep` / `InferenceStep` implementations.

## Example

```rust
use birefnet_model::{
    Backbone, BackboneConfig, BiRefNetConfig, InterpolationStrategy, ModelConfig, ModelError,
};
use burn::prelude::*;

fn main() -> Result<(), ModelError> {
    let device = Device::flex();
    let config = BiRefNetConfig::new(
        ModelConfig::new(InterpolationStrategy::Bilinear)
            .with_backbone(BackboneConfig::new().with_backbone(Backbone::PvtV2B0)),
    );

    let model = config.init(&device)?;
    let output = model.forward(Tensor::zeros([1, 3, 64, 64], &device))?;
    assert_eq!(output.dims(), [1, 1, 64, 64]);
    Ok(())
}
```

> **Note**: For training on CubeCL backends (`cpu`, `wgpu`, `cuda`, `rocm`), use
> `InterpolationStrategy::Nearest`; only `flex` differentiates `InterpolationStrategy::Bilinear`.
> On Windows, initialize large backbones (such as `SwinV1L`) on a thread with a 64 MiB stack.

## Related Crates

- [`birefnet-backbones`](../birefnet-backbones): encoder backbones (Swin v1, PVTv2, ResNet, VGG).
- [`birefnet-weights`](../birefnet-weights): pretrained model catalog and checkpoint loaders.
- [`birefnet-loss`](../birefnet-loss): multi-scale training losses (`train` feature).

## License

MIT OR Apache-2.0
