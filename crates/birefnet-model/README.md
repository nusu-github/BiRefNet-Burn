# birefnet-model

The BiRefNet architecture on Burn: encoder backbone, squeeze stage, bilateral decoder,
optional refinement, and validated configuration.

## Features

| Feature | Default | Effect                                                                |
| ------- | ------- | --------------------------------------------------------------------- |
| `train` | no      | Enables `BiRefNetLoss`, `BiRefNetBatch`, `BiRefNetOutput`, `TrainStep`, and `InferenceStep` |

## Core Components

- **Model (`BiRefNet`, `BiRefNetConfig`)**: Encoder backbone, squeeze block (`BasicDecBlk`, `ResBlk`, `ASPP`, `ASPPDeformable`), bilateral decoder, and optional refinement (`RefUNet`, `Refiner`, `RefinerPVTInChannels4`).
- **Configuration (`ModelConfig`, `TaskConfig`, `BackboneConfig`, `DecoderConfig`, `RefineConfig`)**: Serializable architecture configuration with logical `ModelConfig::validate` checks returning `ModelError`.
- **Training Support (`train` feature)**: `BiRefNetBatch`, `BiRefNetOutput`, and `TrainStep` / `InferenceStep` implementations.

## Usage

```rust
use birefnet_model::{
    Backbone, BackboneConfig, BiRefNetConfig, InterpolationStrategy, ModelConfig,
};
use burn::prelude::*;

let device = Device::flex();

let config = BiRefNetConfig::new(
    ModelConfig::new(InterpolationStrategy::Bilinear)
        .with_backbone(BackboneConfig::new().with_backbone(Backbone::PvtV2B0)),
);

let model = config.init(&device)?;
let output = model.forward(Tensor::zeros([1, 3, 64, 64], &device))?;
assert_eq!(output.dims(), [1, 1, 64, 64]);
# Ok::<(), birefnet_model::ModelError>(())
```

## Related crates

- `birefnet-backbones`: encoder backbones (Swin v1, PVTv2, ResNet, VGG).
- `birefnet-weights`: pretrained model catalog and checkpoint loaders.
- `birefnet-loss`: multi-scale training losses (`train` feature).

## License

MIT OR Apache-2.0
