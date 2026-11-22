# birefnet-inference

End-to-end BiRefNet inference on Burn: ImageNet normalization, spatial resizing, forward pass,
mask postprocessing, and blur-fusion foreground color refinement.

- `Segmenter` runs a `birefnet_model::BiRefNet` on tensors, decoded `DynamicImage`s, encoded
  bytes, buffered readers, or file paths, returning a `SegmentationOutput` at the source
  resolution (alpha mask, optional refined RGB foreground, and RGBA / grayscale export).
- `SegmenterConfig` and `postprocess::PostprocessConfig` are validated, `Copy` builders.
  `SegmenterConfig::with_input_size` takes `[height, width]`.
- `postprocess` exposes the individual mask operations (thresholding, Gaussian blur, disk
  opening / closing, 4-connected small-component removal, and hole filling).

## Example

```rust
use birefnet_inference::{Segmenter, SegmenterConfig, postprocess::PostprocessConfig};
use birefnet_model::{
    Backbone, BackboneConfig, BiRefNetConfig, DecoderConfig, InterpolationStrategy, ModelConfig,
};
use burn::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::flex();

    // Small randomly initialized model for illustration; load pretrained weights via `birefnet-weights`.
    let model = BiRefNetConfig::new(
        ModelConfig::new(InterpolationStrategy::Bilinear)
            .with_backbone(BackboneConfig::new().with_backbone(Backbone::PvtV2B0))
            .with_decoder(DecoderConfig::new().with_dec_ipt_split(false)),
    )
    .init(&device)?;

    let config = SegmenterConfig::new()
        .with_input_size([64, 64])?
        .with_foreground_refinement(None)
        .with_postprocess(PostprocessConfig::new().with_threshold(0.5)?.with_fill_holes(true));
    let segmenter = Segmenter::new(&model, config);

    let image = Tensor::<4>::full([1, 3, 48, 80], 0.5, &device);
    let output = segmenter.segment_tensor(image)?;
    assert_eq!(output.original_size(), [48, 80]);
    assert_eq!(output.to_mask_image()?.dimensions(), (80, 48));
    Ok(())
}
```

> **Windows note**: Model initialization and forward passes recurse deeply; on Windows (1 MiB
> default main-thread stack), run them on a thread with a 64 MiB stack or via `stacker::grow`.

## License

MIT OR Apache-2.0
