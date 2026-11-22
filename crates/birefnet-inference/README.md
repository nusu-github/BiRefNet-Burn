# birefnet-inference

End-to-end BiRefNet inference: ImageNet normalization, resizing, the forward pass, mask
postprocessing, and blur-fusion foreground refinement.

- [`Segmenter`] runs a [`BiRefNet`](birefnet_model::BiRefNet) on tensors, decoded images,
  encoded bytes, readers, or files and returns a [`SegmentationOutput`] at the source
  resolution (mask, optional refined foreground, RGBA / grayscale export).
- [`SegmenterConfig`] and [`postprocess::PostprocessConfig`] are validated, `Copy`
  builders.
- [`postprocess`] exposes the individual mask operations (threshold, Gaussian blur, disk
  opening / closing, small-component removal, hole filling).

## Examples

```rust
use birefnet_inference::{Segmenter, SegmenterConfig, postprocess::PostprocessConfig};
use birefnet_model::{
    Backbone, BackboneConfig, BiRefNetConfig, DecoderConfig, InterpolationStrategy, ModelConfig,
};
use burn::prelude::*;

# // Building a model recurses deeply; doctests run on a 1 MiB main thread on Windows.
# std::thread::Builder::new().stack_size(64 << 20).spawn(|| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
let device = Device::flex();
// A small randomly initialized model; real use loads pretrained weights (`birefnet-weights`).
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
# Ok(())
# })?.join().unwrap()?;
# Ok::<(), Box<dyn std::error::Error + Send + Sync>>(())
```

## License

MIT OR Apache-2.0
