# birefnet-inference

[![Crates.io](https://img.shields.io/crates/v/birefnet-inference.svg)](https://crates.io/crates/birefnet-inference)
[![Documentation](https://docs.rs/birefnet-inference/badge.svg)](https://docs.rs/birefnet-inference)

**High-level inference pipeline and mask postprocessing for `BiRefNet` on Burn**

## Core Components

- **`Segmenter` & `SegmenterConfig`**: End-to-end segmentation runner supporting tensor (`[1, 3, H, W]`), `DynamicImage`, and image file inputs (`segment_tensor`, `segment_image`, `segment_file`), with ImageNet normalization, bicubic resizing via native Burn interpolation, sigmoid activation, optional mask postprocessing, and optional multi-pass blur-fusion foreground refinement.
- **`SegmentationOutput`**: Holds predicted mask `[1, 1, H, W]`, optional refined foreground `[1, 3, H, W]`, original `(height, width)`, and export helpers (`to_rgba_image`, `to_mask_image`).
- **`InferenceError`**: Typed error with `is_model`, `is_util`, `is_image`, `is_io`, `is_invalid_input`, and `is_tensor_read` predicates.

## Postprocessing Functions

- **`apply_threshold`**: Binary mask thresholding (`mask > threshold`).
- **`gaussian_blur`**: 2D Gaussian smoothing via `birefnet_util::gaussian_filter_matlab`.
- **`morphological_opening` / `morphological_closing`**: Disk structuring element morphology via `birefnet-util`.
- **`remove_small_components`**: 4-connected foreground component area filtering (`< min_size`).
- **`fill_holes`**: Border-seeded flood fill for enclosed background holes.
- **`postprocess_mask`**: Sequential pipeline driven by `PostprocessConfig`.
- **`resize_tensor` / `mask_to_u8`**: Native Burn `interpolate` resizing and 8-bit grayscale export.

## Usage

```rust,no_run
use birefnet_inference::{PostprocessConfig, Segmenter, SegmenterConfig};
use birefnet_model::BiRefNet;
use birefnet_weights::{BiRefNetWeightLoading, ManagedModel};
use burn::tensor::Device;

let device = Device::flex();
let managed = ManagedModel::from_local("custom", "model.bpk", None);
let model = managed.build_config().init(&device)?.load_weights("model.bpk", &device)?;

let segmenter = Segmenter::new(
    model,
    SegmenterConfig {
        postprocess: PostprocessConfig {
            threshold: Some(0.5),
            fill_holes: true,
            ..PostprocessConfig::default()
        },
        ..SegmenterConfig::default()
    },
);

let output = segmenter.segment_file("input.png", &device)?;
output.to_rgba_image()?.save("output_mask.png")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Integration

- [`birefnet-model`](../birefnet-model): `BiRefNet` model forward pass
- [`birefnet-util`](../birefnet-util): Image normalization, codec I/O (`image-io`), morphology, filters, and foreground refinement

## License

MIT OR Apache-2.0
