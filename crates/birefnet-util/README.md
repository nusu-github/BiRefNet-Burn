# birefnet-util

[![Crates.io](https://img.shields.io/crates/v/birefnet-util.svg)](https://crates.io/crates/birefnet-util)
[![Documentation](https://docs.rs/birefnet-util/badge.svg)](https://docs.rs/birefnet-util)

**Tensor image utilities, Euclidean distance transforms, morphology, spatial filters, and foreground refinement for `BiRefNet`**

## Features

| Feature    | Default | Effect                                                                    |
| ---------- | ------- | ------------------------------------------------------------------------- |
| `image-io` | no      | Enables image codec helpers (`load_image`, `dynamic_image_to_tensor`, `tensor_to_dynamic_image`, `is_supported_image_format`, etc.) powered by `image` |

With default features disabled, `birefnet-util` compiles only pure Burn tensor operations with zero image-codec dependencies.

## Core Modules

- **Normalization & masking**: `apply_imagenet_normalization`, `apply_mask`, `IMAGENET_MEAN`, `IMAGENET_STD`.
- **Foreground refinement**: Multi-pass blur-fusion color refinement (`refine_foreground_core`, `refine_foreground`, `refine_foreground_batch`).
- **Distance transform**: $O(H \cdot W)$ Felzenszwalb-Huttenlocher exact squared Euclidean distance transform (`euclidean_distance_transform`, `euclidean_distance_transform_with_coords`).
- **Morphology**: Binary 2D `StructuringElement` (`rectangle`, `disk`) and edge-replicated `erosion`, `dilation`, `opening`, `closing`, `gradient`.
- **Spatial filters**: `matlab_gaussian_2d`, `custom_filter`, `gaussian_filter_matlab`.
- **Array statistics**: `histogram`.
- **Image codecs (`image-io` feature)**: `load_image`, `load_image_from_reader`, `decode_image_bytes`, `load_image_batch`, `dynamic_image_to_tensor`, `tensor_to_dynamic_image`, `resize_image_file`, `from_raw_pixels`, `is_extension_supported`, `is_supported_image_format`, `common_image_extensions`, `supported_formats`.

## Usage

```rust
use birefnet_util::{UtilError, apply_imagenet_normalization, gaussian_filter_matlab};
use burn::tensor::{Device, Tensor};

let device = Device::flex();
let image = Tensor::<4>::ones([1, 3, 16, 16], &device) * 0.5;
let normalized = apply_imagenet_normalization(image)?;
let smoothed = gaussian_filter_matlab(normalized, (3, 3), 1.0);
assert_eq!(smoothed.dims(), [1, 3, 16, 16]);
# Ok::<(), UtilError>(())
```

## License

MIT OR Apache-2.0
