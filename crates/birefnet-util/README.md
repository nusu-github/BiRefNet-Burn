# birefnet-util

Image and mask utilities for BiRefNet on Burn: normalization, exact distance transforms, binary
morphology, Gaussian filtering, foreground color refinement, and optional image codecs.

## Features

| Feature    | Default | Effect                                                                    |
| ---------- | ------- | ------------------------------------------------------------------------- |
| `image-io` | no      | Enables the `image` module (PNG, JPEG, BMP, GIF, TIFF, WebP via `image`)  |

Without `image-io`, this crate depends only on Burn.

## Modules

| Module       | Contents                                                                                      |
| ------------ | --------------------------------------------------------------------------------------------- |
| `normalize`  | ImageNet standardization and alpha masking                                                    |
| `distance`   | Exact Euclidean (with optional nearest-site indices) and Manhattan distance transforms        |
| `morphology` | `StructuringElement` plus erosion, dilation, opening, closing, and morphological gradient     |
| `filters`    | MATLAB-compatible Gaussian kernels and edge-replicated filtering                              |
| `foreground` | Two-pass blur-fusion foreground color estimation                                              |
| `image`      | Decoding, encoding, and tensor conversion (`image-io` feature)                                |

Every tensor operation preserves the input float dtype.

## Examples

ImageNet normalization and Gaussian smoothing:

```rust
use birefnet_util::{
    UtilError, filters::gaussian_filter_matlab, normalize::apply_imagenet_normalization,
};
use burn::prelude::*;

fn main() -> Result<(), UtilError> {
    let device = Device::flex();
    let image = Tensor::<4>::full([1, 3, 16, 16], 0.5, &device);
    let normalized = apply_imagenet_normalization(image)?;
    let smoothed = gaussian_filter_matlab(normalized, (3, 3), 1.0);
    assert_eq!(smoothed.dims(), [1, 3, 16, 16]);
    Ok(())
}
```

Morphology and distance transforms (values `>= 0.5` are treated as foreground):

```rust
use birefnet_util::{
    UtilError,
    distance::euclidean_distance_transform,
    morphology::{StructuringElement, dilation},
};
use burn::prelude::*;

fn main() -> Result<(), UtilError> {
    let device = Device::flex();
    let mut dot = [[0.0_f32; 5]; 5];
    dot[2][2] = 1.0;
    let mask = Tensor::<4>::from_floats([[dot]], &device);

    let cross = dilation(mask.clone(), &StructuringElement::disk(1));
    assert_eq!(cross.sum().into_scalar::<f32>(), 5.0);

    // Distance from every pixel to the nearest pixel `< 0.5` (background).
    let inside = euclidean_distance_transform(&mask)?;
    assert_eq!(inside.max().into_scalar::<f32>(), 1.0);
    Ok(())
}
```

## License

MIT OR Apache-2.0
