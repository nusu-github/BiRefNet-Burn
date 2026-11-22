# birefnet-loss

Segmentation losses and the combined multi-scale BiRefNet training objective on Burn.

| Loss | Input | Per-sample value |
| --- | --- | --- |
| [`IouLoss`] | probabilities | `1 - (inter + eps) / (union + eps)` |
| [`PatchIouLoss`] | probabilities | sum of [`IouLoss`] over a grid of patches |
| [`SsimLoss`] | probabilities | `(1 - SSIM) / 2` with an 11×11 Gaussian window |
| [`StructureLoss`] | logits | edge-weighted BCE + edge-weighted IoU |
| [`ContourLoss`] | probabilities | curve length + region terms |
| [`MaeLoss`] | probabilities | mean absolute error |
| [`ThresholdRegularizationLoss`] | probabilities | `2 p (1 - p)` |
| [`ClassificationLoss`] | class logits | cross-entropy summed over levels |

[`PixLoss`] sums the enabled components (weights from [`LossWeightsConfig`]) over every
prediction scale, and [`BiRefNetLoss`] adds the optional auxiliary classification term.
Every `forward` takes a `burn::nn::loss::Reduction`; `forward_no_reduction` returns one value
per sample so `Mean`, `Sum`, and `BatchMean` behave as in `burn::nn::loss`.

## Examples

```rust
use birefnet_loss::BiRefNetLossConfig;
use burn::prelude::*;

let device = Device::flex();
let loss_fn = BiRefNetLossConfig::general_optimized().init();

// One prediction scale of raw logits and a binary target mask.
let predictions = [Tensor::<4>::zeros([1, 1, 64, 64], &device)];
let targets = Tensor::<4>::ones([1, 1, 64, 64], &device);

let total = loss_fn.forward(&predictions, &targets)?;
assert!(total.into_scalar::<f32>() > 0.0);

// The breakdown is read back from the device in a single transaction.
let (_, breakdown) = loss_fn.forward_detailed(&predictions, &targets)?;
let sum: f64 = breakdown.components().iter().map(|(_, value)| value).sum();
assert!((sum - breakdown.total).abs() < 1e-3);
# Ok::<(), birefnet_loss::LossError>(())
```

## License

MIT OR Apache-2.0
